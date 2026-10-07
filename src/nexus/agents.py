"""Building blocks shared by nexus's Deep Agents."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from deepagents.backends.protocol import BackendProtocol
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    ModelRequest,
    ToolCallRequest,
)
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage

from nexus.models import cost_usd
from nexus.staging import StagedBackend

log = logging.getLogger("nexus.agents")

WRITE_TOOLS = frozenset({"write_file", "edit_file", "delete", "execute"})
# Deep Agents' built-in helper that starts a sub-subagent. nexus's orchestrator
# already delegates; a hidden second layer would add cost that no step limit
# counts (it doesn't inherit the agent's middleware).
SUBAGENT_TOOL = "task"

# The project's own agent instructions, loaded as memory when present, so nexus
# itself carries nothing project-specific.
MEMORY_FILES = ["/AGENTS.md", "/CLAUDE.md"]

# Deep Agents writes its own files (oversized messages and tool results,
# summarized history, binary blobs) through the backend directly, which
# permissions don't cover. They go under the backend's artifacts root, so that
# root is a folder kept in the agent's memory: nothing reaches the project, or
# the engineer's batch, and the agent can still read them back.
SCRATCH = "/.nexus-scratch/"


def with_scratch(project_backend: BackendProtocol) -> CompositeBackend:
    return CompositeBackend(
        default=project_backend,
        routes={SCRATCH: StateBackend()},
        artifacts_root=SCRATCH.rstrip("/"),
    )


# Enforced at the tool layer, so they hold even if a model calls a hidden tool.
READ_ONLY_PERMISSIONS = [
    FilesystemPermission(operations=["write"], paths=["/**"], mode="deny"),
]
NO_SECRETS_PERMISSIONS = [
    FilesystemPermission(operations=["read"], paths=["/**/.env", "/**/.env.*"], mode="deny"),
]


class HideTools(AgentMiddleware):
    """Take some tools away from this agent: not offered, and refused if called.

    Deep Agents' own switch (``HarnessProfile.excluded_tools``) is registered
    per model for the whole process, which would also hide write tools from a
    later agent that needs them.
    """

    def __init__(self, names: frozenset[str]) -> None:
        super().__init__()
        self.names = names

    def wrap_model_call(self, request: ModelRequest, handler: Callable):
        tools = [t for t in request.tools if _tool_name(t) not in self.names]
        return handler(request.override(tools=tools))

    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable):
        name = request.tool_call["name"]
        if name in self.names:
            return ToolMessage(
                content=f"Error: the {name} tool is not available.",
                tool_call_id=request.tool_call["id"],
                name=name,
                status="error",
            )
        return handler(request)


LAST_STEP = (
    "You have reached your step limit: this is your last turn and no tools are "
    "available. Write your report now from what you have found. Say clearly what "
    "you could not finish or check."
)


class StepLimit(AgentMiddleware):
    """Cap an agent's model calls, and let its last call write the report.

    On the last allowed call the agent is told so and may not use tools
    (``tool_choice`` none), so it ends with a report of what it found instead
    of being cut off mid-search. LangChain's ``ModelCallLimitMiddleware`` stays
    underneath as the hard stop. ``reached`` says whether the limit was hit.
    Make a new one for every agent run.
    """

    def __init__(self, max_calls: int) -> None:
        super().__init__()
        if max_calls < 1:
            raise ValueError("max_calls must be at least 1")
        self.max_calls = max_calls
        self.calls = 0
        self.reached = False

    def wrap_model_call(self, request: ModelRequest, handler: Callable):
        self.calls += 1
        if self.calls >= self.max_calls:
            self.reached = True
            log.info("step limit of %d model calls reached; asking for the report", self.max_calls)
            request = request.override(
                messages=[*request.messages, HumanMessage(LAST_STEP)],
                tool_choice={"type": "none"},
            )
        return handler(request)

    def middleware(self) -> list[AgentMiddleware[Any, Any, Any]]:
        hard_stop = ModelCallLimitMiddleware(run_limit=self.max_calls, exit_behavior="end")
        return [hard_stop, self]


def _tool_name(tool) -> str | None:
    if isinstance(tool, dict):
        return tool.get("name")
    return getattr(tool, "name", None)


def existing_dir(path: str) -> str:
    """Pydantic validator for a ``project`` input: an existing directory."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"project directory not found: {path}")
    return str(resolved)


def read_only_agent(
    project: str, model: BaseChatModel, system_prompt: str, name: str, limit: StepLimit
):
    """A Deep Agent that can explore ``project`` but not change it, in at most
    ``limit.max_calls`` model calls.

    * the backend is rooted at the project, with path traversal blocked;
    * write tools are not offered, and every write is denied by permission
      anyway;
    * it can't start sub-subagents (Deep Agents' ``task`` helper is taken away);
    * there is no shell tool (Deep Agents only adds one for sandbox backends);
    * ``.env`` files are unreadable;
    * Deep Agents' own files stay in memory (``SCRATCH``).
    """
    return create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        backend=with_scratch(FilesystemBackend(root_dir=project, virtual_mode=True)),
        permissions=READ_ONLY_PERMISSIONS + NO_SECRETS_PERMISSIONS,
        middleware=[HideTools(WRITE_TOOLS | {SUBAGENT_TOOL}), *limit.middleware()],
        memory=MEMORY_FILES,
        name=name,
    )


ENGINEER_PERMISSIONS = [
    FilesystemPermission(operations=["write"], paths=["/.git/**"], mode="deny"),
    FilesystemPermission(operations=["write"], paths=["/**/.env", "/**/.env.*"], mode="deny"),
]


def staged_engineer(
    project: str, staging_dir: str, model: BaseChatModel, system_prompt: str, limit: StepLimit
):
    """A Deep Agent that can change ``project``, but only through staging, in
    at most ``limit.max_calls`` model calls.

    Its writes land in ``staging_dir`` (see ``nexus.staging``), never in the
    project; the caller collects them as one batch with ``backend.changes()``.
    It cannot write inside ``.git`` or to ``.env`` files, cannot read ``.env``
    files, has no shell, and can't start sub-subagents. Deep Agents' own files stay in memory
    (``SCRATCH``), so they never join the batch. Returns ``(agent, backend)``.
    """
    backend = StagedBackend(project, staging_dir)
    agent = create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        backend=with_scratch(backend),
        permissions=ENGINEER_PERMISSIONS + NO_SECRETS_PERMISSIONS,
        middleware=[HideTools(frozenset({SUBAGENT_TOOL})), *limit.middleware()],
        memory=MEMORY_FILES,
        name="engineer",
    )
    return agent, backend


def run_agent(agent, message: str, limit: StepLimit) -> tuple[str, float]:
    """Run an agent on one message; return its final text and what it cost.

    The usage callback counts every model call the agent makes. If the agent
    hit its step limit, the text says so: the report may be incomplete.
    """
    with get_usage_metadata_callback() as usage:
        result = agent.invoke({"messages": [{"role": "user", "content": message}]})
    text = result["messages"][-1].text
    if limit.reached:
        text += (
            f"\n\n(Stopped at the step limit of {limit.max_calls} model calls; "
            "this report may be incomplete.)"
        )
    return text, cost_usd(usage.usage_metadata)
