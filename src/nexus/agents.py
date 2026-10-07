"""Building blocks shared by nexus's Deep Agents."""

from collections.abc import Callable
from pathlib import Path

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.language_models import BaseChatModel

from nexus.models import cost_usd

WRITE_TOOLS = frozenset({"write_file", "edit_file", "delete", "execute"})

# The project's own agent instructions, loaded as memory when present, so nexus
# itself carries nothing project-specific.
MEMORY_FILES = ["/AGENTS.md", "/CLAUDE.md"]

# Enforced at the tool layer, so they hold even if a model calls a hidden tool.
READ_ONLY_PERMISSIONS = [
    FilesystemPermission(operations=["write"], paths=["/**"], mode="deny"),
]
NO_SECRETS_PERMISSIONS = [
    FilesystemPermission(operations=["read"], paths=["/**/.env", "/**/.env.*"], mode="deny"),
]


class HideTools(AgentMiddleware):
    """Stop offering some tools to the model, for this agent only.

    Deep Agents' own switch (``HarnessProfile.excluded_tools``) is registered
    per model for the whole process, which would also hide write tools from a
    later agent that needs them. Permissions still deny the calls; hiding the
    tools just keeps the model from planning around tools it can't use.
    """

    def __init__(self, names: frozenset[str]) -> None:
        super().__init__()
        self.names = names

    def wrap_model_call(self, request: ModelRequest, handler: Callable):
        tools = [t for t in request.tools if _tool_name(t) not in self.names]
        return handler(request.override(tools=tools))


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


def read_only_agent(project: str, model: BaseChatModel, system_prompt: str, name: str):
    """A Deep Agent that can explore ``project`` but not change it.

    * the backend is rooted at the project, with path traversal blocked;
    * write tools are not offered, and every write is denied by permission
      anyway; subagents (the built-in ``task`` helper) inherit the rules;
    * there is no shell tool (Deep Agents only adds one for sandbox backends);
    * ``.env`` files are unreadable.
    """
    return create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        backend=FilesystemBackend(root_dir=project, virtual_mode=True),
        permissions=READ_ONLY_PERMISSIONS + NO_SECRETS_PERMISSIONS,
        middleware=[HideTools(WRITE_TOOLS)],
        memory=MEMORY_FILES,
        name=name,
    )


def run_agent(agent, message: str) -> tuple[str, float]:
    """Run an agent on one message; return its final text and what it cost.

    The usage callback counts every model call, including any subagent the
    agent starts.
    """
    with get_usage_metadata_callback() as usage:
        result = agent.invoke({"messages": [{"role": "user", "content": message}]})
    return result["messages"][-1].text, cost_usd(usage.usage_metadata)
