"""Answer an engineering request about a project, read-only.

    START -> answer -> END

One Deep Agent explores the project with read-only file tools (ls, read_file,
glob, grep, plus `task` for a subagent with the same limits) and answers in
text. It cannot change anything:

  * the backend is rooted at the project, with path traversal blocked;
  * write tools are not offered to the model, and every write is denied by a
    permission rule anyway; subagents inherit the rules;
  * there is no shell tool (Deep Agents only adds one for sandbox backends);
  * secrets files (.env) are unreadable.

The project's own AGENTS.md / CLAUDE.md are loaded as memory when present, so
nexus itself carries nothing project-specific.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, TypedDict

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from pydantic import AfterValidator

from nexus.agents import NO_SECRETS_PERMISSIONS, READ_ONLY_PERMISSIONS, WRITE_TOOLS, HideTools
from nexus.models import cost_usd, make_model
from nexus.state import BaseState

SYSTEM_PROMPT = """\
You are a senior software engineer answering a request about the project in \
your filesystem. The filesystem root `/` is the project root.

Ground every claim in files you have read, and cite them as `path:line` where \
it helps. If the files don't settle a point, say what you checked and what \
remains uncertain instead of guessing.

You have read-only access: you can explore and explain, not change files. If \
the request asks for a change, describe the change you would make, where, and \
why, as a concrete plan.

Answer in Markdown, starting with the direct answer."""

MEMORY_FILES = ["/AGENTS.md", "/CLAUDE.md"]


def _existing_dir(path: str) -> str:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"project directory not found: {path}")
    return str(resolved)


class AskInput(TypedDict):
    request: str
    project: Annotated[str, AfterValidator(_existing_dir)]


class AskState(BaseState, total=False):
    request: str
    project: str
    answer: str


def build_agent(project: str, model: BaseChatModel):
    return create_deep_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        backend=FilesystemBackend(root_dir=project, virtual_mode=True),
        permissions=READ_ONLY_PERMISSIONS + NO_SECRETS_PERMISSIONS,
        middleware=[HideTools(WRITE_TOOLS)],
        memory=MEMORY_FILES,
        name="ask",
    )


def build(model_factory: Callable[[], BaseChatModel] = make_model) -> StateGraph:
    def answer(state: AskState) -> dict:
        agent = build_agent(state["project"], model_factory())
        # Counts every model call, including any subagent the agent starts.
        with get_usage_metadata_callback() as usage:
            result = agent.invoke({"messages": [{"role": "user", "content": state["request"]}]})
        return {
            "answer": result["messages"][-1].text,
            "cost_usd": cost_usd(usage.usage_metadata),
        }

    graph = StateGraph(AskState, input_schema=AskInput)
    # No graph-level retry policy here: one agent run is many model calls, and
    # re-running it would repeat their cost. The Anthropic SDK already retries
    # each call on 429, 5xx and connection errors.
    graph.add_node("answer", answer)
    graph.add_edge(START, "answer")
    graph.add_edge("answer", END)
    return graph


# The model is created when a run starts, not at import, so loading this module
# (tests, langgraph dev) needs no API key.
graph = build().compile(name="ask")
