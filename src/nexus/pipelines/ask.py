"""Answer an engineering request about a project with one read-only agent.

    START -> answer -> END

One Deep Agent explores the project with read-only file tools (ls, read_file,
glob, grep, plus `task` for a helper subagent with the same limits) and
answers in text. See ``nexus.agents.read_only_agent`` for what it cannot do.
The ``team`` pipeline answers the same requests with an orchestrator and
specialist roles.
"""

from collections.abc import Callable
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from pydantic import AfterValidator

from nexus.agents import StepLimit, existing_dir, read_only_agent, run_agent
from nexus.models import make_model
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


class AskInput(TypedDict):
    request: str
    project: Annotated[str, AfterValidator(existing_dir)]


class AskState(BaseState, total=False):
    request: str
    project: str
    answer: str


# Model calls one answer may take; the last one writes the answer.
MAX_MODEL_CALLS = 30


def build_agent(project: str, model: BaseChatModel, limit: StepLimit):
    return read_only_agent(project, model, SYSTEM_PROMPT, name="ask", limit=limit)


def build(model_factory: Callable[[], BaseChatModel] = make_model) -> StateGraph:
    def answer(state: AskState) -> dict:
        limit = StepLimit(MAX_MODEL_CALLS)
        agent = build_agent(state["project"], model_factory(), limit)
        text, cost = run_agent(agent, state["request"], limit)
        return {"answer": text, "cost_usd": cost}

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
