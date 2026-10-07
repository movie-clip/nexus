"""Domain-free pipeline that exercises the engine's routing.

    START -> step_a -> step_b -> check -+- ok -------> finish -> END
                        ^               +- breach ---> finish
                        |               +- give_up --> finish
                        +---- retry ----+

The ``scenario`` input decides what ``check`` reports:
  ok      pass first time
  breach  threshold broken (a human gate goes here in slice 4)
  flaky   one retry, then pass
  stuck   retry until the limit, then give up with an error
"""

from operator import add
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from nexus.registry import register
from nexus.state import GIVE_UP, RETRY, BaseState, retry_count, retry_update

Scenario = Literal["ok", "breach", "flaky", "stuck"]


class SmokeInput(TypedDict, total=False):
    scenario: Scenario
    max_retries: int


class SmokeState(BaseState, SmokeInput, total=False):
    visited: Annotated[list[str], add]


def step_a(state: SmokeState) -> dict:
    return {"visited": ["step_a"]}


def step_b(state: SmokeState) -> dict:
    return {"visited": ["step_b"]}


def check(state: SmokeState) -> dict:
    scenario = state.get("scenario", "ok")
    wants_retry = scenario == "stuck" or (scenario == "flaky" and retry_count(state, "check") == 0)
    if wants_retry:
        update = retry_update(state, state.get("max_retries", 2), "check")
    else:
        update = {"verdict": "breach" if scenario == "breach" else "ok"}
    return {**update, "visited": ["check"]}


def finish(state: SmokeState) -> dict:
    return {"visited": ["finish"]}


def route_after_check(state: SmokeState) -> str:
    return state["verdict"]


@register("smoke")
def build() -> StateGraph:
    graph = StateGraph(SmokeState, input_schema=SmokeInput)
    graph.add_node("step_a", step_a)
    graph.add_node("step_b", step_b)
    graph.add_node("check", check)
    graph.add_node("finish", finish)
    graph.add_edge(START, "step_a")
    graph.add_edge("step_a", "step_b")
    graph.add_edge("step_b", "check")
    graph.add_conditional_edges(
        "check",
        route_after_check,
        {"ok": "finish", "breach": "finish", GIVE_UP: "finish", RETRY: "step_b"},
    )
    graph.add_edge("finish", END)
    return graph
