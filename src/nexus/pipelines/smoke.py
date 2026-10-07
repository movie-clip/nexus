"""Domain-free pipeline that exercises the engine.

    START -> step_a -> step_b -> check -+- ok -------> finish -> END
                        ^               +- breach ---> finish
                        |               +- give_up --> finish
                        +---- retry ----+

Inputs:
  scenario     what ``check`` reports:
                 ok      pass first time
                 breach  threshold broken (a human gate goes here later)
                 flaky   one retry, then pass
                 stuck   retry until max_retries, then give up with an error
  max_retries  cap for the retry route (default 2)
  fail_node    make this node raise, to drill failure handling:
  fail_kind      transient  retried with backoff (default)
                 bug        a ValueError, not retried
  fail_times   fail only the first N attempts; 0 (default) means every attempt
"""

from operator import add
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy

from nexus.retry import DEFAULT_RETRY, GIVE_UP, RETRY, retry_count, retry_update
from nexus.state import BaseState

Node = Literal["step_a", "step_b", "check", "finish"]


class InjectedTransient(Exception):
    """Stands in for a timeout or 5xx: the default retry policy retries it."""


class InjectedBug(ValueError):
    """Stands in for a programming error: the default retry policy does not retry it."""


class SmokeInput(TypedDict, total=False):
    scenario: Literal["ok", "breach", "flaky", "stuck"]
    max_retries: int
    fail_node: Node
    fail_kind: Literal["transient", "bug"]
    fail_times: int


class SmokeState(BaseState, SmokeInput, total=False):
    visited: Annotated[list[str], add]


def maybe_fail(state: SmokeState, node: Node, runtime: Runtime) -> None:
    if state.get("fail_node") != node:
        return
    attempt = runtime.execution_info.node_attempt if runtime.execution_info else 1
    times = state.get("fail_times", 0)
    if times and attempt > times:
        return
    if state.get("fail_kind", "transient") == "bug":
        raise InjectedBug(f"injected bug in {node}")
    raise InjectedTransient(f"injected transient fault in {node} (attempt {attempt})")


def step_a(state: SmokeState, runtime: Runtime) -> dict:
    maybe_fail(state, "step_a", runtime)
    return {"visited": ["step_a"]}


def step_b(state: SmokeState, runtime: Runtime) -> dict:
    maybe_fail(state, "step_b", runtime)
    return {"visited": ["step_b"]}


def check(state: SmokeState, runtime: Runtime) -> dict:
    maybe_fail(state, "check", runtime)
    scenario = state.get("scenario", "ok")
    wants_retry = scenario == "stuck" or (scenario == "flaky" and retry_count(state, "check") == 0)
    if wants_retry:
        update = retry_update(state, state.get("max_retries", 2), "check")
    else:
        update = {"verdict": "breach" if scenario == "breach" else "ok"}
    return {**update, "visited": ["check"]}


def finish(state: SmokeState, runtime: Runtime) -> dict:
    maybe_fail(state, "finish", runtime)
    return {"visited": ["finish"]}


def route_after_check(state: SmokeState) -> str:
    return state["verdict"]


def build(retry_policy: RetryPolicy = DEFAULT_RETRY) -> StateGraph:
    graph = StateGraph(SmokeState, input_schema=SmokeInput)
    graph.add_node("step_a", step_a, retry_policy=retry_policy)
    graph.add_node("step_b", step_b, retry_policy=retry_policy)
    graph.add_node("check", check, retry_policy=retry_policy)
    graph.add_node("finish", finish, retry_policy=retry_policy)
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


# Compiled without a checkpointer: `langgraph dev` / Agent Server supply their
# own, and nexus.run adds one for local runs.
graph = build().compile(name="smoke")
