"""Saved runs: one start per thread, and resume after a failure."""

from typing import TypedDict

import pytest
from langgraph.graph import END, START, StateGraph

from nexus.errors import InputError
from nexus.run import open_runs_db, resume_pipeline, run_pipeline


class S(TypedDict, total=False):
    n: int
    steps: list[str]


class In(TypedDict, total=False):
    n: int


def flaky_graph(calls: dict, fail_first_b: bool = True):
    """a -> b -> c; b raises on its first call only (a non-retried error)."""

    def a(state):
        calls["a"] = calls.get("a", 0) + 1
        return {"steps": [*state.get("steps", []), "a"]}

    def b(state):
        calls["b"] = calls.get("b", 0) + 1
        if fail_first_b and calls["b"] == 1:
            raise ValueError("b broke")
        return {"steps": [*state["steps"], "b"]}

    def c(state):
        calls["c"] = calls.get("c", 0) + 1
        return {"steps": [*state["steps"], "c"]}

    g = StateGraph(S, input_schema=In)
    for name, fn in [("a", a), ("b", b), ("c", c)]:
        g.add_node(name, fn)
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", END)
    return g.compile(name="flaky")


def test_failed_run_resumes_without_rerunning_finished_steps(isolated_runs_db):
    calls = {}
    graph = flaky_graph(calls)
    first = run_pipeline(graph, "t1", {"n": 1}, open_runs_db())
    assert (first.status, first.failed_node) == ("failed", "b")

    # A fresh connection, as a new process would open.
    second = resume_pipeline("t1", None, open_runs_db(), graph=graph)
    assert second.status == "completed"
    assert second.state["steps"] == ["a", "b", "c"]
    assert calls == {"a": 1, "b": 2, "c": 1}  # a is not run again


def test_a_thread_can_only_be_started_once(isolated_runs_db):
    graph = flaky_graph({}, fail_first_b=False)
    run_pipeline(graph, "once", {"n": 1}, open_runs_db())
    with pytest.raises(InputError, match="already exists; continue it with nexus.resume"):
        run_pipeline(graph, "once", {"n": 1}, open_runs_db())


def test_resume_refuses_unknown_and_completed_threads(isolated_runs_db):
    graph = flaky_graph({}, fail_first_b=False)
    with pytest.raises(InputError, match="no run with thread 'nope'"):
        resume_pipeline("nope", None, open_runs_db(), graph=graph)
    run_pipeline(graph, "done", {"n": 1}, open_runs_db())
    with pytest.raises(InputError, match="already completed"):
        resume_pipeline("done", None, open_runs_db(), graph=graph)


def test_decision_for_a_run_that_is_not_waiting_is_refused(isolated_runs_db):
    graph = flaky_graph({})
    run_pipeline(graph, "f", {"n": 1}, open_runs_db())  # failed, not waiting
    with pytest.raises(InputError, match="is not waiting for a decision"):
        resume_pipeline("f", {"decision": "approve"}, open_runs_db(), graph=graph)


def test_resume_finds_the_pipeline_recorded_with_the_run(isolated_runs_db, in_repo_root):
    inputs = {"fail_node": "check", "fail_kind": "bug"}
    from nexus.pipelines import smoke

    first = run_pipeline(smoke.graph, "smoke-1", inputs, open_runs_db())
    assert first.failed_node == "check"
    # No graph given: resume loads "smoke" from langgraph.json by the saved name.
    again = resume_pipeline("smoke-1", None, open_runs_db())
    assert again.failed_node == "check"  # the injected bug fires again...
    assert again.state["visited"] == ["step_a", "step_b"]  # ...but nothing before it re-ran
