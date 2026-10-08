"""Saved runs: one start per thread, and resume after a failure."""

from typing import TypedDict

import pytest
from langchain_core.runnables import RunnableConfig
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
    calls: dict[str, int] = {}
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


def two_failures_graph(calls: dict):
    """a -> b -> c; b and c each raise on their first call."""

    def step(name):
        def node(state):
            calls[name] = calls.get(name, 0) + 1
            if name != "a" and calls[name] == 1:
                raise ValueError(f"{name} broke")
            return {"steps": [*state.get("steps", []), name]}

        return node

    g = StateGraph(S, input_schema=In)
    for name in "abc":
        g.add_node(name, step(name))
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", END)
    return g.compile(name="flaky")


def test_every_checkpoint_keeps_the_pipeline_name_across_resumes(isolated_runs_db):
    graph = two_failures_graph({})
    run_pipeline(graph, "t", {"n": 1}, open_runs_db())  # fails at b
    second = resume_pipeline("t", None, open_runs_db(), graph=graph)  # b passes, c fails
    assert second.failed_node == "c"

    config: RunnableConfig = {"configurable": {"thread_id": "t"}}
    names = [c.metadata.get("pipeline") for c in open_runs_db().list(config)]
    assert len(names) > 3 and set(names) == {"flaky"}


def test_resume_finds_the_pipeline_recorded_with_the_run(isolated_runs_db, in_repo_root):
    inputs = {"fail_node": "check", "fail_kind": "bug"}
    from nexus.pipelines import smoke

    first = run_pipeline(smoke.graph, "smoke-1", inputs, open_runs_db())
    assert first.failed_node == "check"
    # No graph given: resume loads "smoke" from langgraph.json by the saved name.
    again = resume_pipeline("smoke-1", None, open_runs_db())
    assert again.failed_node == "check"  # the injected bug fires again...
    assert again.state["visited"] == ["step_a", "step_b"]  # ...but nothing before it re-ran


class ChatState(TypedDict, total=False):
    heard: list[str]
    follow_up: str


class ChatIn(TypedDict, total=False):
    follow_up: str


def chat_graph():
    """One node that records each follow-up it is sent."""

    def listen(state):
        return {"heard": [*state.get("heard", []), state.get("follow_up", "")]}

    g = StateGraph(ChatState, input_schema=ChatIn)
    g.add_node("listen", listen)
    g.add_edge(START, "listen")
    g.add_edge("listen", END)
    return g.compile(name="chat")


def test_a_message_continues_a_completed_run_on_its_thread(isolated_runs_db):
    graph = chat_graph()
    run_pipeline(graph, "c", {}, open_runs_db())
    result = resume_pipeline("c", None, open_runs_db(), graph=graph, message="hello")
    assert result.status == "completed"
    assert result.state["heard"] == ["", "hello"]


def test_a_completed_run_that_takes_messages_says_so(isolated_runs_db):
    graph = chat_graph()
    run_pipeline(graph, "c", {}, open_runs_db())
    with pytest.raises(InputError, match="already completed; send a follow-up message"):
        resume_pipeline("c", None, open_runs_db(), graph=graph)


@pytest.mark.parametrize(
    ("fails", "kwargs", "message"),
    [
        (False, {"message": "hi"}, "pipeline 'flaky' doesn't take follow-up messages"),
        (True, {"message": "hi"}, "stopped at a failed step; retry it"),
        (False, {"message": "  "}, "the message is empty"),
        (False, {"message": "hi", "decision": {"decision": "approve"}}, "not both"),
    ],
)
def test_bad_messages_are_refused(isolated_runs_db, fails, kwargs, message):
    graph = flaky_graph({}, fail_first_b=fails)
    run_pipeline(graph, "m", {"n": 1}, open_runs_db())
    with pytest.raises(InputError, match=message):
        resume_pipeline("m", checkpointer=open_runs_db(), graph=graph, **kwargs)


def test_a_completed_result_says_whether_its_pipeline_takes_messages(isolated_runs_db):
    assert run_pipeline(chat_graph(), "c", {}, open_runs_db()).follow_ups
    flaky = flaky_graph({}, fail_first_b=False)
    assert not run_pipeline(flaky, "f", {"n": 1}, open_runs_db()).follow_ups
