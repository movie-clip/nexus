"""The engineer's staged batch, the approval pause, and apply: scripted models."""

import pytest

from nexus.errors import InputError
from nexus.pipelines import team
from nexus.run import open_runs_db, resume_pipeline, run_pipeline
from tests.fakes import ScriptedOrchestrator, decide, final, scripted, tool_call

ORIGINAL = "class Jump {\r\n    float Speed = 10f;\r\n}\r\n"


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "Jump.cs").write_bytes(ORIGINAL.encode())
    return root


def engineer_that_edits():
    return scripted(
        tool_call("edit_file", file_path="/Jump.cs", old_string="10f", new_string="12f"),
        final("Raised Speed to 12 in Jump.cs."),
    )


def make_graph(orchestrator, engineers):
    queue = list(engineers)

    def model_for(role):
        return orchestrator if role == "orchestrator" else queue.pop(0)

    return team.build(model_for=model_for).compile(name="team")


def start(project, graph, thread="t"):
    inputs = {"request": "Make jumps higher", "project": str(project)}
    return run_pipeline(graph, thread, inputs, open_runs_db())


def test_engineer_batch_pauses_for_approval_without_touching_the_project(project):
    orch = ScriptedOrchestrator(decide("engineer", brief="Set Speed to 12f in /Jump.cs."))
    result = start(project, make_graph(orch, [engineer_that_edits()]))

    assert result.status == "waiting"
    assert result.exit_code == 4
    asked = result.waiting_for
    assert asked["files"] == ["/Jump.cs"]
    assert "-    float Speed = 10f;" in asked["diff"]
    assert "+    float Speed = 12f;" in asked["diff"]
    assert "Raised Speed to 12" in asked["summary"]
    assert (project / "Jump.cs").read_bytes() == ORIGINAL.encode()  # nothing written yet


def test_approve_applies_the_batch_and_keeps_crlf(project):
    orch = ScriptedOrchestrator(
        decide("engineer", brief="Set Speed to 12f."),
        decide("finish", answer="Speed is now 12 (`Jump.cs:2`)."),
    )
    graph = make_graph(orch, [engineer_that_edits()])
    start(project, graph)

    # A new connection, as `python -m nexus.resume` in a new terminal would open.
    result = resume_pipeline("t", {"decision": "approve", "note": ""}, open_runs_db(), graph=graph)

    assert result.status == "completed"
    assert (project / "Jump.cs").read_bytes() == ORIGINAL.replace("10f", "12f").encode()
    user = result.state["reports"][-1]
    assert user["role"] == "user"
    assert user["report"] == "The user approved the batch. Applied: /Jump.cs."
    assert result.state["pending"] == []
    # The orchestrator heard the outcome before finishing.
    assert "The user approved the batch" in orch.calls[-1][1][1][1]


def test_reject_writes_nothing_and_passes_the_note_on(project):
    orch = ScriptedOrchestrator(
        decide("engineer", brief="Set Speed to 12f."),
        decide("finish", answer="Stopped: you asked to keep the speed."),
    )
    graph = make_graph(orch, [engineer_that_edits()])
    start(project, graph)
    result = resume_pipeline(
        "t", {"decision": "reject", "note": "keep it at 10"}, open_runs_db(), graph=graph
    )

    assert (project / "Jump.cs").read_bytes() == ORIGINAL.encode()
    assert result.state["reports"][-1]["report"] == (
        "The user rejected the batch; nothing was written to the project. Their note: keep it at 10"
    )


def test_file_changed_while_waiting_is_not_overwritten(project):
    orch = ScriptedOrchestrator(decide("engineer", brief="b"), decide("finish", answer="a"))
    graph = make_graph(orch, [engineer_that_edits()])
    start(project, graph)
    (project / "Jump.cs").write_bytes(b"edited by hand\r\n")

    result = resume_pipeline("t", {"decision": "approve"}, open_runs_db(), graph=graph)

    assert (project / "Jump.cs").read_bytes() == b"edited by hand\r\n"
    assert "changed on disk after staging: /Jump.cs" in result.state["reports"][-1]["report"]


def test_engineer_without_changes_goes_straight_back(project):
    orch = ScriptedOrchestrator(decide("engineer", brief="b"), decide("finish", answer="a"))
    result = start(project, make_graph(orch, [scripted(final("Nothing needed changing."))]))
    assert result.status == "completed"
    assert [r["role"] for r in result.state["reports"]] == ["engineer"]


def test_approval_outcomes_do_not_use_up_rounds(project):
    orch = ScriptedOrchestrator(
        decide("engineer", brief="b"),
        decide("finish", answer="a"),
    )
    graph = make_graph(orch, [engineer_that_edits()])
    run_pipeline(
        graph,
        "t",
        {"request": "r", "project": str(project), "max_rounds": 2},
        open_runs_db(),
    )
    resume_pipeline("t", {"decision": "approve"}, open_runs_db(), graph=graph)
    # 1 engineer round used; the "user" report doesn't count, so 1 round is left.
    assert "<rounds_left>1</rounds_left>" in orch.calls[-1][1][1][1]
    assert orch.calls[-1][0] is team.Decision


def test_engineer_cannot_write_into_git(project):
    (project / ".git").mkdir()
    eng = scripted(
        tool_call("write_file", file_path="/.git/config", content="x"),
        final("tried"),
    )
    orch = ScriptedOrchestrator(decide("engineer", brief="b"), decide("finish", answer="a"))
    result = start(project, make_graph(orch, [eng]))
    assert result.status == "completed"  # nothing staged, so no pause
    assert not (project / ".git" / "config").exists()


def test_diff_marks_new_and_deleted_files():
    diff = team.unified_diff(
        [
            {"path": "/new.cs", "before": None, "after": "a\n"},
            {"path": "/old.cs", "before": "b\n", "after": None},
        ]
    )
    assert "--- /dev/null\n+++ b/new.cs" in diff
    assert "--- a/old.cs\n+++ /dev/null" in diff


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (("write_file", "/.git/config"), "Error: permission denied for write on /.git/config"),
        (("write_file", "/sub/.env"), "Error: permission denied for write on /sub/.env"),
        (("read_file", "/.env"), "Error: permission denied for read on /.env"),
        (("write_file", "/../escape.txt"), "Error: Path traversal not allowed: /../escape.txt"),
    ],
)
def test_engineer_tool_limits(project, tmp_path, call, error):
    from langchain_core.messages import ToolMessage

    from nexus.agents import StepLimit, staged_engineer

    (project / ".git").mkdir()
    (project / ".env").write_text("K=1")
    name, path = call
    args = {"file_path": path} | ({"content": "x"} if name == "write_file" else {})
    agent, backend = staged_engineer(
        str(project),
        str(tmp_path / "stage"),
        scripted(tool_call(name, **args), final("ok")),
        "x",
        StepLimit(5),
    )
    out = agent.invoke({"messages": [{"role": "user", "content": "go"}]})
    assert [m.text for m in out["messages"] if isinstance(m, ToolMessage)] == [error]
    assert backend.changes() == []


def test_diff_warns_about_line_ending_changes_it_cannot_show():
    diff = team.unified_diff([{"path": "/m.cs", "before": "a\r\nb\r\n", "after": "a\nb\r\nc\n"}])
    assert diff.startswith("WARNING /m.cs: line endings change on 1 unchanged lines")
    clean = team.unified_diff([{"path": "/m.cs", "before": "a\r\nb\r\n", "after": "a\r\nB\r\n"}])
    assert "WARNING" not in clean


@pytest.mark.parametrize("bad", [{"decision": "maybe"}, "approve", {"note": "x"}])
def test_a_malformed_batch_answer_asks_again_and_writes_nothing(project, bad):
    orch = ScriptedOrchestrator(
        decide("engineer", brief="Set Speed to 12f."), decide("finish", answer="Stopped.")
    )
    graph = make_graph(orch, [engineer_that_edits()])
    start(project, graph)
    again = resume_pipeline("t", bad, open_runs_db(), graph=graph)
    assert again.status == "waiting"
    assert "answer approve or reject; got" in again.waiting_for["question"]
    assert "+    float Speed = 12f;" in again.waiting_for["diff"]
    assert (project / "Jump.cs").read_bytes() == ORIGINAL.encode()
    # The repeated question takes a real answer.
    done = resume_pipeline("t", {"decision": "reject"}, open_runs_db(), graph=graph)
    assert done.status == "completed"
    assert (project / "Jump.cs").read_bytes() == ORIGINAL.encode()


def test_a_waiting_run_needs_a_decision_to_continue(project):
    orch = ScriptedOrchestrator(decide("engineer", brief="Set Speed to 12f."))
    graph = make_graph(orch, [engineer_that_edits()])
    start(project, graph)
    with pytest.raises(InputError, match="is waiting for a decision"):
        resume_pipeline("t", None, open_runs_db(), graph=graph)


def test_a_waiting_run_refuses_a_message(project):
    orch = ScriptedOrchestrator(decide("engineer", brief="Set Speed to 12f."))
    graph = make_graph(orch, [engineer_that_edits()])
    start(project, graph)
    with pytest.raises(InputError, match="is waiting for a decision; answer it first"):
        resume_pipeline("t", None, open_runs_db(), graph=graph, message="Use 11f.")
