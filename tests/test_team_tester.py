"""The tester in the team pipeline: picking, approving and running commands."""

from pathlib import Path

import pytest

from nexus.pipelines import team
from nexus.run import open_runs_db, resume_pipeline, run_pipeline
from tests.fakes import ScriptedOrchestrator, decide, final, scripted, tool_call
from tests.test_commands import FAKE_TOOL, write_config

CONFIG = """\
    [commands.build]
    description = "Fast compile check."
    run = [PYTHON, "tool.py", "{mode}"]
    approval = "auto"

    [commands.build.params]
    mode = { pattern = "ok|fail", default = "ok" }

    [commands.tests]
    description = "Slow test run."
    run = [PYTHON, "tool.py", "touch"]
    approval = "ask"
"""


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "tool.py").write_text(FAKE_TOOL)
    (root / "Jump.cs").write_bytes(b"float Speed = 10f;\r\n")
    write_config(tmp_path / "projects", root, CONFIG)
    return root


def make_graph(tmp_path, orchestrator, engineers=()):
    queue = list(engineers)

    def model_for(role):
        return orchestrator if role == "orchestrator" else queue.pop(0)

    builder = team.build(model_for=model_for, projects_dir=tmp_path / "projects")
    return builder.compile(name="team")


def start(project, graph, **extra):
    inputs = {"request": "Make jumps higher", "project": str(project), **extra}
    return run_pipeline(graph, "t", inputs, open_runs_db())


def test_orchestrator_sees_the_configured_commands(project, tmp_path):
    orch = ScriptedOrchestrator(decide("finish", answer="a"))
    start(project, make_graph(tmp_path, orch))
    message = orch.calls[0][1][1][1]
    assert "<commands>\n- build: Fast compile check." in message
    assert "mode (must match 'ok|fail', default 'ok')" in message


def test_auto_command_runs_without_asking_and_reports_back(project, tmp_path):
    orch = ScriptedOrchestrator(
        decide("tester", command="build", args={"mode": "fail"}, reason="check the build"),
        decide("finish", answer="The build fails."),
    )
    result = start(project, make_graph(tmp_path, orch))

    assert result.status == "completed"
    (report,) = result.state["reports"]
    assert report["role"] == "tester"
    assert report["brief"] == "Run build with {'mode': 'fail'}: check the build"
    assert "Result: FAILED, exit code 1" in report["report"]
    assert "error CS0103" in report["report"]
    assert result.state["prepared"] == {}
    # The log sits next to the runs database, under the thread id.
    log = Path(report["report"].rsplit("Full log: ", 1)[1])
    assert log.parent.name == "t" and log.parent.parent.name == "logs" and log.is_file()


def test_ask_command_pauses_then_runs_on_approve(project, tmp_path):
    orch = ScriptedOrchestrator(
        decide("tester", command="tests", reason="verify the change"),
        decide("finish", answer="Done."),
    )
    graph = make_graph(tmp_path, orch)
    paused = start(project, graph)

    assert paused.status == "waiting"
    asked = paused.waiting_for
    assert asked["question"] == "Run this command?"
    assert "verify the change" in asked["summary"]
    assert "tool.py touch" in asked["command"]
    assert f"in {project.resolve()}" in asked["command"]
    assert not (project / "tracked.txt").exists()  # nothing ran yet

    result = resume_pipeline("t", {"decision": "approve"}, open_runs_db(), graph=graph)
    assert result.status == "completed"
    assert (project / "tracked.txt").exists()
    assert "Result: succeeded, exit code 0" in result.state["reports"][-1]["report"]


def test_rejected_command_does_not_run(project, tmp_path):
    orch = ScriptedOrchestrator(
        decide("tester", command="tests"),
        decide("finish", answer="Stopped."),
    )
    graph = make_graph(tmp_path, orch)
    start(project, graph)
    result = resume_pipeline(
        "t", {"decision": "reject", "note": "the editor is open"}, open_runs_db(), graph=graph
    )

    assert not (project / "tracked.txt").exists()
    assert result.state["reports"][-1] == {
        "role": "user",
        "brief": "Run this command?",
        "report": "The user rejected running tests; it did not run. Their note: the editor is open",
    }
    assert "The user rejected running tests" in orch.calls[-1][1][1][1]


@pytest.mark.parametrize(
    ("command", "args", "why"),
    [
        ("build", {"mode": "ok && del *"}, "doesn't match"),
        ("deploy", {}, "unknown command 'deploy'; available: build, tests"),
    ],
)
def test_bad_command_is_refused_and_reported(project, tmp_path, command, args, why):
    orch = ScriptedOrchestrator(
        decide("tester", command=command, args=args), decide("finish", answer="a")
    )
    result = start(project, make_graph(tmp_path, orch))
    assert result.status == "completed"  # no pause: nothing to approve
    (report,) = result.state["reports"]
    assert report["role"] == "tester"
    assert report["report"].startswith("Not run: ") and why in report["report"]


def test_project_without_config_has_no_commands(tmp_path):
    bare = tmp_path / "bare"
    bare.mkdir()
    orch = ScriptedOrchestrator(decide("tester", command="build"), decide("finish", answer="a"))
    result = start(bare, make_graph(tmp_path, orch))
    assert "<commands>\nNo commands are configured" in orch.calls[0][1][1][1]
    assert result.state["reports"][0]["report"] == (
        "Not run: no commands are configured for this project."
    )


def test_change_then_build_flow(project, tmp_path):
    """engineer -> approve -> apply -> tester (auto) -> finish."""
    engineer = scripted(
        tool_call("edit_file", file_path="/Jump.cs", old_string="10f", new_string="12f"),
        final("Raised Speed."),
    )
    orch = ScriptedOrchestrator(
        decide("engineer", brief="Set Speed to 12f in /Jump.cs."),
        decide("tester", command="build"),
        decide("finish", answer="Speed is 12 and it builds."),
    )
    graph = make_graph(tmp_path, orch, [engineer])
    assert start(project, graph).status == "waiting"
    result = resume_pipeline("t", {"decision": "approve"}, open_runs_db(), graph=graph)

    assert result.status == "completed"
    assert [r["role"] for r in result.state["reports"]] == ["engineer", "user", "tester"]
    assert (project / "Jump.cs").read_bytes() == b"float Speed = 12f;\r\n"
    assert "Result: succeeded" in result.state["reports"][-1]["report"]


def test_command_rejections_do_not_use_up_rounds(project, tmp_path):
    orch = ScriptedOrchestrator(decide("tester", command="tests"), decide("finish", answer="a"))
    graph = make_graph(tmp_path, orch)
    start(project, graph, max_rounds=1)
    resume_pipeline("t", {"decision": "reject"}, open_runs_db(), graph=graph)
    assert "<rounds_left>1</rounds_left>" in orch.calls[-1][1][1][1]


def test_cli_prints_the_command_to_approve(project, tmp_path, capsys):
    from nexus.__main__ import show

    orch = ScriptedOrchestrator(decide("tester", command="tests", reason="verify"))
    assert show(start(project, make_graph(tmp_path, orch))) == 4
    out = capsys.readouterr()
    assert "Run this command?" in out.out and "tool.py touch" in out.out
    assert "--decision approve" in out.err


def test_a_command_that_cannot_start_is_reported_not_raised(project, tmp_path, monkeypatch):
    def cannot_start(prepared):
        raise FileNotFoundError("tool vanished")

    monkeypatch.setattr(team.commands, "run_prepared", cannot_start)
    orch = ScriptedOrchestrator(decide("tester", command="build"), decide("finish", answer="a"))
    result = start(project, make_graph(tmp_path, orch))
    assert result.status == "completed"
    assert "Command build could not start: tool vanished" in result.state["reports"][-1]["report"]


# --- verify: commands run after every approved batch ---------------------------------


def with_verify(tmp_path, project, verify: str) -> None:
    write_config(tmp_path / "projects", project, f"verify = {verify}\n\n" + CONFIG)


def edit_jump() -> object:
    return scripted(
        tool_call("edit_file", file_path="/Jump.cs", old_string="10f", new_string="12f"),
        final("Raised Speed."),
    )


def change_and_approve(project, tmp_path, orch, decision="approve", **extra):
    graph = make_graph(tmp_path, orch, [edit_jump()])
    assert start(project, graph, **extra).status == "waiting"
    return resume_pipeline("t", {"decision": decision}, open_runs_db(), graph=graph)


def test_an_approved_batch_is_verified_without_asking(project, tmp_path):
    with_verify(tmp_path, project, '["build", "tests"]')  # "tests" is set to ask
    orch = ScriptedOrchestrator(
        decide("engineer", brief="Set Speed to 12f in /Jump.cs."),
        decide("finish", answer="Done."),
    )
    result = change_and_approve(project, tmp_path, orch)

    assert result.status == "completed"
    roles = [r["role"] for r in result.state["reports"]]
    assert roles == ["engineer", "user", "verify", "verify"]
    build, tests = result.state["reports"][2:]
    assert build["brief"] == "Run automatically after the approved change: build"
    assert "Command: build" in build["report"] and "Result: succeeded" in build["report"]
    assert "Command: tests" in tests["report"]
    assert (project / "tracked.txt").exists()  # ran without a pause
    assert "Run automatically after the approved change: tests" in orch.calls[-1][1][1][1]


def test_verify_stops_at_the_first_failure(project, tmp_path):
    write_config(
        tmp_path / "projects",
        project,
        'verify = ["broken", "tests"]\n\n'
        + CONFIG
        + '\n[commands.broken]\ndescription = "d"\nrun = [PYTHON, "tool.py", "fail"]\n',
    )
    orch = ScriptedOrchestrator(decide("engineer", brief="b"), decide("finish", answer="a"))
    result = change_and_approve(project, tmp_path, orch)

    (failed,) = [r for r in result.state["reports"] if r["role"] == "verify"]
    assert "Command: broken" in failed["report"] and "Result: FAILED" in failed["report"]
    assert not (project / "tracked.txt").exists()


def test_a_rejected_batch_is_not_verified(project, tmp_path):
    with_verify(tmp_path, project, '["tests"]')
    orch = ScriptedOrchestrator(decide("engineer", brief="b"), decide("finish", answer="a"))
    result = change_and_approve(project, tmp_path, orch, decision="reject")

    assert [r["role"] for r in result.state["reports"]] == ["engineer", "user"]
    assert not (project / "tracked.txt").exists()


def test_verify_does_not_use_up_rounds(project, tmp_path):
    with_verify(tmp_path, project, '["build"]')
    orch = ScriptedOrchestrator(decide("engineer", brief="b"), decide("finish", answer="a"))
    change_and_approve(project, tmp_path, orch, max_rounds=2)
    assert "<rounds_left>1</rounds_left>" in orch.calls[-1][1][1][1]  # only the engineer counted


def test_the_orchestrator_is_told_what_runs_automatically(project, tmp_path):
    with_verify(tmp_path, project, '["build"]')
    orch = ScriptedOrchestrator(decide("finish", answer="a"))
    start(project, make_graph(tmp_path, orch))
    assert "Run automatically after each approved change: build" in orch.calls[0][1][1][1]


def test_a_verify_command_that_cannot_start_is_reported(project, tmp_path, monkeypatch):
    def cannot_start(prepared):
        raise FileNotFoundError("tool vanished")

    monkeypatch.setattr(team.commands, "run_prepared", cannot_start)
    with_verify(tmp_path, project, '["build", "tests"]')
    orch = ScriptedOrchestrator(decide("engineer", brief="b"), decide("finish", answer="a"))
    result = change_and_approve(project, tmp_path, orch)

    (report,) = [r for r in result.state["reports"] if r["role"] == "verify"]
    assert report["report"] == "Command build could not start: tool vanished"


def test_a_verify_command_that_cannot_be_prepared_is_reported(project, tmp_path, monkeypatch):
    def refuse(*args):
        raise team.commands.CommandError("program 'python' not found")

    with_verify(tmp_path, project, '["build", "tests"]')
    monkeypatch.setattr(team.commands, "prepare", refuse)
    orch = ScriptedOrchestrator(decide("engineer", brief="b"), decide("finish", answer="a"))
    result = change_and_approve(project, tmp_path, orch)

    (report,) = [r for r in result.state["reports"] if r["role"] == "verify"]
    assert report["report"] == "Not run: program 'python' not found."


@pytest.mark.parametrize("bad", [{"decision": "maybe"}, "approve"])
def test_a_malformed_command_answer_asks_again_and_runs_nothing(project, tmp_path, bad):
    orch = ScriptedOrchestrator(decide("tester", command="tests"), decide("finish", answer="a"))
    graph = make_graph(tmp_path, orch)
    start(project, graph)
    again = resume_pipeline("t", bad, open_runs_db(), graph=graph)
    assert again.status == "waiting"
    assert again.waiting_for["question"].startswith("Run this command? (answer approve or reject")
    assert not (project / "tracked.txt").exists()
    done = resume_pipeline("t", {"decision": "approve"}, open_runs_db(), graph=graph)
    assert done.status == "completed"
    assert (project / "tracked.txt").exists()  # ran once, after the real approval
