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


def test_tester_without_a_command_fails_the_run(project, tmp_path):
    orch = ScriptedOrchestrator(decide("tester", command=" "))
    result = start(project, make_graph(tmp_path, orch))
    assert result.status == "failed"
    assert "orchestrator chose tester without a command" in result.error


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
