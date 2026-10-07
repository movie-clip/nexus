"""The commands a person types: `python -m nexus` and `python -m nexus.resume`,
run in-process with scripted models (the team graph is swapped for one)."""

import json

import pytest

from nexus import __main__ as nexus_cli
from nexus import resume as resume_cli
from nexus.pipelines import team
from tests.fakes import ScriptedOrchestrator, decide
from tests.test_team_engineer import ORIGINAL, engineer_that_edits, make_graph


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "Jump.cs").write_bytes(ORIGINAL.encode())
    return root


@pytest.fixture
def cli(monkeypatch, in_repo_root):
    """No real .env: the scripted models need no key, and none should leak in."""
    monkeypatch.setattr(nexus_cli, "load_env", lambda: None)
    monkeypatch.setattr(resume_cli, "load_env", lambda: None)


def use_team(monkeypatch, orchestrator, engineers=()):
    # resume finds the graph by name in langgraph.json, which points here too.
    monkeypatch.setattr(team, "graph", make_graph(orchestrator, list(engineers)))


def run(project, *extra):
    return nexus_cli.main(["Make jumps higher", "--project", str(project), "--thread", "t", *extra])


def test_change_waits_then_approve_applies_it(cli, monkeypatch, project, capsys):
    orch = ScriptedOrchestrator(
        decide("engineer", brief="Set Speed to 12f."),
        decide("finish", answer="Speed is now 12."),
    )
    use_team(monkeypatch, orch, [engineer_that_edits()])

    assert run(project) == 4
    out = capsys.readouterr()
    assert "Apply these changes to the project?" in out.out
    assert "+    float Speed = 12f;" in out.out
    assert "nexus.resume --thread t --decision approve" in out.err
    assert (project / "Jump.cs").read_bytes() == ORIGINAL.encode()

    assert resume_cli.main(["--thread", "t", "--decision", "approve"]) == 0
    out = capsys.readouterr()
    assert out.out.strip() == "Speed is now 12."
    assert "[completed] thread=t" in out.err
    assert b"12f" in (project / "Jump.cs").read_bytes()


def test_reject_passes_the_note_on_and_writes_nothing(cli, monkeypatch, project, capsys):
    orch = ScriptedOrchestrator(
        decide("engineer", brief="Set Speed to 12f."),
        decide("finish", answer="Stopped: the user wants 11f."),
    )
    use_team(monkeypatch, orch, [engineer_that_edits()])
    run(project)

    code = resume_cli.main(["--thread", "t", "--decision", "reject", "--note", "use 11f"])
    assert code == 0
    assert "Their note: use 11f" in orch.calls[-1][1][1][1]
    assert (project / "Jump.cs").read_bytes() == ORIGINAL.encode()


def test_failed_run_says_how_to_retry_and_resume_retries(cli, monkeypatch, project, capsys):
    orch = ScriptedOrchestrator()  # no decisions: the orchestrator step raises
    use_team(monkeypatch, orch)
    assert run(project) == 3
    assert "Retry the failed step: python -m nexus.resume --thread t" in capsys.readouterr().err

    orch.decisions.append(decide("finish", answer="Done."))
    assert resume_cli.main(["--thread", "t"]) == 0
    assert capsys.readouterr().out.strip() == "Done."


def test_json_prints_the_whole_result(cli, monkeypatch, project, capsys):
    use_team(monkeypatch, ScriptedOrchestrator(decide("finish", answer="Hi.")))
    assert run(project, "--json") == 0
    result = json.loads(capsys.readouterr().out)
    assert (result["status"], result["thread_id"], result["state"]["answer"]) == (
        "completed",
        "t",
        "Hi.",
    )


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["q", "--project", ".", "--pipeline", "ask", "--max-rounds", "2"], "team pipeline only"),
        (["q", "--project", "no/such/dir"], "project directory not found"),
        (["q", "--project", ".", "--max-rounds", "11"], "max_rounds"),
        (["q", "--project", ".", "--pipeline", "ask", "--budget", "1"], "team pipeline only"),
        (["q", "--project", ".", "--budget", "0"], "max_cost_usd"),
        (["q", "--project", ".", "--budget", "lots"], "invalid float value"),
    ],
)
def test_nexus_refuses_bad_arguments(cli, capsys, argv, message):
    with pytest.raises(SystemExit) as exit_info:
        nexus_cli.main(argv)
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--thread", "t", "--note", "x"], "--note goes with --decision"),
        (["--thread", "nope"], "no run with thread 'nope'"),
        (["--thread", "t", "--decision", "maybe"], "invalid choice"),
    ],
)
def test_resume_refuses_bad_arguments(cli, capsys, argv, message):
    with pytest.raises(SystemExit) as exit_info:
        resume_cli.main(argv)
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


def test_budget_flag_reaches_the_orchestrator(cli, monkeypatch, project):
    orch = ScriptedOrchestrator(decide("finish", answer="Hi."))
    use_team(monkeypatch, orch)
    assert run(project, "--budget", "0.5") == 0
    assert "<budget>spent $0.00 of $0.50; $0.50 left</budget>" in orch.calls[0][1][1][1]


def test_a_failed_run_shows_what_the_attempt_spent(cli, monkeypatch, project, capsys):
    from nexus.run import RunResult

    failed = RunResult("t", "failed", {"cost_usd": 0.25}, "scout", "boom", spent_usd=0.4)
    assert nexus_cli.show(failed) == 3
    err = capsys.readouterr().err
    assert "[failed] thread=t cost=$0.2500 (this attempt spent $0.4000" in err
