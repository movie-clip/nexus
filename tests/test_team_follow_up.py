"""Follow-up messages on a finished team run: scripted models."""

import pytest

from nexus.models import PRICES
from nexus.pipelines import team
from nexus.run import open_runs_db, resume_pipeline, run_pipeline
from tests.fakes import ScriptedOrchestrator, decide, final, scripted
from tests.test_team import user_text
from tests.test_team_engineer import ORIGINAL, engineer_that_edits

REQUEST = "Review Jump.cs and plan the fixes first."


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "Jump.cs").write_bytes(ORIGINAL.encode())
    return root


def make_graph(orchestrator, specialists: dict):
    queues = {role: list(models) for role, models in specialists.items()}

    def model_for(role):
        return orchestrator if role == "orchestrator" else queues[role].pop(0)

    return team.build(model_for=model_for).compile(name="team")


def start(project, graph):
    inputs = {"request": REQUEST, "project": str(project)}
    return run_pipeline(graph, "t", inputs, open_runs_db())


def follow_up(graph, message):
    return resume_pipeline("t", None, open_runs_db(), graph=graph, message=message)


def test_a_follow_up_sees_the_conversation_and_every_report(project):
    orch = ScriptedOrchestrator(
        decide("scout", brief="Read Jump.cs."),
        decide("finish", answer="PLAN: 1. raise Speed."),
        decide("finish", answer="Noted."),
    )
    graph = make_graph(orch, {"scout": [scripted(final("SCOUT-REPORT"))]})
    start(project, graph)

    result = follow_up(graph, "Go with step 1.")
    assert result.status == "completed"
    assert result.state["answer"] == "Noted."
    assert result.state["request"] == "Go with step 1."
    assert result.state["turns"] == [
        {
            "request": REQUEST,
            "answer": "PLAN: 1. raise Speed.",
            "reports_at_end": 1,
            "cost_at_end": 0,
        }
    ]
    sent = user_text(orch.calls[-1][1])
    for earlier in (REQUEST, "PLAN: 1. raise Speed.", "SCOUT-REPORT"):
        assert earlier in sent
    new_request = "<request>\nGo with step 1.\n</request>"
    assert sent.index("PLAN: 1. raise Speed.") < sent.index(new_request)


def test_a_follow_up_can_send_the_engineer_without_scouting_again(project):
    orch = ScriptedOrchestrator(
        decide("finish", answer="PLAN: raise Speed to 12f."),
        decide("engineer", brief="Set Speed to 12f in /Jump.cs."),
    )
    graph = make_graph(orch, {"engineer": [engineer_that_edits()]})
    start(project, graph)

    result = follow_up(graph, "Go.")
    assert result.status == "waiting"
    assert result.waiting_for["files"] == ["/Jump.cs"]


def test_every_turn_is_kept_in_order(project):
    orch = ScriptedOrchestrator(
        decide("finish", answer="A1"),
        decide("finish", answer="A2"),
        decide("finish", answer="A3"),
    )
    graph = make_graph(orch, {})
    start(project, graph)
    follow_up(graph, "Q2")
    result = follow_up(graph, "Q3")
    assert [(t["request"], t["answer"]) for t in result.state["turns"]] == [
        (REQUEST, "A1"),
        ("Q2", "A2"),
    ]
    assert result.state["answer"] == "A3"


def test_a_new_run_cannot_start_with_a_follow_up(project):
    graph = make_graph(ScriptedOrchestrator(), {})
    inputs = {"request": REQUEST, "project": str(project), "follow_up": "Go."}
    result = run_pipeline(graph, "t", inputs, open_runs_db())
    assert result.status == "failed"
    assert "a follow-up needs a finished request" in str(result.error)


def test_each_follow_up_gets_its_own_rounds_and_budget(project):
    orch = ScriptedOrchestrator(
        decide("scout", brief="Read Jump.cs."),
        decide("finish", answer="PLAN"),
        decide("scout", brief="Check the plan's first step."),
        decide("finish", answer="Done"),
    )
    # $0.10 of output at Sonnet 5.5's rate, then a free report.
    scouts = [scripted(final("R1", 0, 10_000, model="claude-sonnet-5-5")), scripted(final("R2"))]
    graph = make_graph(orch, {"scout": scouts})
    inputs = {"request": REQUEST, "project": str(project), "max_rounds": 1, "max_cost_usd": 0.15}
    run_pipeline(graph, "t", inputs, open_runs_db())

    result = follow_up(graph, "Go.")
    schemas = [schema for schema, _ in orch.calls]
    assert schemas == [team.Decision, team.FinalDecision, team.Decision, team.FinalDecision]
    first_of_follow_up = user_text(orch.calls[2][1])
    assert "<rounds_left>1</rounds_left>" in first_of_follow_up
    assert "<budget>spent $0.00 of $0.15; $0.15 left</budget>" in first_of_follow_up
    # The run's cost still counts every turn.
    assert result.state["cost_usd"] == pytest.approx(0.01 * PRICES["claude-sonnet-5-5"].output)


def test_reports_say_which_turn_they_came_from(project):
    orch = ScriptedOrchestrator(
        decide("scout", brief="Read Jump.cs."),
        decide("finish", answer="PLAN"),
        decide("scout", brief="Check it."),
        decide("finish", answer="Done"),
    )
    graph = make_graph(orch, {"scout": [scripted(final("R1")), scripted(final("R2"))]})
    start(project, graph)
    follow_up(graph, "Go.")
    first_run, last = user_text(orch.calls[1][1]), user_text(orch.calls[-1][1])
    assert '<report n="1" role="scout">' in first_run  # one turn so far: no tag
    assert '<report n="1" role="scout" turn="1">' in last
    assert '<report n="2" role="scout" turn="2">' in last
