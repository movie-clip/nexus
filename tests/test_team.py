"""The team pipeline, with scripted models: no API calls, no cost."""

import json

import pytest

from nexus.errors import InputError
from nexus.models import PRICES
from nexus.pipelines import team
from nexus.run import run_pipeline
from tests.fakes import ScriptedOrchestrator, decide, final, scripted

REQUEST = "How does jumping work?"


@pytest.fixture
def project(tmp_path):
    (tmp_path / "Jump.cs").write_text("float JumpSpeed = 10f;\n")
    return tmp_path


def run_team(project, orchestrator, specialists: dict, **inputs):
    """Run the team graph; ``specialists`` maps role -> list of scripted models,
    one per dispatch of that role."""
    queues = {role: list(models) for role, models in specialists.items()}

    def model_for(role):
        return orchestrator if role == "orchestrator" else queues[role].pop(0)

    graph = team.build(model_for=model_for).compile()
    return run_pipeline(graph, "t", {"request": REQUEST, "project": str(project), **inputs})


def user_text(messages) -> str:
    """The orchestrator's user message, from (role, text) tuples."""
    return next(text for role, text in messages if role == "user")


# --- routing ------------------------------------------------------------------


def test_scout_then_finish(project):
    orch = ScriptedOrchestrator(
        decide("scout", brief="Find where JumpSpeed is defined."),
        decide("finish", answer="JumpSpeed is 10 (`Jump.cs:1`)."),
    )
    result = run_team(project, orch, {"scout": [scripted(final("Jump.cs:1 sets it to 10"))]})
    assert result.status == "completed"
    assert result.state["answer"] == "JumpSpeed is 10 (`Jump.cs:1`)."
    assert [r["role"] for r in result.state["reports"]] == ["scout"]
    assert result.state["reports"][0]["report"] == "Jump.cs:1 sets it to 10"


def test_scout_reviewer_finish(project):
    orch = ScriptedOrchestrator(
        decide("scout", brief="Find JumpSpeed."),
        decide("reviewer", brief="Check: JumpSpeed is 10 at Jump.cs:1."),
        decide("finish", answer="It is 10."),
    )
    result = run_team(
        project,
        orch,
        {
            "scout": [scripted(final("Jump.cs:1 = 10"))],
            "reviewer": [scripted(final("Accurate. Confirmed Jump.cs:1."))],
        },
    )
    assert [r["role"] for r in result.state["reports"]] == ["scout", "reviewer"]
    assert result.state["answer"] == "It is 10."


def test_orchestrator_sees_every_brief_and_report(project):
    orch = ScriptedOrchestrator(
        decide("scout", brief="BRIEF-ONE"),
        decide("finish", answer="done"),
    )
    run_team(project, orch, {"scout": [scripted(final("REPORT-ONE"))]})
    first, second = (user_text(messages) for _, messages in orch.calls)
    assert REQUEST in first and "none yet" in first
    assert "BRIEF-ONE" in second and "REPORT-ONE" in second


def test_specialist_sees_only_its_brief_not_the_request(project):
    scout = scripted(final("ok"))
    orch = ScriptedOrchestrator(
        decide("scout", brief="Self-contained brief."),
        decide("finish", answer="done"),
    )
    run_team(project, orch, {"scout": [scout]})
    sent = " ".join(str(m.content) for m in scout.seen_prompts[0])
    assert "Self-contained brief." in sent
    assert REQUEST not in sent


# --- round cap ----------------------------------------------------------------


def test_round_cap_offers_only_finish(project):
    orch = ScriptedOrchestrator(
        decide("scout", brief="look"),
        decide("scout", brief="look again"),
        decide("finish", answer="best effort"),
    )
    result = run_team(
        project,
        orch,
        {"scout": [scripted(final("r1")), scripted(final("r2"))]},
        max_rounds=2,
    )
    schemas = [schema for schema, _ in orch.calls]
    assert schemas == [team.Decision, team.Decision, team.FinalDecision]
    assert "No rounds are left" in user_text(orch.calls[-1][1])
    assert result.state["answer"] == "best effort"


def test_zero_rounds_answers_without_specialists(project):
    orch = ScriptedOrchestrator(decide("finish", answer="from request alone"))
    result = run_team(project, orch, {}, max_rounds=0)
    assert [schema for schema, _ in orch.calls] == [team.FinalDecision]
    assert result.state.get("reports", []) == []


def test_final_schema_cannot_express_another_round():
    schema = team.FinalDecision.model_json_schema()
    assert schema["properties"]["next"]["const"] == "finish"
    # The schema sent to the API: every field required, nothing extra allowed.
    assert set(schema["required"]) == {"next", "reason", "brief", "answer"}


# --- malformed decisions fail clearly -----------------------------------------


@pytest.mark.parametrize(
    ("decision", "message"),
    [
        (decide("scout", brief="  "), "chose scout without a brief"),
        (decide("finish", answer=""), "finished without an answer"),
    ],
)
def test_incomplete_decision_fails_the_run(project, decision, message):
    result = run_team(project, ScriptedOrchestrator(decision), {})
    assert result.status == "failed"
    assert result.failed_node == "orchestrator"
    assert message in result.error


# --- inputs, models, cost -----------------------------------------------------


@pytest.mark.parametrize("bad", [{"max_rounds": 11}, {"max_rounds": -1}, {"max_rounds": "x"}])
def test_bad_max_rounds_rejected(project, bad):
    with pytest.raises(InputError, match="max_rounds"):
        run_pipeline(team.graph, "t", {"request": "x", "project": str(project), **bad})


def test_role_models():
    assert team.ORCHESTRATOR.model == "claude-opus-5-5"
    assert team.SPECIALISTS["scout"].model == "claude-sonnet-5-5"  # the user's choice
    assert team.SPECIALISTS["reviewer"].model == "claude-opus-5-5"
    for role in ["orchestrator", *team.SPECIALISTS]:
        assert (
            team.default_model(role).model
            == (team.ORCHESTRATOR if role == "orchestrator" else team.SPECIALISTS[role]).model
        )


def test_cost_sums_specialists_at_their_own_model_rates(project):
    orch = ScriptedOrchestrator(decide("scout", brief="b"), decide("finish", answer="a"))
    scout = scripted(final("r", 1_000_000, 100_000, model="claude-sonnet-5-5"))
    result = run_team(project, orch, {"scout": [scout]})
    sonnet = PRICES["claude-sonnet-5-5"]
    assert result.state["cost_usd"] == pytest.approx(sonnet.input + 0.1 * sonnet.output)


def test_registered_in_langgraph_json():
    from tests.conftest import ROOT

    graphs = json.loads((ROOT / "langgraph.json").read_text())["graphs"]
    assert graphs["team"] == "./src/nexus/pipelines/team.py:graph"
