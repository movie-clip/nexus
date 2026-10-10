"""The team pipeline, with scripted models: no API calls, no cost."""

import json
import logging

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage

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
def test_incomplete_decision_is_sent_back_to_the_model(project, decision, message):
    orch = ScriptedOrchestrator(decision, decide("finish", answer="Fixed."))
    result = run_team(project, orch, {})
    assert (result.status, result.state["answer"]) == ("completed", "Fixed.")
    retry = orch.calls[1][1]
    assert retry[-2][0] == "assistant"  # what it said, then what was wrong with it
    assert retry[-1] == ("user", f"That decision can't be carried out: {message}. Decide again.")


@pytest.mark.parametrize(
    ("decision", "message"),
    [
        (decide("scout", brief="  "), "chose scout without a brief"),
        (decide("tester", command=" "), "chose tester without a command"),
    ],
)
def test_a_decision_still_incomplete_after_two_fixes_fails_the_run(project, decision, message):
    orch = ScriptedOrchestrator(decision, decision, decision)
    result = run_team(project, orch, {})
    assert (result.status, result.failed_node) == ("failed", "orchestrator")
    assert result.error == f"ValueError: orchestrator {message}, 3 times"
    assert len(orch.calls) == 3


# --- inputs, models, cost -----------------------------------------------------


@pytest.mark.parametrize("bad", [{"max_rounds": 11}, {"max_rounds": -1}, {"max_rounds": "x"}])
def test_bad_max_rounds_rejected(project, bad):
    with pytest.raises(InputError, match="max_rounds"):
        run_pipeline(team.graph, "t", {"request": "x", "project": str(project), **bad})


def test_role_models():
    # The user's choice during development: every role on Sonnet 5.5.
    assert team.ORCHESTRATOR.model == "claude-sonnet-5-5"
    assert all(team.role_spec(r).model == "claude-sonnet-5-5" for r in team.SPECIALISTS)
    limits = {r: team.role_spec(r).max_model_calls for r in ["orchestrator", *team.SPECIALISTS]}
    assert limits == {"orchestrator": 1, "scout": 30, "reviewer": 30, "engineer": 60}
    for role in ["orchestrator", *team.SPECIALISTS]:
        model = team.default_model(role)
        assert isinstance(model, ChatAnthropic)
        assert model.model == team.role_spec(role).model
        # The user's choice (2026-10-10): every role at medium effort.
        assert model.effort == team.role_spec(role).effort == "medium"


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


def test_an_orchestrator_reply_that_is_not_a_decision_fails_the_round(tmp_path):
    from langchain_core.runnables import RunnableLambda

    class Raw:
        """Returns a plain dict instead of the schema object."""

        def with_structured_output(self, schema, method):
            return RunnableLambda(lambda messages: {"next": "finish"})

    graph = team.build(model_for=lambda role: Raw()).compile()  # type: ignore[arg-type,return-value]
    result = run_pipeline(graph, "t", {"request": "q", "project": str(tmp_path)})
    assert (result.status, result.failed_node) == ("failed", "orchestrator")
    assert result.error == "TypeError: orchestrator returned dict, not a decision"


# --- budget -------------------------------------------------------------------


def test_orchestrator_sees_the_budget(project):
    orch = ScriptedOrchestrator(decide("finish", answer="a"))
    run_team(project, orch, {})
    assert "<budget>spent $0.00 of $2.00; $2.00 left</budget>" in user_text(orch.calls[0][1])


def test_spent_budget_offers_only_finish(project):
    orch = ScriptedOrchestrator(
        decide("scout", brief="look"),
        decide("finish", answer="best effort"),
    )
    # $0.10 of output at Sonnet 5.5's rate, against a $0.05 budget.
    scout = scripted(final("r1", 0, 10_000, model="claude-sonnet-5-5"))
    result = run_team(project, orch, {"scout": [scout]}, max_cost_usd=0.05)
    assert [schema for schema, _ in orch.calls] == [team.Decision, team.FinalDecision]
    message = user_text(orch.calls[-1][1])
    assert "<budget>spent $0.10 of $0.05; $0.00 left</budget>" in message
    assert "The budget is spent: finish now" in message
    assert result.state["answer"] == "best effort"


def test_rounds_and_budget_left_allow_another_round(project):
    limits = team.Limits(rounds_left=1, spent_usd=1.99, budget_usd=2.0)
    assert limits.can_dispatch and limits.money_left == pytest.approx(0.01)
    assert not team.Limits(rounds_left=0, spent_usd=0, budget_usd=2.0).can_dispatch
    assert not team.Limits(rounds_left=3, spent_usd=2.0, budget_usd=2.0).can_dispatch


@pytest.mark.parametrize("bad", [0, -1, 21, "x"])
def test_bad_budget_rejected(project, bad):
    with pytest.raises(InputError, match="max_cost_usd"):
        run_pipeline(
            team.graph, "t", {"request": "x", "project": str(project), "max_cost_usd": bad}
        )


# --- live cost ----------------------------------------------------------------


def priced_search(output_tokens: int) -> AIMessage:
    """A tool call that cost something (Sonnet 5.5 output tokens)."""
    return AIMessage(
        content="",
        tool_calls=[{"name": "ls", "args": {"path": "/"}, "id": "call-ls"}],
        response_metadata={"model_name": "claude-sonnet-5-5"},
        usage_metadata={"input_tokens": 0, "output_tokens": output_tokens, "total_tokens": 0},
    )


def test_every_model_call_is_priced_and_logged_as_it_ends(project, caplog):
    orch = ScriptedOrchestrator(decide("scout", brief="b"), decide("finish", answer="a"))
    scout = scripted(priced_search(10_000), final("r", 0, 20_000, model="claude-sonnet-5-5"))
    with caplog.at_level(logging.INFO, logger="nexus.models"):
        result = run_team(project, orch, {"scout": [scout]})
    assert result.spent_usd == pytest.approx(result.state["cost_usd"])
    assert result.spent_usd == pytest.approx(0.30)  # 30k output tokens at $10/M
    lines = [r.getMessage() for r in caplog.records if r.name == "nexus.models"]
    assert lines == [
        "model call 1 (scout): $0.1000, run so far $0.1000",
        "model call 2 (scout): $0.2000, run so far $0.3000",
    ]


def test_a_failed_step_still_counts_what_it_spent(project):
    orch = ScriptedOrchestrator(decide("scout", brief="b"))
    scout = scripted(priced_search(10_000))  # then the script runs out: the step fails
    result = run_team(project, orch, {"scout": [scout]})
    assert (result.status, result.failed_node) == ("failed", "scout")
    assert result.state["cost_usd"] == 0  # the failed step's cost never reached state
    assert result.spent_usd == pytest.approx(0.10)
