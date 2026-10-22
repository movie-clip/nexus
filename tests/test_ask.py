"""The ask pipeline, with a scripted model: no API calls, no cost."""

import inspect
import logging

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.messages.ai import UsageMetadata

from nexus.agents import StepLimit
from nexus.errors import InputError
from nexus.models import PRICES, UNKNOWN_PRICE, cost_usd
from nexus.pipelines import ask
from nexus.run import run_pipeline
from tests.fakes import MODEL, final, scripted, tool_call


def run_agent(project, model):
    out = ask.build_agent(str(project), model, StepLimit(ask.MAX_MODEL_CALLS)).invoke(
        {"messages": [{"role": "user", "content": "go"}]}
    )
    return out["messages"]


def tool_results(messages) -> list[str]:
    return [m.text for m in messages if isinstance(m, ToolMessage)]


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "README.md").write_text("The jump height is 2 meters.\n")
    (root / ".env").write_text("SECRET=hunter2\n")
    (tmp_path / "outside.txt").write_text("outside the project\n")
    return root


# --- what the agent can and cannot do -----------------------------------------


def test_agent_reads_project_files(project):
    messages = run_agent(
        project, scripted(tool_call("read_file", file_path="/README.md"), final("2 meters"))
    )
    assert "jump height is 2 meters" in tool_results(messages)[0]
    assert messages[-1].text == "2 meters"


def test_write_tools_are_not_offered_and_no_shell_exists(project):
    model = scripted(final("ok"))
    run_agent(project, model)
    offered = set(model.seen_tools[0])
    assert {"ls", "read_file", "glob", "grep"} <= offered
    assert not offered & {"write_file", "edit_file", "delete", "execute"}


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("write_file", {"file_path": "/new.txt", "content": "x"}),
        ("task", {"description": "write /new.txt", "subagent_type": "general-purpose"}),
    ],
)
def test_hidden_tools_are_refused_if_called(project, name, args):
    messages = run_agent(project, scripted(tool_call(name, **args), final("tried")))
    assert tool_results(messages)[0] == f"Error: the {name} tool is not available."
    assert not (project / "new.txt").exists()


def test_read_only_permissions_deny_writes_on_their_own(project):
    """The second layer: without HideTools, the permission rule still denies."""
    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend

    from nexus.agents import READ_ONLY_PERMISSIONS

    model = scripted(tool_call("write_file", file_path="/new.txt", content="x"), final("tried"))
    agent = create_deep_agent(
        model=model,
        backend=FilesystemBackend(root_dir=project, virtual_mode=True),
        permissions=READ_ONLY_PERMISSIONS,
    )
    out = agent.invoke({"messages": [{"role": "user", "content": "go"}]})
    assert tool_results(out["messages"])[0] == "Error: permission denied for write on /new.txt"
    assert not (project / "new.txt").exists()


@pytest.mark.parametrize("path", ["/../outside.txt", "../outside.txt"])
def test_paths_outside_the_project_are_blocked(project, path):
    messages = run_agent(project, scripted(tool_call("read_file", file_path=path), final("x")))
    assert tool_results(messages)[0].startswith("Error: Path traversal not allowed")


@pytest.mark.parametrize("path", ["/.env", "/sub/.env", "/sub/.env.local"])
def test_env_files_are_unreadable(project, path):
    sub = project / "sub"
    sub.mkdir()
    (sub / ".env").write_text("SECRET=hunter2\n")
    (sub / ".env.local").write_text("SECRET=hunter2\n")
    messages = run_agent(project, scripted(tool_call("read_file", file_path=path), final("x")))
    assert tool_results(messages)[0] == f"Error: permission denied for read on {path}"


def test_project_agents_md_is_loaded_as_memory(project):
    (project / "AGENTS.md").write_text("Unity project; run dotnet build to compile.\n")
    model = scripted(final("ok"))
    run_agent(project, model)
    prompt_text = " ".join(str(m.content) for m in model.seen_prompts[0])
    assert "run dotnet build to compile" in prompt_text


# --- the pipeline -------------------------------------------------------------


def test_default_model_runs_at_medium_effort():
    factory = inspect.signature(ask.build).parameters["model_factory"].default
    assert factory().effort == "medium"


def test_pipeline_returns_answer_and_cost(project):
    graph = ask.build(model_factory=lambda: scripted(final("2 meters", 1000, 200))).compile()
    result = run_pipeline(graph, "t", {"request": "how high?", "project": str(project)})
    assert result.status == "completed"
    assert result.state["answer"] == "2 meters"
    assert result.state["cost_usd"] == pytest.approx((1000 * 4.00 + 200 * 20.00) / 1e6)


def test_pipeline_rejects_missing_project_before_any_model_call(tmp_path):
    def no_model():
        raise AssertionError("model must not be created")

    graph = ask.build(model_factory=no_model).compile()
    with pytest.raises(InputError, match="project directory not found"):
        run_pipeline(graph, "t", {"request": "x", "project": str(tmp_path / "nope")})
    with pytest.raises(InputError):
        run_pipeline(graph, "t", {"project": str(tmp_path)})  # request is required


# --- cost ---------------------------------------------------------------------


def test_cost_counts_cache_reads_and_writes_at_their_own_rates():
    usage: dict[str, UsageMetadata] = {
        MODEL: {
            "input_tokens": 10_000,  # includes the cache tokens below
            "output_tokens": 1_000,
            "total_tokens": 11_000,
            "input_token_details": {"cache_read": 6_000, "cache_creation": 2_000},
        }
    }
    p = PRICES[MODEL]
    expected = 2_000 * p.input + 6_000 * p.cache_read + 2_000 * p.cache_write + 1_000 * p.output
    assert cost_usd(usage) == pytest.approx(expected / 1e6)


def test_cost_matches_longest_model_id_first():
    usage: UsageMetadata = {
        "input_tokens": 1_000_000,
        "output_tokens": 0,
        "total_tokens": 1_000_000,
    }
    assert cost_usd({"claude-opus-5-5": usage}) == pytest.approx(4.00)
    assert cost_usd({"claude-opus-5": usage}) == pytest.approx(5.00)


@pytest.mark.parametrize(
    "model, input_usd, output_usd",
    [
        # Haiku 5.5 costs more above 100K prompt tokens; usage is summed over
        # calls, so nexus uses the higher rate: it may overstate, never understate.
        ("claude-haiku-5-5", 0.50, 2.50),
        ("claude-sonnet-4-6", 3.00, 15.00),
        ("claude-opus-4-7", 5.00, 25.00),
        ("claude-opus-4-6", 5.00, 25.00),
        ("claude-mythos-5-1", 10.00, 50.00),
    ],
)
def test_every_current_model_has_its_own_price(model, input_usd, output_usd, caplog):
    usage: UsageMetadata = {
        "input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
        "total_tokens": 2_000_000,
    }
    with caplog.at_level(logging.WARNING, logger="nexus.models"):
        assert cost_usd({model: usage}) == pytest.approx(input_usd + output_usd)
    assert "no price" not in caplog.text


def test_cache_writes_cost_a_quarter_more_than_input():
    for price in PRICES.values():
        assert price.cache_write == pytest.approx(1.25 * price.input)


def test_unknown_model_is_priced_high_not_free(caplog):
    usage: UsageMetadata = {
        "input_tokens": 0,
        "output_tokens": 1_000_000,
        "total_tokens": 1_000_000,
    }
    with caplog.at_level(logging.WARNING, logger="nexus.models"):
        assert cost_usd({"claude-mystery-9": usage}) == pytest.approx(UNKNOWN_PRICE.output)
    assert "claude-mystery-9" in caplog.text
