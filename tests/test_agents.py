"""Deep Agents' own files (offloaded messages, tool results, history) never
reach the project: they bypass permissions, so they are kept in memory."""

import pytest

from nexus.agents import (
    LAST_STEP,
    SCRATCH,
    StepLimit,
    read_only_agent,
    run_agent,
    staged_engineer,
)
from tests.fakes import final, scripted, tool_call

# Over Deep Agents' limit for one message (50k tokens at 4 characters each):
# the agent saves the message to a file and keeps a short preview instead.
HUGE_BRIEF = "Check this draft:\n" + "x" * 210_000


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "A.cs").write_text("class A {}\n")
    return root


def files_in(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))


def test_read_only_agent_keeps_offloaded_messages_out_of_the_project(project):
    model = scripted(final("Done."))
    limit = StepLimit(5)
    agent = read_only_agent(str(project), model, "You review.", "reviewer", limit)
    text, _ = run_agent(agent, HUGE_BRIEF, limit)
    assert text == "Done."
    assert files_in(project) == ["A.cs"]
    # The model saw the preview and where the full text went: the scratch area.
    sent = model.seen_prompts[0][-1].text
    assert f"{SCRATCH}conversation_history/" in sent
    assert len(sent) < 50_000


def test_offloaded_message_can_still_be_read_back(project):
    model = scripted(tool_call("ls", path=SCRATCH + "conversation_history"), final("Done."))
    limit = StepLimit(5)
    agent = read_only_agent(str(project), model, "You review.", "reviewer", limit)
    run_agent(agent, HUGE_BRIEF, limit)
    listing = model.seen_prompts[1][-1].text
    assert ".md" in listing
    assert files_in(project) == ["A.cs"]


def test_engineer_batch_never_contains_offloaded_files(project, tmp_path):
    model = scripted(
        tool_call("write_file", file_path="/B.cs", content="class B {}\n"),
        final("Added B."),
    )
    staging = tmp_path / "staging"
    limit = StepLimit(5)
    agent, backend = staged_engineer(str(project), str(staging), model, "You edit.", limit)
    run_agent(agent, HUGE_BRIEF, limit)
    assert [change.path for change in backend.changes()] == ["/B.cs"]
    assert files_in(project) == ["A.cs"]
    assert files_in(staging) == ["B.cs"]


# --- step limits ------------------------------------------------------------------


def searches(n: int):
    return [tool_call("grep", pattern=f"thing{i}") for i in range(n)]


def test_last_allowed_call_asks_for_the_report_without_tools(project):
    model = scripted(*searches(2), final("Found thing0 and thing1; thing2 unchecked."))
    limit = StepLimit(3)
    agent = read_only_agent(str(project), model, "You scout.", "scout", limit)
    text, _ = run_agent(agent, "Find things.", limit)

    assert len(model.seen_prompts) == 3
    assert model.seen_tool_choice[:2] == [None, None]
    assert model.seen_tool_choice[2] == {"type": "none"}
    assert model.seen_prompts[2][-1].text == LAST_STEP
    assert text.startswith("Found thing0 and thing1")
    assert text.endswith(
        "(Stopped at the step limit of 3 model calls; this report may be incomplete.)"
    )


def test_a_model_that_ignores_the_last_step_is_stopped_anyway(project):
    model = scripted(*searches(10))  # never writes a report
    limit = StepLimit(3)
    agent = read_only_agent(str(project), model, "You scout.", "scout", limit)
    text, _ = run_agent(agent, "Find things.", limit)
    assert len(model.seen_prompts) == 3  # the hard stop: no fourth call
    assert "step limit of 3 model calls" in text


def test_an_agent_within_its_limit_is_untouched(project):
    model = scripted(*searches(1), final("Done."))
    limit = StepLimit(3)
    agent = read_only_agent(str(project), model, "You scout.", "scout", limit)
    text, _ = run_agent(agent, "Find things.", limit)
    assert (text, limit.reached) == ("Done.", False)
    assert model.seen_tool_choice == [None, None]
    assert all(LAST_STEP not in m.text for m in model.seen_prompts[-1])


def test_the_engineer_has_a_step_limit_too(project, tmp_path):
    model = scripted(*searches(1), final("Changed nothing."))
    limit = StepLimit(2)
    agent, _ = staged_engineer(str(project), str(tmp_path / "s"), model, "You edit.", limit)
    text, _ = run_agent(agent, "Edit.", limit)
    assert model.seen_tool_choice == [None, {"type": "none"}]
    assert "step limit of 2 model calls" in text


def test_a_limit_must_allow_one_call():
    with pytest.raises(ValueError, match="at least 1"):
        StepLimit(0)


def test_hide_tools_also_hides_tools_given_as_dicts():
    from langchain.agents.middleware import ModelRequest

    from nexus.agents import HideTools

    seen = []
    request = ModelRequest(
        model=scripted(),
        messages=[],
        tools=[{"name": "task", "description": "d"}, {"name": "grep", "description": "d"}],
    )
    HideTools(frozenset({"task"})).wrap_model_call(request, lambda r: seen.append(r.tools))
    assert seen == [[{"name": "grep", "description": "d"}]]
