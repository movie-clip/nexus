"""Deep Agents' own files (offloaded messages, tool results, history) never
reach the project: they bypass permissions, so they are kept in memory."""

import pytest

from nexus.agents import SCRATCH, read_only_agent, run_agent, staged_engineer
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
    agent = read_only_agent(str(project), model, "You review.", name="reviewer")
    text, _ = run_agent(agent, HUGE_BRIEF)
    assert text == "Done."
    assert files_in(project) == ["A.cs"]
    # The model saw the preview and where the full text went: the scratch area.
    sent = model.seen_prompts[0][-1].text
    assert f"{SCRATCH}conversation_history/" in sent
    assert len(sent) < 50_000


def test_offloaded_message_can_still_be_read_back(project):
    model = scripted(tool_call("ls", path=SCRATCH + "conversation_history"), final("Done."))
    agent = read_only_agent(str(project), model, "You review.", name="reviewer")
    run_agent(agent, HUGE_BRIEF)
    listing = model.seen_prompts[1][-1].text
    assert ".md" in listing
    assert files_in(project) == ["A.cs"]


def test_engineer_batch_never_contains_offloaded_files(project, tmp_path):
    model = scripted(
        tool_call("write_file", file_path="/B.cs", content="class B {}\n"),
        final("Added B."),
    )
    staging = tmp_path / "staging"
    agent, backend = staged_engineer(str(project), str(staging), model, "You edit.")
    run_agent(agent, HUGE_BRIEF)
    assert [change.path for change in backend.changes()] == ["/B.cs"]
    assert files_in(project) == ["A.cs"]
    assert files_in(staging) == ["B.cs"]
