"""The skills library nexus ships (src/nexus/skills): each skill is one that
Deep Agents can load and nexus's specialists can follow."""

import re
from pathlib import Path

import pytest

# Deep Agents' own parser and name check, so the tests judge a skill the way
# loading it does (private: no public API checks a single SKILL.md).
from deepagents.middleware.skills import _parse_skill_metadata, _validate_skill_name

import nexus
from nexus.agents import (
    SKILLS_MOUNT,
    StepLimit,
    read_only_agent,
    run_agent,
    skill_shelf,
    staged_engineer,
)
from nexus.pipelines import team
from nexus.run import open_runs_db, run_pipeline
from tests.fakes import ScriptedOrchestrator, decide, final, scripted, tool_call

LIBRARY = Path(nexus.__file__).parent / "skills"
SKILLS = sorted(p.name for p in LIBRARY.iterdir() if p.is_dir())
# Things nexus's specialists don't have; a skill that relies on them needs a
# nexus version (see src/nexus/skills/README.md).
UNAVAILABLE = re.compile(
    r"sub-?agent|git (diff|log|rev-parse)|issue tracker|Skill tool|/setup-"
    r"|confirm them with the user",
    re.IGNORECASE,
)


def test_the_library_has_the_chosen_skills():
    assert SKILLS == ["code-review", "codebase-design", "tdd"]


@pytest.mark.parametrize("skill", SKILLS)
def test_frontmatter_follows_the_agent_skills_rules(skill):
    text = (LIBRARY / skill / "SKILL.md").read_text(encoding="utf-8")
    meta = _parse_skill_metadata(text, f"/{skill}/SKILL.md", skill)
    assert meta is not None, "Deep Agents would skip it"
    # Deep Agents only warns about these; the spec requires them.
    assert meta["name"] == skill
    assert _validate_skill_name(meta["name"], skill) == (True, "")
    assert len(meta["description"]) <= 1024  # longer is cut off
    assert "<" not in meta["description"]


@pytest.mark.parametrize("skill", SKILLS)
def test_every_link_stays_inside_the_skill(skill):
    # A role is given copies of its skill folders only, so a link elsewhere
    # would point at nothing.
    folder = LIBRARY / skill
    for doc in folder.rglob("*.md"):
        for target in re.findall(r"\]\(([^)#]+)\)", doc.read_text(encoding="utf-8")):
            if "://" in target:
                continue
            resolved = (doc.parent / target).resolve()
            assert resolved.is_file(), f"{doc.name} links to missing {target}"
            assert resolved.is_relative_to(folder.resolve()), f"{doc.name} links outside: {target}"


@pytest.mark.parametrize("skill", SKILLS)
def test_no_skill_relies_on_what_specialists_lack(skill):
    for doc in (LIBRARY / skill).rglob("*.md"):
        found = UNAVAILABLE.search(doc.read_text(encoding="utf-8"))
        assert not found, f"{skill}/{doc.name}: {found.group(0)!r}"


# --- skills given to agents -------------------------------------------------------


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "A.cs").write_text("class A {}\n")
    return root


def system_prompt(model) -> str:
    return model.seen_prompts[0][0].text


def test_an_agent_is_told_only_about_its_own_skills(project):
    model = scripted(final("Done."))
    limit = StepLimit(5)
    with skill_shelf(("tdd",)) as shelf:
        agent = read_only_agent(str(project), model, "You review.", "reviewer", limit, shelf)
        run_agent(agent, "Check A.", limit)
    prompt = system_prompt(model)
    assert f"{SKILLS_MOUNT}tdd/SKILL.md" in prompt
    assert "code-review" not in prompt and "codebase-design" not in prompt


def test_an_agent_reads_a_skill_and_its_reference_files(project):
    model = scripted(
        tool_call("read_file", file_path=f"{SKILLS_MOUNT}tdd/SKILL.md", limit=1000),
        tool_call("read_file", file_path=f"{SKILLS_MOUNT}tdd/mocking.md"),
        final("Done."),
    )
    limit = StepLimit(5)
    with skill_shelf(("tdd", "codebase-design")) as shelf:
        agent = read_only_agent(str(project), model, "You review.", "reviewer", limit, shelf)
        run_agent(agent, "Check A.", limit)
    assert "# Test-Driven Development" in model.seen_prompts[1][-1].text
    assert "# When to Mock" in model.seen_prompts[2][-1].text


def test_an_agent_without_skills_has_no_skills_folder(project):
    model = scripted(tool_call("ls", path=SKILLS_MOUNT), final("Done."))
    limit = StepLimit(5)
    agent = read_only_agent(str(project), model, "You review.", "reviewer", limit)
    run_agent(agent, "Check A.", limit)
    assert "Skills" not in system_prompt(model)
    assert "tdd" not in model.seen_prompts[1][-1].text


def test_the_engineer_cannot_change_a_skill_and_skills_never_join_its_batch(project, tmp_path):
    skill = f"{SKILLS_MOUNT}tdd/SKILL.md"
    model = scripted(
        tool_call("write_file", file_path=f"{SKILLS_MOUNT}tdd/new.md", content="x"),
        tool_call("edit_file", file_path=skill, old_string="TDD", new_string="XXX"),
        tool_call("write_file", file_path="/B.cs", content="class B {}\n"),
        final("Added B."),
    )
    limit = StepLimit(6)
    original = (LIBRARY / "tdd" / "SKILL.md").read_bytes()
    with skill_shelf(("tdd",)) as shelf:
        assert shelf is not None
        agent, backend = staged_engineer(
            str(project), str(tmp_path / "staging"), model, "You edit.", limit, shelf
        )
        run_agent(agent, "Add B.", limit)
        assert (Path(shelf) / "tdd" / "SKILL.md").read_bytes() == original
        assert not (Path(shelf) / "tdd" / "new.md").exists()
    assert [change.path for change in backend.changes()] == ["/B.cs"]
    assert "permission denied" in model.seen_prompts[1][-1].text.lower()
    assert "permission denied" in model.seen_prompts[2][-1].text.lower()
    assert (LIBRARY / "tdd" / "SKILL.md").read_bytes() == original


def test_a_shelf_holds_copies_of_the_named_skills_and_is_removed_after():
    with skill_shelf(("tdd", "code-review")) as shelf:
        assert shelf is not None
        assert sorted(p.name for p in Path(shelf).iterdir()) == ["code-review", "tdd"]
        assert (Path(shelf) / "tdd" / "tests.md").is_file()
    assert not Path(shelf).exists()
    with skill_shelf(()) as none:
        assert none is None


def test_an_unknown_skill_is_refused():
    with pytest.raises(ValueError, match="no skill 'grill-me'"), skill_shelf(("grill-me",)):
        pass


def test_each_role_gets_its_skills(project):
    assert team.ENGINEER.skills == ("tdd", "codebase-design")
    assert team.READ_ONLY_SPECIALISTS["reviewer"].skills == ("code-review", "codebase-design")
    assert team.READ_ONLY_SPECIALISTS["scout"].skills == ()

    orch = ScriptedOrchestrator(
        decide("scout", brief="Look."),
        decide("reviewer", brief="Review A.cs."),
        decide("engineer", brief="Add B."),
    )
    models = {r: scripted(final("ok")) for r in ("scout", "reviewer", "engineer")}

    def model_for(role):
        return orch if role == "orchestrator" else models[role]

    graph = team.build(model_for=model_for).compile(name="team")
    inputs = {"request": "Review and fix.", "project": str(project)}
    run_pipeline(graph, "t", inputs, open_runs_db())
    seen = {role: system_prompt(model) for role, model in models.items()}
    assert "/SKILL.md" not in seen["scout"]
    assert f"{SKILLS_MOUNT}code-review/SKILL.md" in seen["reviewer"]
    assert f"{SKILLS_MOUNT}tdd/SKILL.md" not in seen["reviewer"]
    assert f"{SKILLS_MOUNT}tdd/SKILL.md" in seen["engineer"]
    assert f"{SKILLS_MOUNT}code-review/SKILL.md" not in seen["engineer"]
    for role in ("reviewer", "engineer"):
        assert f"{SKILLS_MOUNT}codebase-design/SKILL.md" in seen[role]


@pytest.mark.parametrize(
    ("role", "named"),
    [(team.READ_ONLY_SPECIALISTS["reviewer"], {"code-review"}), (team.ENGINEER, {"tdd"})],
)
def test_a_prompt_only_names_skills_its_role_carries(role, named):
    # Prompts tell the agent to read these in full; a skill missing from the
    # role would send it looking for a file that isn't there.
    assert set(re.findall(r"the ([a-z-]+) skill", role.prompt)) == named
    assert named <= set(role.skills)
