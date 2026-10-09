"""The skills library nexus ships (src/nexus/skills): each skill is one that
Deep Agents can load and nexus's specialists can follow."""

import re
from pathlib import Path

import pytest

# Deep Agents' own parser and name check, so the tests judge a skill the way
# loading it does (private: no public API checks a single SKILL.md).
from deepagents.middleware.skills import _parse_skill_metadata, _validate_skill_name

import nexus

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
