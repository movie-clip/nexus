# Skills

Agent Skills (`<name>/SKILL.md`, the open format at agentskills.io) that nexus's
specialists can read when a task matches. Each role is given only its own
skills (`skills` on its `Role` in `nexus.pipelines.team`); an agent sees their
names and descriptions, and reads a skill in full only when it needs it.

## Source

Copied from [mattpocock/skills](https://github.com/mattpocock/skills) at commit
`49dd158d1076134a641b33efb035946536778336` (2026-10-09), `skills/engineering/`.
MIT licence: see [LICENSE](LICENSE). Copied, not installed, so they change only
when we change them; read every file before updating them.

Left out: each skill's `agents/openai.yaml` (Codex display metadata).

## nexus changes

nexus's specialists have no shell, no version control, no sub-agents and no
issue tracker, and can't ask the user anything while they work. Where a skill
relied on those, it was changed:

- `codebase-design`: verbatim, except the link to `DESIGN-IT-TWICE.md` (parallel
  sub-agents and a conversation with the user) is removed, and that file is not
  copied.
- `tdd`: the description names the brief; seams come from the brief (or the
  engineer picks the fewest and lists them in its report) instead of being
  confirmed with the user; tests are written before code but not run, because
  the engineer can't run them; points to the `codebase-design` skill by name
  instead of a Skill tool. `tests.md` and `mocking.md` are verbatim.
- `code-review`: rewritten around a change named in the brief instead of
  `git diff` against a fixed point; one agent does the two axes as separate
  passes instead of parallel sub-agents; the spec is what the brief states, not
  an issue tracker; `AGENTS.md`/`CLAUDE.md` count as standards. The smell
  baseline and "Why two axes" are verbatim (one word each: "change" for "diff",
  "what was asked" for "what the issue asked").
