---
name: code-review
description: "Review a change to the code along two axes: Standards (does the code follow this project's documented coding standards?) and Spec (does the code match what was asked for?), and report them side by side. Use when the brief asks you to review a change, or names changed files to check."
---

Two-axis review of a change already made to the project:

- **Standards**: does the code conform to this project's documented coding standards?
- **Spec**: does the code faithfully implement what was asked for?

Do the two axes as separate passes and report them separately, so one doesn't colour the other.

## Process

### 1. Pin the change

Your brief names the change: the files it touched, and what changed in each. You have no version control history, so the brief and the files as they are now are the change. Read every changed file in full before judging it. If the brief doesn't say which files changed, say so at the top of your report and review only what you can identify from it.

### 2. Identify the spec

The spec is what the change was asked to do, as your brief states it: the user's request, a plan, or a list of requirements. If the brief points to a spec file in the project (under `docs/`, `specs/` or similar), read it. If there is no spec, skip the Spec pass and say "no spec available" in the report.

### 3. Identify the standards sources

Search the project for every file that documents how code should be written. When `CODING_STANDARDS.md` or `CONTRIBUTING.md` exists, it must be on the list, and so must `AGENTS.md` or `CLAUDE.md` when present.

On top of whatever the project documents, the Standards axis always carries the **smell baseline** below: a fixed set of Fowler code smells (_Refactoring_, ch.3) that applies even when a project documents nothing. Two rules bind it:

- **The project overrides.** A documented project standard always wins; where it endorses something the baseline would flag, suppress the smell.
- **Always a judgement call.** Each smell is a labelled heuristic ("possible Feature Envy"), never a hard violation. Like any standard here, skip anything tooling already enforces.

Each smell reads *what it is* → *how to fix*; match it against the change:

- **Mysterious Name**: a function, variable, or type whose name doesn't reveal what it does or holds. → rename it; if no honest name comes, the design's murky.
- **Duplicated Code**: the same logic shape appears in more than one hunk or file in the change. → extract the shared shape, call it from both.
- **Feature Envy**: a method that reaches into another object's data more than its own. → move the method onto the data it envies.
- **Data Clumps**: the same few fields or params keep travelling together (a type wanting to be born). → bundle them into one type, pass that.
- **Primitive Obsession**: a primitive or string standing in for a domain concept that deserves its own type. → give the concept its own small type.
- **Repeated Switches**: the same `switch`/`if`-cascade on the same type recurs across the change. → replace with polymorphism, or one map both sites share.
- **Shotgun Surgery**: one logical change forces scattered edits across many files in the change. → gather what changes together into one module.
- **Divergent Change**: one file or module is edited for several unrelated reasons. → split so each module changes for one reason.
- **Speculative Generality**: abstraction, parameters, or hooks added for needs the spec doesn't have. → delete it; inline back until a real need shows.
- **Message Chains**: long `a.b().c().d()` navigation the caller shouldn't depend on. → hide the walk behind one method on the first object.
- **Middle Man**: a class or function that mostly just delegates onward. → cut it, call the real target direct.
- **Refused Bequest**: a subclass or implementer that ignores or overrides most of what it inherits. → drop the inheritance, use composition.

### 4. Review each axis

Do the Standards pass first, then the Spec pass. Keep each pass to its own question.

**Standards**: report, per file and place where relevant, (a) every place the change violates a documented standard: cite the standard (file and rule); and (b) any baseline smell you spot: name it and quote the code. Distinguish hard violations from judgement calls: documented-standard breaches can be hard, but baseline smells are always judgement calls, and a documented project standard overrides the baseline. Skip anything tooling enforces.

**Spec**: report (a) requirements the spec asked for that are missing or partial; (b) behaviour in the change that wasn't asked for (scope creep); (c) requirements that look implemented but where the implementation looks wrong. Quote the spec line for each finding.

Cite every finding as `path:line`.

### 5. Report

Present the two passes under `## Standards` and `## Spec` headings. Do **not** merge or rerank findings, because the two axes are deliberately separate (see _Why two axes_).

End with a one-line summary: total findings per axis, and the worst issue _within each axis_ (if any). Don't pick a single winner across axes: that's the reranking the separation exists to prevent.

## Why two axes

A change can pass one axis and fail the other:

- Code that follows every standard but implements the wrong thing → **Standards pass, Spec fail.**
- Code that does exactly what was asked but breaks the project's conventions → **Spec pass, Standards fail.**

Reporting them separately stops one axis from masking the other.
