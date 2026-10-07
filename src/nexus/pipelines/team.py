"""Answer an engineering request with an orchestrator and specialist roles.

    START -> orchestrator -+- scout ---------------------------------------+
               ^           +- reviewer ------------------------------------+
               |           +- engineer -+- (no changes) -------------------+
               |           |            +- approve -> apply -+- (rejected) +
               |           |                                 +- verify ----+
               |           +- tester ---+- (bad command) ------------------+
               |           |            +- approve_command -+- (rejected) -+
               |           |            |                   +- run_command +
               |           |            +- (auto) -> run_command ----------+
               +-----------+--------------- reports back <-----------------+
                           +- finish -> END

Each round the orchestrator (one structured Claude call, no file access) reads
the request and every report so far and picks the next step: send a
specialist a self-contained brief, or finish with the final answer. The choice
comes back as JSON enforced by the API's structured output, and LangGraph routes
on it in plain code. When no rounds are left, the only schema offered is
"finish", so the round cap holds no matter what the model would prefer.

Scout and reviewer are read-only Deep Agents (``nexus.agents.read_only_agent``).
The engineer's writes are staged (``nexus.staging``): when it is done, the run
pauses at ``approve`` and shows the whole batch as one diff. The user approves
(``apply`` writes it to the project, skipping files changed on disk since) or
rejects with a note; either way the outcome goes back to the orchestrator as a
report from "user". After an approval, ``verify`` runs the project's ``verify``
commands (a compile, say) in order without asking, stopping at the first
failure; they report as "verify" and don't use up rounds. Specialists see only
their brief; the orchestrator sees every brief and report.

The tester is not a model: the orchestrator names one of the project's
configured commands (``nexus.commands``, ``projects/<name>.toml``) and its
param values; ``tester`` checks them, the run pauses at ``approve_command``
unless the command is set to ``approval = "auto"``, and ``run_command`` runs it
and reports a summary (errors, failed tests, files it changed, output tail).
"""

import difflib
import logging
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from operator import add
from pathlib import Path
from typing import Annotated, Literal, NotRequired, TypedDict

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import AfterValidator, BaseModel, Field

from nexus import commands
from nexus.agents import (
    StepLimit,
    existing_dir,
    read_only_agent,
    run_agent,
    staged_engineer,
)
from nexus.models import DEFAULT_MODEL, cost_usd, make_model
from nexus.staging import Change, apply_changes
from nexus.state import BaseState

log = logging.getLogger("nexus.team")

DEFAULT_MAX_ROUNDS = 6
# US dollars one run may spend (the user's choice, 2026-10-07). Checked before
# each round, so a run can go over by at most the round that crossed it, which
# its specialist's step limit bounds.
DEFAULT_BUDGET_USD = 2.0


@dataclass(frozen=True)
class Role:
    model: str
    effort: str
    prompt: str
    # Model calls one dispatch may take; the last one writes the report.
    # Simple fact-finding takes 3-10 (Anthropic's guide); the cap is for runaways.
    max_model_calls: int


ORCHESTRATOR = Role(
    model=DEFAULT_MODEL,
    effort="high",
    max_model_calls=1,  # one structured call per round
    prompt="""\
You lead a small team handling an engineering request about a software \
project. You cannot read or change the project yourself; your specialists can. \
Each round you choose one step.

Steps:
- scout: explores the repo (read-only) and reports what is actually there: \
where things live and how they work today, with `path:line` citations. Use it \
to gather facts.
- reviewer: checks a draft answer, a list of claims, or a change already \
applied to the project (read-only) and returns a verdict per point with \
evidence. Use it when the answer makes non-trivial claims, or after a \
non-trivial change was applied.
- engineer: changes the project. Use it only when the request asks for a \
change. Its edits are staged and shown to the user as one batch; the user \
approves (it is applied to the project) or rejects it with a note. You then \
get the outcome as a report from "user". The engineer cannot run builds or tests.
- tester: runs one of the project's commands listed under <commands> (builds, \
tests) and reports the result: errors, failed tests, files the command changed, \
and the end of its output. Set `command` to its name and `args` to values for \
its params (each must match the param's pattern; leave out a param to use its \
default or skip it). Only the listed commands exist. The user may have to \
approve the run, and may reject it with a note. Commands listed as run \
automatically after each approved change need no tester round: their results \
arrive as reports from "verify".
- finish: you write the final answer to the user.

Limits: <rounds_left> is how many more steps you may take, and <budget> what \
the run has spent and has left, in US dollars. When either runs out, only \
finish is offered, so plan to answer within them.

How to decide:
- Scale effort to the request. A narrow factual question: one scout round, then \
finish. A broad or high-stakes question: scout (again if a report leaves a \
gap), then reviewer on your draft, then finish. A change: scout first unless \
the request already names exactly what to change, then engineer.
- After an approved change, read the "verify" reports; don't run those \
commands again. If no "verify" report came, check the change with the \
tester's fastest build command when one exists. If a check fails because of \
the change, send the engineer a brief with the exact errors; if it fails for \
another reason (environment, time limit, unrelated failures), don't loop: \
report it in the answer. Send the reviewer to check a non-trivial change.
- Tests are written and run only when the request asks for them: then the \
engineer writes them (say so in its brief) and the tester runs the narrowest \
tests that cover the change. Don't run commands for a request that changes \
nothing unless it asks for that.
- After a rejection, send the engineer again with a brief that addresses the \
user's note, or finish and explain what was proposed and why it stopped.
- Don't repeat a step that already answered the question. If a report leaves a \
specific gap, send a narrower brief about that gap.
- Finish when every claim you will make is backed by a report. If something is \
still unverified when you finish, say so in the answer.

Writing a brief: the specialist sees only your brief, not the request or other \
reports. Make it self-contained: the objective, what is already known (with \
paths), where to start, what to return, and what is out of scope. For the \
reviewer, include the full draft or the claims to check. For the engineer, \
state exactly what to change, where, and any constraints.

The final answer: Markdown, the direct answer first, citing files as \
`path:line` from the reports. If changes were applied, list the files changed, \
what was built or tested and the result, and what the user should still check \
themselves. Don't mention the team, rounds or specialists.

Fields: `next`; `reason` (one sentence: why this step; the user sees it when \
asked to approve a command); `brief` (the task when next is scout, reviewer or \
engineer, otherwise ""); `command` and `args` (when next is tester, otherwise \
"" and []); `answer` (the final answer when next is finish, otherwise "").""",
)

READ_ONLY_SPECIALISTS: dict[str, Role] = {
    # Every role is on DEFAULT_MODEL (Sonnet 5.5) during development. When
    # tuning, the scout can stay on Sonnet: most of its work is reading.
    "scout": Role(
        model=DEFAULT_MODEL,
        effort="medium",
        max_model_calls=30,
        prompt="""\
You are the scout on an engineering team. Explore the project in your \
filesystem (`/` is the project root) to answer the brief you are given.

Report facts: where things are and how they work, citing `path:line`, quoting \
short code where it matters. Keep what you read separate from what you infer, \
and list what you checked but could not settle. Your report goes to a lead who \
has not seen the files, so be complete but concise.

You have read-only access and cannot change files.""",
    ),
    "reviewer": Role(
        model=DEFAULT_MODEL,
        effort="high",
        max_model_calls=30,
        prompt="""\
You are the reviewer on an engineering team. You are given a draft answer, a \
list of claims, or a description of a change already applied to the project in \
your filesystem (`/` is the project root).

Check each point against the code yourself; don't trust what you were told. \
Start with an overall verdict: accurate, minor fixes needed, or significant \
errors. Then, per point: confirmed / wrong / unverifiable, with `path:line` \
evidence, and the correction when it is wrong. For a change, also look for \
bugs it introduces and places it should have touched but didn't. Finish with \
anything important that was missed.

You have read-only access and cannot change files.""",
    ),
}

ENGINEER = Role(
    model=DEFAULT_MODEL,
    effort="high",
    max_model_calls=60,
    prompt="""\
You are the engineer on an engineering team. Make the change described in \
your brief to the project in your filesystem (`/` is the project root).

- Read the relevant code first and follow the project's existing patterns; its \
AGENTS.md / CLAUDE.md, when loaded, take precedence.
- Make the smallest change that fully does the job. Don't refactor or reformat \
unrelated code.
- Your edits are staged: the user reviews them as one batch before anything \
reaches the project. `read_file` shows your staged versions; `ls`, `glob` and \
`grep` show the project as it was before your changes.
- Don't write or change tests unless your brief asks for them.
- You cannot run commands, builds or tests; after the user approves your \
batch, the project's checks (such as a compile) run automatically.

Report, file by file, what you changed and why; anything you were unsure \
about; and how to verify the change (what to build or test).""",
)

SPECIALISTS = [*READ_ONLY_SPECIALISTS, "engineer"]  # the roles that run a model
Step = Literal["scout", "reviewer", "engineer", "tester", "finish"]


class CommandArg(BaseModel):
    name: str = Field(description="A param of the chosen command.")
    value: str


class Decision(BaseModel):
    next: Step
    reason: str
    brief: str = Field(description="The specialist's task; empty for tester and finish.")
    command: str = Field(description="The command to run when next is tester; otherwise empty.")
    args: list[CommandArg] = Field(description="Param values for the command; otherwise empty.")
    answer: str = Field(description="The final answer; empty unless next is finish.")


class FinalDecision(BaseModel):
    """The only schema offered once no rounds are left."""

    next: Literal["finish"]
    reason: str
    brief: str = Field(description="Always empty.")
    answer: str = Field(description="The final answer.")


class Report(TypedDict):
    role: str  # a specialist, or "user" for an approval outcome
    brief: str
    report: str


class ChangeDict(TypedDict):
    path: str
    before: str | None
    after: str | None


class TeamInput(TypedDict):
    request: str
    project: Annotated[str, AfterValidator(existing_dir)]
    max_rounds: NotRequired[Annotated[int, Field(ge=0, le=10)]]
    max_cost_usd: NotRequired[Annotated[float, Field(gt=0, le=20)]]


class TeamState(BaseState, total=False):
    request: str
    project: str
    max_rounds: int
    max_cost_usd: float
    reports: Annotated[list[Report], add]
    brief: str
    pending: list[ChangeDict]  # the engineer's staged batch, awaiting approval
    approval: dict  # {"decision": "approve" | "reject", "note": str}
    command: dict  # the orchestrator's pick for the tester: {"name": str, "args": {...}}
    prepared: dict  # that command checked and filled in (commands.Prepared), awaiting its run
    answer: str


def role_spec(role: str) -> Role:
    if role == "orchestrator":
        return ORCHESTRATOR
    if role == "engineer":
        return ENGINEER
    return READ_ONLY_SPECIALISTS[role]


def default_model(role: str) -> BaseChatModel:
    spec = role_spec(role)
    return make_model(spec.model, effort=spec.effort)


def rounds_used(state: TeamState) -> int:
    """Dispatches the orchestrator chose; approval outcomes and verify runs don't count."""
    return sum(1 for r in state.get("reports", []) if r["role"] not in ("user", "verify"))


@dataclass(frozen=True)
class Limits:
    """What the run may still do: rounds and money."""

    rounds_left: int
    spent_usd: float
    budget_usd: float

    @classmethod
    def of(cls, state: TeamState) -> "Limits":
        rounds_left = max(state.get("max_rounds", DEFAULT_MAX_ROUNDS) - rounds_used(state), 0)
        budget = state.get("max_cost_usd", DEFAULT_BUDGET_USD)
        return cls(rounds_left, state.get("cost_usd", 0.0), budget)

    @property
    def money_left(self) -> float:
        return max(self.budget_usd - self.spent_usd, 0.0)

    @property
    def can_dispatch(self) -> bool:
        return self.rounds_left > 0 and self.money_left > 0


def orchestrator_message(state: TeamState, limits: Limits, commands_text: str) -> str:
    parts = [
        f"<request>\n{state['request']}\n</request>",
        f"<commands>\n{commands_text}\n</commands>",
        f"<rounds_left>{limits.rounds_left}</rounds_left>",
        f"<budget>spent ${limits.spent_usd:.2f} of ${limits.budget_usd:.2f}; "
        f"${limits.money_left:.2f} left</budget>",
    ]
    reports = state.get("reports", [])
    if reports:
        parts.append("<reports>")
        for i, r in enumerate(reports, 1):
            parts.append(
                f'<report n="{i}" role="{r["role"]}">\n'
                f"<brief>\n{r['brief']}\n</brief>\n<result>\n{r['report']}\n</result>\n</report>"
            )
        parts.append("</reports>")
    else:
        parts.append("<reports>none yet</reports>")
    if limits.rounds_left == 0:
        parts.append("No rounds are left: finish now with the best answer the reports support.")
    elif limits.money_left <= 0:
        parts.append(
            "The budget is spent: finish now with the best answer the reports support, "
            "and say what is still unchecked."
        )
    return "\n\n".join(parts)


def unified_diff(batch: list[ChangeDict]) -> str:
    """One reviewable diff for the whole batch.

    The diff compares lines without their endings, so a line whose ending
    changed but text didn't would be invisible; such files get a warning line.
    """
    out = []
    for change in batch:
        hidden = _hidden_ending_changes(change["before"], change["after"])
        if hidden:
            out.append(f"WARNING {change['path']}: line endings change on {hidden} unchanged lines")
        before = (change["before"] or "").splitlines()
        after = (change["after"] or "").splitlines()
        old = f"a{change['path']}" if change["before"] is not None else "/dev/null"
        new = f"b{change['path']}" if change["after"] is not None else "/dev/null"
        out.extend(difflib.unified_diff(before, after, old, new, lineterm=""))
    return "\n".join(out) + "\n"


def _hidden_ending_changes(before: str | None, after: str | None) -> int:
    if before is None or after is None:
        return 0
    old = before.splitlines(keepends=True)
    new = after.splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(
        a=[line.rstrip("\r\n") for line in old], b=[line.rstrip("\r\n") for line in new]
    )
    return sum(
        old[i1 + k] != new[j1 + k]
        for tag, i1, i2, j1, _ in matcher.get_opcodes()
        if tag == "equal"
        for k in range(i2 - i1)
    )


def build(
    model_for: Callable[[str], BaseChatModel] = default_model,
    projects_dir: Path = commands.PROJECTS_DIR,
) -> StateGraph:
    def orchestrator(state: TeamState) -> dict:
        limits = Limits.of(state)
        schema = Decision if limits.can_dispatch else FinalDecision
        model = model_for("orchestrator").with_structured_output(schema, method="json_schema")
        config = commands.find_config(state["project"], projects_dir)
        messages = [
            ("system", ORCHESTRATOR.prompt),
            ("user", orchestrator_message(state, limits, commands.describe(config))),
        ]
        # An incomplete decision (a step without its brief, say) goes back to
        # the model with the problem named, a few times, before the round fails.
        with get_usage_metadata_callback() as usage:
            for _ in range(1 + MAX_DECISION_FIXES):
                decision = model.invoke(messages)
                if not isinstance(decision, Decision | FinalDecision):
                    raise TypeError(
                        f"orchestrator returned {type(decision).__name__}, not a decision"
                    )
                problem = decision_problem(decision)
                if problem is None:
                    break
                log.info("incomplete decision (%s); asking again", problem)
                messages = [
                    *messages,
                    ("assistant", decision.model_dump_json()),
                    ("user", f"That decision can't be carried out: {problem}. Decide again."),
                ]
            else:
                raise ValueError(f"orchestrator {problem}, {1 + MAX_DECISION_FIXES} times")
        log.info(
            "round=%d next=%s spent=$%.2f reason=%s",
            rounds_used(state) + 1,
            decision.next,
            limits.spent_usd,
            decision.reason,
        )

        update: dict = {"verdict": decision.next, "cost_usd": cost_usd(usage.usage_metadata)}
        if decision.next == "finish":
            update["answer"] = decision.answer
        elif isinstance(decision, Decision) and decision.next == "tester":
            args = {a.name: a.value for a in decision.args}
            update["command"] = {"name": decision.command, "args": args}
            update["brief"] = f"Run {decision.command} with {args or 'no args'}: {decision.reason}"
        else:
            update["brief"] = decision.brief
        return update

    def read_only(role: str):
        spec = READ_ONLY_SPECIALISTS[role]

        def node(state: TeamState) -> dict:
            limit = StepLimit(spec.max_model_calls)
            agent = read_only_agent(state["project"], model_for(role), spec.prompt, role, limit)
            text, cost = run_agent(agent, state["brief"], limit)
            report: Report = {"role": role, "brief": state["brief"], "report": text}
            return {"reports": [report], "cost_usd": cost}

        return node

    def engineer(state: TeamState) -> dict:
        with tempfile.TemporaryDirectory(prefix="nexus-staging-") as staging:
            limit = StepLimit(ENGINEER.max_model_calls)
            agent, backend = staged_engineer(
                state["project"], staging, model_for("engineer"), ENGINEER.prompt, limit
            )
            text, cost = run_agent(agent, state["brief"], limit)
            batch = [asdict(change) for change in backend.changes()]
        files = ", ".join(c["path"] for c in batch) or "none"
        report: Report = {
            "role": "engineer",
            "brief": state["brief"],
            "report": f"{text}\n\nStaged files: {files}",
        }
        return {"reports": [report], "pending": batch, "cost_usd": cost}

    def approve(state: TeamState) -> dict:
        # Nothing but the question happens here: on resume LangGraph re-runs
        # this node from the top, and interrupt() returns the user's decision.
        batch = state["pending"]
        question = {
            "question": "Apply these changes to the project?",
            "summary": state["reports"][-1]["report"],
            "files": [c["path"] for c in batch],
            "diff": unified_diff(batch),
        }
        return {"approval": ask_decision(question)}

    def apply(state: TeamState) -> dict:
        batch = state["pending"]
        approval = state["approval"]
        if approval["decision"] == "approve":
            applied, conflicts = apply_changes(state["project"], [Change(**c) for c in batch])
            text = f"The user approved the batch. Applied: {', '.join(applied) or 'none'}."
            if conflicts:
                text += (
                    f" Not applied, because these files changed on disk after staging: "
                    f"{', '.join(conflicts)}."
                )
            log.info("applied=%d conflicts=%d", len(applied), len(conflicts))
        else:
            text = "The user rejected the batch; nothing was written to the project."
            log.info("rejected batch of %d files", len(batch))
        if approval.get("note"):
            text += f" Their note: {approval['note']}"
        report: Report = {
            "role": "user",
            "brief": "Approve the engineer's changes?",
            "report": text,
        }
        return {"reports": [report], "pending": []}

    def tester(state: TeamState, config: RunnableConfig) -> dict:
        """Check the orchestrator's command and fill it in; nothing runs yet."""
        name, args = state["command"]["name"], state["command"]["args"]
        thread_id = config.get("configurable", {}).get("thread_id", "no-thread")
        stem = commands.log_stem(thread_id, len(state.get("reports", [])) + 1, name)
        try:
            project_config = commands.find_config(state["project"], projects_dir)
            prepared = commands.prepare(project_config, state["project"], name, args, stem)
        except commands.CommandError as exc:
            log.info("command refused: %s", exc)
            report: Report = {
                "role": "tester",
                "brief": state["brief"],
                "report": f"Not run: {exc}.",
            }
            return {"reports": [report], "prepared": {}}
        return {"prepared": asdict(prepared)}

    def approve_command(state: TeamState) -> dict:
        prepared = commands.Prepared(**state["prepared"])
        question = {
            "question": "Run this command?",
            "summary": state["brief"],
            "command": (
                f"{prepared.display()}\n\n"
                f"in {prepared.cwd}, time limit {prepared.timeout} s, log {prepared.log_file}"
            ),
        }
        approval = ask_decision(question)
        if approval["decision"] == "approve":
            return {"approval": approval}
        text = f"The user rejected running {prepared.name}; it did not run."
        if approval["note"]:
            text += f" Their note: {approval['note']}"
        report: Report = {"role": "user", "brief": "Run this command?", "report": text}
        return {"approval": approval, "reports": [report], "prepared": {}}

    def run_command(state: TeamState) -> dict:
        text, _ = run_and_summarize(commands.Prepared(**state["prepared"]), state["project"])
        report: Report = {"role": "tester", "brief": state["brief"], "report": text}
        return {"reports": [report], "prepared": {}}

    def verify(state: TeamState, config: RunnableConfig) -> dict:
        """Run the project's verify commands after an approved batch.

        In order, without asking (listing them in the project's config is the
        user's approval), stopping at the first that fails or can't run.
        """
        project_config = commands.find_config(state["project"], projects_dir)
        thread_id = config.get("configurable", {}).get("thread_id", "no-thread")
        n = len(state.get("reports", []))
        reports: list[Report] = []
        for name in project_config.verify if project_config else []:
            n += 1
            try:
                stem = commands.log_stem(thread_id, n, name)
                prepared = commands.prepare(project_config, state["project"], name, {}, stem)
            except commands.CommandError as exc:
                text, ok = f"Not run: {exc}.", False
            else:
                text, ok = run_and_summarize(prepared, state["project"])
            brief = f"Run automatically after the approved change: {name}"
            reports.append({"role": "verify", "brief": brief, "report": text})
            if not ok:
                break
        return {"reports": reports}

    graph = StateGraph(TeamState, input_schema=TeamInput)
    # No graph-level retries on model-driven nodes, so a failure never silently
    # repeats paid work. The SDK retries each API call.
    graph.add_node("orchestrator", orchestrator)
    for role in READ_ONLY_SPECIALISTS:
        graph.add_node(role, read_only(role))
        graph.add_edge(role, "orchestrator")
    graph.add_node("engineer", engineer)
    graph.add_node("approve", approve)
    graph.add_node("apply", apply)
    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges(
        "orchestrator",
        lambda state: state["verdict"],
        {**{role: role for role in SPECIALISTS}, "tester": "tester", "finish": END},
    )
    graph.add_conditional_edges(
        "engineer",
        lambda state: "approve" if state.get("pending") else "orchestrator",
        ["approve", "orchestrator"],
    )
    graph.add_edge("approve", "apply")
    graph.add_conditional_edges(
        "apply",
        lambda state: "verify" if state["approval"]["decision"] == "approve" else "orchestrator",
        ["verify", "orchestrator"],
    )
    # Not retried, like run_command; a resume after a crash runs it again.
    graph.add_node("verify", verify)
    graph.add_edge("verify", "orchestrator")
    graph.add_node("tester", tester)
    graph.add_node("approve_command", approve_command)
    # Never retried: a command's failure is its result, not an error to repeat.
    # If nexus itself dies mid-run, nexus.resume runs the approved command again.
    graph.add_node("run_command", run_command)
    graph.add_conditional_edges(
        "tester", after_tester, ["approve_command", "run_command", "orchestrator"]
    )
    graph.add_conditional_edges(
        "approve_command",
        lambda state: "run_command" if state.get("prepared") else "orchestrator",
        ["run_command", "orchestrator"],
    )
    graph.add_edge("run_command", "orchestrator")
    return graph


MAX_DECISION_FIXES = 2  # times an incomplete decision is sent back before the round fails


def decision_problem(decision: Decision | FinalDecision) -> str | None:
    """What makes a decision impossible to carry out, or None if it is complete."""
    if decision.next == "finish":
        return None if decision.answer.strip() else "finished without an answer"
    if isinstance(decision, Decision) and decision.next == "tester":
        return None if decision.command.strip() else "chose tester without a command"
    return None if decision.brief.strip() else f"chose {decision.next} without a brief"


def ask_decision(question: dict) -> dict:
    """Pause until the user answers approve or reject; ask again on anything else.

    A malformed answer must not fail the node: LangGraph would replay that same
    answer on every resume, and the run could never continue. Each repeated
    question is a new interrupt, so the next resume answers that one.
    """
    asked = question
    while True:
        answer = interrupt(asked)
        if isinstance(answer, dict) and answer.get("decision") in ("approve", "reject"):
            return {"decision": answer["decision"], "note": str(answer.get("note") or "")}
        asked = {
            **question,
            "question": f"{question['question']} (answer approve or reject; got {answer!r})",
        }


def run_and_summarize(prepared: commands.Prepared, project: str) -> tuple[str, bool]:
    """Run a checked command; return its report and whether it passed."""
    log.info("running %s (time limit %d s)", prepared.name, prepared.timeout)
    before = commands.git_snapshot(project)
    try:
        outcome = commands.run_prepared(prepared)
    except OSError as exc:
        return f"Command {prepared.name} could not start: {exc}", False
    changed = commands.changed_between(before, commands.git_snapshot(project))
    log.info("%s exit=%s in %.0f s", prepared.name, outcome.exit_code, outcome.seconds)
    return commands.summarize(prepared, outcome, changed), commands.passed(prepared, outcome)


def after_tester(state: TeamState) -> str:
    prepared = state.get("prepared")
    if not prepared:
        return "orchestrator"  # refused; the report says why
    return "run_command" if prepared["approval"] == "auto" else "approve_command"


graph = build().compile(name="team")
