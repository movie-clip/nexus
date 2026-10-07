"""Answer an engineering request with an orchestrator and specialist roles.

    START -> orchestrator -+- scout ------------------------------+
               ^           +- reviewer ---------------------------+
               |           +- engineer -+- (no changes) ----------+
               |           |            +- approve -> apply ------+
               +-----------+--------------- reports back <--------+
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
report from "user". Specialists see only their brief; the orchestrator sees
every brief and report.
"""

import difflib
import logging
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from operator import add
from typing import Annotated, Literal, NotRequired, TypedDict

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import AfterValidator, BaseModel, Field

from nexus.agents import existing_dir, read_only_agent, run_agent, staged_engineer
from nexus.models import DEFAULT_MODEL, cost_usd, make_model
from nexus.staging import Change, apply_changes
from nexus.state import BaseState

log = logging.getLogger("nexus.team")

DEFAULT_MAX_ROUNDS = 4


@dataclass(frozen=True)
class Role:
    model: str
    effort: str
    prompt: str


ORCHESTRATOR = Role(
    model=DEFAULT_MODEL,
    effort="high",
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
- finish: you write the final answer to the user.

How to decide:
- Scale effort to the request. A narrow factual question: one scout round, then \
finish. A broad or high-stakes question: scout (again if a report leaves a \
gap), then reviewer on your draft, then finish. A change: scout first unless \
the request already names exactly what to change, then engineer.
- After an approved change, send the reviewer to check it when it is non-trivial. \
After a rejection, send the engineer again with a brief that addresses the \
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
`path:line` from the reports. If changes were applied, list the files changed \
and what the user should build or test to verify them. Don't mention the team, \
rounds or specialists.

Fields: `next`; `reason` (one sentence: why this step); `brief` (the task when \
next is a specialist, otherwise ""); `answer` (the final answer when next is \
finish, otherwise "").""",
)

READ_ONLY_SPECIALISTS: dict[str, Role] = {
    # Every role is on DEFAULT_MODEL (Sonnet 5.5) during development. When
    # tuning, the scout can stay on Sonnet: most of its work is reading.
    "scout": Role(
        model=DEFAULT_MODEL,
        effort="medium",
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
- You cannot run commands, builds or tests.

Report, file by file, what you changed and why; anything you were unsure \
about; and how to verify the change (what to build or test).""",
)

SPECIALISTS = [*READ_ONLY_SPECIALISTS, "engineer"]
Step = Literal["scout", "reviewer", "engineer", "finish"]


class Decision(BaseModel):
    next: Step
    reason: str
    brief: str = Field(description="The specialist's task; empty when next is finish.")
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


class TeamState(BaseState, total=False):
    request: str
    project: str
    max_rounds: int
    reports: Annotated[list[Report], add]
    brief: str
    pending: list[ChangeDict]  # the engineer's staged batch, awaiting approval
    approval: dict  # {"decision": "approve" | "reject", "note": str}
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
    """Specialist dispatches so far; the user's approval outcomes don't count."""
    return sum(1 for r in state.get("reports", []) if r["role"] != "user")


def orchestrator_message(state: TeamState, rounds_left: int) -> str:
    parts = [
        f"<request>\n{state['request']}\n</request>",
        f"<rounds_left>{rounds_left}</rounds_left>",
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
    if rounds_left == 0:
        parts.append("No rounds are left: finish now with the best answer the reports support.")
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


def build(model_for: Callable[[str], BaseChatModel] = default_model) -> StateGraph:
    def orchestrator(state: TeamState) -> dict:
        used = rounds_used(state)
        rounds_left = max(state.get("max_rounds", DEFAULT_MAX_ROUNDS) - used, 0)
        schema = Decision if rounds_left else FinalDecision
        model = model_for("orchestrator").with_structured_output(schema, method="json_schema")
        messages = [
            ("system", ORCHESTRATOR.prompt),
            ("user", orchestrator_message(state, rounds_left)),
        ]
        with get_usage_metadata_callback() as usage:
            decision = model.invoke(messages)
        log.info("round=%d next=%s reason=%s", used + 1, decision.next, decision.reason)

        update: dict = {"verdict": decision.next, "cost_usd": cost_usd(usage.usage_metadata)}
        if decision.next == "finish":
            if not decision.answer.strip():
                raise ValueError("orchestrator finished without an answer")
            update["answer"] = decision.answer
        else:
            if not decision.brief.strip():
                raise ValueError(f"orchestrator chose {decision.next} without a brief")
            update["brief"] = decision.brief
        return update

    def read_only(role: str):
        spec = READ_ONLY_SPECIALISTS[role]

        def node(state: TeamState) -> dict:
            agent = read_only_agent(state["project"], model_for(role), spec.prompt, name=role)
            text, cost = run_agent(agent, state["brief"])
            report: Report = {"role": role, "brief": state["brief"], "report": text}
            return {"reports": [report], "cost_usd": cost}

        return node

    def engineer(state: TeamState) -> dict:
        with tempfile.TemporaryDirectory(prefix="nexus-staging-") as staging:
            agent, backend = staged_engineer(
                state["project"], staging, model_for("engineer"), ENGINEER.prompt
            )
            text, cost = run_agent(agent, state["brief"])
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
        decision = interrupt(
            {
                "question": "Apply these changes to the project?",
                "summary": state["reports"][-1]["report"],
                "files": [c["path"] for c in batch],
                "diff": unified_diff(batch),
            }
        )
        if not isinstance(decision, dict) or decision.get("decision") not in ("approve", "reject"):
            raise ValueError(f"expected decision approve or reject, got {decision!r}")
        return {"approval": {"decision": decision["decision"], "note": decision.get("note", "")}}

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
        {**{role: role for role in SPECIALISTS}, "finish": END},
    )
    graph.add_conditional_edges(
        "engineer",
        lambda state: "approve" if state.get("pending") else "orchestrator",
        ["approve", "orchestrator"],
    )
    graph.add_edge("approve", "apply")
    graph.add_edge("apply", "orchestrator")
    return graph


graph = build().compile(name="team")
