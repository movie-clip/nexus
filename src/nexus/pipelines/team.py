"""Answer an engineering request with an orchestrator and specialist roles.

    START -> orchestrator -+- scout ----+
               ^           +- reviewer -+
               +-----------+            |
               |  (report back)  <------+
               +- finish -> END

Each round the orchestrator (one structured Claude call, no file access) reads
the request and every report so far and picks the next step: send a
specialist a self-contained brief, or finish with the final answer. The choice
comes back as JSON enforced by the API's structured output, and LangGraph routes
on it in plain code. When no rounds are left, the only schema offered is
"finish", so the round cap holds no matter what the model would prefer.

Specialists are read-only Deep Agents (see ``nexus.agents.read_only_agent``),
each with its own prompt and model. They see only their brief, and the
orchestrator sees only their reports.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from operator import add
from typing import Annotated, Literal, NotRequired, TypedDict

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from pydantic import AfterValidator, BaseModel, Field

from nexus.agents import existing_dir, read_only_agent, run_agent
from nexus.models import cost_usd, make_model
from nexus.state import BaseState

log = logging.getLogger("nexus.team")

DEFAULT_MAX_ROUNDS = 4


@dataclass(frozen=True)
class Role:
    model: str
    effort: str
    prompt: str


ORCHESTRATOR = Role(
    model="claude-opus-5-5",
    effort="high",
    prompt="""\
You lead a small team answering an engineering request about a software \
project. You cannot read the project yourself; your specialists can, read-only. \
Each round you choose one step.

Steps:
- scout: explores the repo and reports what is actually there (where things \
live, how they work today) with `path:line` citations. Use it to gather facts.
- reviewer: checks a draft answer or a list of claims against the code and \
returns a verdict per claim (confirmed / wrong / unverifiable) with evidence. \
Use it when the answer makes non-trivial claims, especially ones a scout \
inferred rather than quoted.
- finish: you write the final answer to the user.

How to decide:
- Scale effort to the request. A narrow factual question: one scout round, then \
finish. A broad or high-stakes question: scout (again if a report leaves a \
gap), then reviewer on your draft, then finish.
- Don't repeat a step that already answered the question. If a report leaves a \
specific gap, send a narrower brief about that gap.
- Finish when every claim you will make is backed by a report. If something is \
still unverified when you finish, say so in the answer.

Writing a brief: the specialist sees only your brief, not the request or other \
reports. Make it self-contained: the objective, what is already known (with \
paths), where to start, what to return, and what is out of scope. For the \
reviewer, include the full draft or the claims to check.

The final answer: Markdown, the direct answer first, citing files as \
`path:line` from the reports. Don't mention the team, rounds or specialists.

Fields: `next`; `reason` (one sentence: why this step); `brief` (the task when \
next is a specialist, otherwise ""); `answer` (the final answer when next is \
finish, otherwise "").""",
)

SPECIALISTS: dict[str, Role] = {
    # Sonnet 5.5 for the scout: most of its work is reading and searching.
    "scout": Role(
        model="claude-sonnet-5-5",
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
        model="claude-opus-5-5",
        effort="high",
        prompt="""\
You are the reviewer on an engineering team. You are given a draft answer or a \
list of claims about the project in your filesystem (`/` is the project root).

Check each claim against the code yourself; don't trust the draft. Start with \
an overall verdict: accurate, minor fixes needed, or significant errors. Then, \
per claim: confirmed / wrong / unverifiable, with `path:line` evidence, and the \
correction when it is wrong. Finish with anything important the draft missed.

You have read-only access and cannot change files.""",
    ),
}

Step = Literal["scout", "reviewer", "finish"]


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
    role: str
    brief: str
    report: str


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
    answer: str


def default_model(role: str) -> BaseChatModel:
    spec = ORCHESTRATOR if role == "orchestrator" else SPECIALISTS[role]
    return make_model(spec.model, effort=spec.effort)


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
                f'<report round="{i}" role="{r["role"]}">\n'
                f"<brief>\n{r['brief']}\n</brief>\n<result>\n{r['report']}\n</result>\n</report>"
            )
        parts.append("</reports>")
    else:
        parts.append("<reports>none yet</reports>")
    if rounds_left == 0:
        parts.append("No rounds are left: finish now with the best answer the reports support.")
    return "\n\n".join(parts)


def build(model_for: Callable[[str], BaseChatModel] = default_model) -> StateGraph:
    def orchestrator(state: TeamState) -> dict:
        reports = state.get("reports", [])
        rounds_left = max(state.get("max_rounds", DEFAULT_MAX_ROUNDS) - len(reports), 0)
        schema = Decision if rounds_left else FinalDecision
        model = model_for("orchestrator").with_structured_output(schema, method="json_schema")
        messages = [
            ("system", ORCHESTRATOR.prompt),
            ("user", orchestrator_message(state, rounds_left)),
        ]
        with get_usage_metadata_callback() as usage:
            decision = model.invoke(messages)
        log.info("round=%d next=%s reason=%s", len(reports) + 1, decision.next, decision.reason)

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

    def specialist(role: str):
        spec = SPECIALISTS[role]

        def node(state: TeamState) -> dict:
            agent = read_only_agent(state["project"], model_for(role), spec.prompt, name=role)
            text, cost = run_agent(agent, state["brief"])
            report: Report = {"role": role, "brief": state["brief"], "report": text}
            return {"reports": [report], "cost_usd": cost}

        return node

    graph = StateGraph(TeamState, input_schema=TeamInput)
    # As in `ask`: no graph-level retries on model-driven nodes, so a failure
    # never silently repeats paid work. The SDK retries each API call.
    graph.add_node("orchestrator", orchestrator)
    for role in SPECIALISTS:
        graph.add_node(role, specialist(role))
        graph.add_edge(role, "orchestrator")
    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges(
        "orchestrator",
        lambda state: state["verdict"],
        {**{role: role for role in SPECIALISTS}, "finish": END},
    )
    return graph


graph = build().compile(name="team")
