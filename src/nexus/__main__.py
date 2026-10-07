"""Send nexus a request, as plain text.

    python -m nexus "how does jumping work?" --project C:\\projects\\impact

By default the ``team`` pipeline answers: an orchestrator sends read-only
specialists (scout, reviewer) to explore the project, then writes the answer.
``--pipeline ask`` uses a single read-only agent instead.

If the team's engineer proposes changes, the run pauses (exit code 4) and
prints them as one diff; nothing is written until you approve with
``python -m nexus.resume --thread <id> --decision approve`` (or reject). The
same pause comes before the tester runs a build or test command, unless the
project's config (``projects/<name>.toml``) sets that command to run without asking.

The answer is printed to stdout as Markdown; progress (including each routing
decision), cost and the thread id go to stderr. ``--json`` prints the full run
result instead. Exit codes are the same as ``nexus.run``.
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime

from nexus.errors import InputError
from nexus.pipelines import ask, team
from nexus.run import RunResult, load_env, log_handler, open_runs_db, run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m nexus",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("request", help="what you want, in plain words")
    parser.add_argument("--project", required=True, help="path to the project's repo")
    parser.add_argument("--pipeline", choices=["team", "ask"], default="team")
    parser.add_argument(
        "--max-rounds",
        type=int,
        help=f"team only: specialist rounds allowed (default {team.DEFAULT_MAX_ROUNDS})",
    )
    parser.add_argument(
        "--budget",
        type=float,
        metavar="USD",
        help=f"team only: US dollars the run may spend (default {team.DEFAULT_BUDGET_USD:g})",
    )
    parser.add_argument("--thread", help="thread id (default: <pipeline>-<timestamp>)")
    parser.add_argument("--json", action="store_true", help="print the full run result as JSON")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, handlers=[log_handler()])
    load_env()

    thread_id = args.thread or f"{args.pipeline}-{datetime.now():%Y%m%dT%H%M%S}"
    inputs = {"request": args.request, "project": args.project}
    for flag, key, value in [
        ("--max-rounds", "max_rounds", args.max_rounds),
        ("--budget", "max_cost_usd", args.budget),
    ]:
        if value is not None:
            if args.pipeline != "team":
                parser.error(f"{flag} applies to the team pipeline only")
            inputs[key] = value
    graph = team.graph if args.pipeline == "team" else ask.graph
    try:
        result = run_pipeline(graph, thread_id, inputs, open_runs_db())
    except InputError as exc:
        parser.error(str(exc))
    return show(result, as_json=args.json)


def show(result: RunResult, as_json: bool = False) -> int:
    """Print a run's outcome for a person (or as JSON) and return its exit code."""
    if as_json:
        print(json.dumps(asdict(result), indent=2, default=str))
    elif result.status == "completed":
        print(result.state.get("answer", ""))
    elif result.status == "waiting":
        asked = result.waiting_for or {}
        print(asked.get("summary", ""))
        print(f"\n{asked.get('question', 'Waiting for a decision.')}\n")
        print(asked.get("diff") or asked.get("command", ""))
    cost = result.state.get("cost_usd", 0.0)
    line = f"\n[{result.status}] thread={result.thread_id} cost=${cost:.4f}"
    if result.status == "failed":
        # The failed step's spending is not in the run's cost; show it.
        line += f" (this attempt spent ${result.spent_usd:.4f}, the failed step included)"
    print(line, file=sys.stderr)
    resume = f"python -m nexus.resume --thread {result.thread_id}"
    if result.status == "waiting":
        print(
            f"Approve: {resume} --decision approve\n"
            f'Reject:  {resume} --decision reject --note "what to change"',
            file=sys.stderr,
        )
    elif result.status == "failed":
        print(f"Retry the failed step: {resume}", file=sys.stderr)
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
