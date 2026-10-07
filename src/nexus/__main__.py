"""Send nexus a request, as plain text.

    python -m nexus "how does jumping work?" --project C:\\projects\\impact

By default the ``team`` pipeline answers: an orchestrator sends read-only
specialists (scout, reviewer) to explore the project, then writes the answer.
``--pipeline ask`` uses a single read-only agent instead.

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
from nexus.run import load_env, log_handler, run_pipeline


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
    parser.add_argument("--thread", help="thread id (default: <pipeline>-<timestamp>)")
    parser.add_argument("--json", action="store_true", help="print the full run result as JSON")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, handlers=[log_handler()])
    load_env()

    thread_id = args.thread or f"{args.pipeline}-{datetime.now():%Y%m%dT%H%M%S}"
    inputs = {"request": args.request, "project": args.project}
    if args.max_rounds is not None:
        if args.pipeline != "team":
            parser.error("--max-rounds applies to the team pipeline only")
        inputs["max_rounds"] = args.max_rounds
    graph = team.graph if args.pipeline == "team" else ask.graph
    try:
        result = run_pipeline(graph, thread_id, inputs)
    except InputError as exc:
        parser.error(str(exc))

    if args.json:
        print(json.dumps(asdict(result), indent=2, default=str))
    elif result.status == "completed":
        print(result.state.get("answer", ""))
    cost = result.state.get("cost_usd", 0.0)
    print(f"\n[{result.status}] thread={thread_id} cost=${cost:.4f}", file=sys.stderr)
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
