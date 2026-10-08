"""Continue a saved nexus run.

    python -m nexus.resume --thread team-20261007T163000 --decision approve
    python -m nexus.resume --thread team-20261007T163000 --decision reject --note "keep it"
    python -m nexus.resume --thread team-20261007T163000          # retry a failed step
    python -m nexus.resume --thread team-20261007T163000 --message "go with steps 1-3"

A run that paused for approval needs ``--decision``; a run that failed partway
is continued from the step that failed, keeping everything finished before it.
A completed team run takes a follow-up ``--message``: the team continues on the
same thread, with the earlier requests, answers and reports in view.
The pipeline is the one the run was started with. Output and exit codes are
the same as ``python -m nexus``.
"""

import argparse
import logging
import sys

from nexus.__main__ import show
from nexus.errors import InputError
from nexus.run import load_env, log_handler, open_runs_db, resume_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m nexus.resume",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--thread", required=True, help="the thread id the run printed")
    reply = parser.add_mutually_exclusive_group()
    reply.add_argument("--decision", choices=["approve", "reject"])
    reply.add_argument("--message", help="a follow-up to a completed run, in plain words")
    parser.add_argument("--note", default="", help="with --decision: guidance for the next step")
    parser.add_argument("--json", action="store_true", help="print the full run result as JSON")
    args = parser.parse_args(argv)
    if args.note and not args.decision:
        parser.error("--note goes with --decision")
    logging.basicConfig(level=logging.INFO, handlers=[log_handler()])
    load_env()

    decision = {"decision": args.decision, "note": args.note} if args.decision else None
    try:
        result = resume_pipeline(args.thread, decision, open_runs_db(), message=args.message)
    except InputError as exc:
        parser.error(str(exc))
    return show(result, as_json=args.json)


if __name__ == "__main__":
    sys.exit(main())
