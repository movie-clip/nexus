"""Run a registered pipeline once.

    python -m nexus.run --pipeline smoke --thread t1 --set scenario=flaky

``--set key=value`` may repeat. Keys must be in the pipeline's input schema,
and values are converted to the schema's types (``max_retries=3`` becomes an
int). Progress is logged to stderr, one line per finished node; the final
state is printed to stdout as JSON. Exit code is 1 if the run recorded errors.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from typing import Any, get_type_hints

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from pydantic import TypeAdapter, ValidationError

import nexus.pipelines  # noqa: F401  (registers built-in pipelines)
from nexus import registry
from nexus.state import BaseState

log = logging.getLogger("nexus.run")

RESERVED = frozenset(get_type_hints(BaseState))


class InputError(ValueError):
    """Bad run inputs: the caller's mistake, not a pipeline failure."""


def validate_inputs(graph: StateGraph, inputs: dict[str, Any]) -> dict[str, Any]:
    """Check run inputs against the pipeline's declared input schema.

    Rejects unknown keys (a typo must not silently fall back to defaults),
    engine-owned fields, and values of the wrong type. Raises ``InputError``.
    """
    schema = graph.input_schema
    if schema is graph.state_schema:
        raise InputError("pipeline must declare input_schema= on its StateGraph")
    allowed = set(get_type_hints(schema)) - RESERVED
    unknown = sorted(set(inputs) - allowed)
    if unknown:
        raise InputError(f"unknown input(s) {unknown}; allowed: {sorted(allowed)}")
    try:
        return TypeAdapter(schema).validate_python(inputs)
    except ValidationError as exc:
        raise InputError(str(exc)) from None


def run_pipeline(
    name: str,
    thread_id: str,
    inputs: dict[str, Any],
    checkpointer: BaseCheckpointSaver | None = None,
) -> dict[str, Any]:
    builder = registry.get(name)()
    inputs = validate_inputs(builder, inputs)
    graph = builder.compile(checkpointer=checkpointer or InMemorySaver())
    config = {"configurable": {"thread_id": thread_id}}

    log.info("start pipeline=%s thread=%s", name, thread_id)
    for update in graph.stream(inputs, config=config, stream_mode="updates"):
        for node, delta in update.items():
            # Key names only: values may hold data that shouldn't reach logs.
            log.info("node=%s wrote=%s", node, sorted(delta or {}))
    return graph.get_state(config).values


def parse_sets(pairs: list[str]) -> dict[str, str]:
    inputs: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise InputError(f"expected key=value, got {pair!r}")
        inputs[key] = value
    return inputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m nexus.run", description=__doc__)
    parser.add_argument("--pipeline", required=True, choices=registry.names())
    parser.add_argument("--thread", help="thread id (default: <pipeline>-<timestamp>)")
    parser.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(message)s")

    thread_id = args.thread or f"{args.pipeline}-{datetime.now():%Y%m%dT%H%M%S}"
    try:
        final = run_pipeline(args.pipeline, thread_id, parse_sets(args.sets))
    except InputError as exc:
        parser.error(str(exc))

    print(json.dumps({"thread": thread_id, "state": final}, indent=2, default=str))
    return 1 if final.get("errors") else 0


if __name__ == "__main__":
    sys.exit(main())
