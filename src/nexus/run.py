"""Run a pipeline from langgraph.json once, started by hand.

    python -m nexus.run --pipeline smoke --set scenario=flaky
    python -m nexus.run --pipeline smoke --set fail_node=step_b --set fail_times=1

``--set key=value`` may repeat. Keys must be in the pipeline's input schema,
and values are converted to the schema's types (``max_retries=3`` becomes an
int).

Progress is logged to stderr, one line per finished node or retry; a failed
run also logs its traceback there once. The result is printed to stdout as
JSON. Exit codes:
  0  completed          1  completed but the run recorded errors
  2  bad arguments      3  failed: a node raised and its retries ran out
"""

import argparse
import importlib
import importlib.util
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, get_type_hints

from dotenv import load_dotenv
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from pydantic import TypeAdapter, ValidationError

from nexus.errors import InputError
from nexus.state import BaseState

log = logging.getLogger("nexus.run")

RESERVED = frozenset(get_type_hints(BaseState))
CONFIG_FILE = Path("langgraph.json")
NEXUS_ROOT = Path(__file__).resolve().parents[2]
EXIT_OK, EXIT_RUN_ERRORS, EXIT_FAILED = 0, 1, 3
RETRY_LOGGER = "langgraph.pregel._retry"


@dataclass
class RunResult:
    thread_id: str
    status: Literal["completed", "failed"]
    state: dict[str, Any]
    failed_node: str | None = None
    error: str | None = None

    @property
    def exit_code(self) -> int:
        if self.status == "failed":
            return EXIT_FAILED
        return EXIT_RUN_ERRORS if self.state.get("errors") else EXIT_OK


# --- loading graphs from langgraph.json -------------------------------------


def graph_specs(config_file: Path = CONFIG_FILE) -> dict[str, str]:
    """Name -> "path/to/file.py:variable", as listed under "graphs"."""
    return json.loads(config_file.read_text(encoding="utf-8"))["graphs"]


def load_graph(name: str, config_file: Path = CONFIG_FILE) -> CompiledStateGraph:
    specs = graph_specs(config_file)
    if name not in specs:
        raise InputError(f"unknown pipeline {name!r} (known: {', '.join(sorted(specs))})")
    path, _, attr = specs[name].rpartition(":")
    module = _import_file(config_file.parent / path)
    return getattr(module, attr)


def _import_file(path: Path):
    # Import by dotted name when the file is on sys.path (it is, for the
    # installed nexus package), so it is the same module the tests import.
    # The deepest sys.path entry wins: with cwd on sys.path, src/nexus/x.py must
    # import as "nexus.x", not "src.nexus.x".
    path = path.resolve()
    names = []
    for entry in map(Path, sys.path):
        try:
            names.append(path.relative_to(entry.resolve()).with_suffix("").parts)
        except (ValueError, OSError):
            continue
    if names:
        return importlib.import_module(".".join(min(names, key=len)))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- running ----------------------------------------------------------------


def validate_inputs(graph: CompiledStateGraph, inputs: dict[str, Any]) -> dict[str, Any]:
    """Check run inputs against the pipeline's declared input schema.

    Rejects unknown keys (a typo must not silently fall back to defaults),
    engine-owned fields, and values of the wrong type. Raises ``InputError``.
    """
    builder = graph.builder
    if builder.input_schema is builder.state_schema:
        raise InputError("pipeline must declare input_schema= on its StateGraph")
    allowed = set(get_type_hints(builder.input_schema)) - RESERVED
    unknown = sorted(set(inputs) - allowed)
    if unknown:
        raise InputError(f"unknown input(s) {unknown}; allowed: {sorted(allowed)}")
    try:
        return TypeAdapter(builder.input_schema).validate_python(inputs)
    except ValidationError as exc:
        raise InputError(str(exc)) from None


def run_pipeline(
    graph: CompiledStateGraph,
    thread_id: str,
    inputs: dict[str, Any],
    checkpointer: BaseCheckpointSaver | None = None,
) -> RunResult:
    """Run a graph to the end. Bad inputs raise ``InputError``; a node that
    fails after its retries does not raise, it returns ``status="failed"``."""
    inputs = validate_inputs(graph, inputs)
    graph = graph.copy(update={"checkpointer": checkpointer or InMemorySaver()})
    config = {"configurable": {"thread_id": thread_id}}
    log.info("start graph=%s thread=%s", graph.get_name(), thread_id)
    try:
        for update in graph.stream(inputs, config=config, stream_mode="updates"):
            for node, delta in update.items():
                # Key names only: values may hold data that shouldn't reach logs.
                log.info("node=%s wrote=%s", node, sorted(delta or {}))
    except Exception as exc:
        # The checkpoint keeps every finished step and names the failed task.
        snapshot = graph.get_state(config)
        failed = next((t for t in snapshot.tasks if t.error), None)
        result = RunResult(
            thread_id,
            "failed",
            snapshot.values,
            failed_node=failed.name if failed else None,
            error=f"{type(exc).__name__}: {exc}",
        )
        log.error("failed node=%s error=%s", result.failed_node, result.error, exc_info=exc)
        return result

    log.info("completed thread=%s", thread_id)
    return RunResult(thread_id, "completed", graph.get_state(config).values)


# --- CLI --------------------------------------------------------------------


def load_env() -> None:
    """Load secrets (ANTHROPIC_API_KEY, ...) from nexus's git-ignored .env.

    Variables already set in the environment win over the file.
    """
    load_dotenv(NEXUS_ROOT / ".env", override=False)


class OneLineRetries(logging.Filter):
    """LangGraph logs every retry with a full traceback; keep each to one line."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == RETRY_LOGGER:
            record.exc_info = None
            record.exc_text = None
        return True


def log_handler() -> logging.Handler:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    handler.addFilter(OneLineRetries())
    return handler


def parse_sets(pairs: list[str]) -> dict[str, str]:
    inputs: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise InputError(f"expected key=value, got {pair!r}")
        inputs[key] = value
    return inputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m nexus.run",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pipeline", required=True, help="a graph name from langgraph.json")
    parser.add_argument("--thread", help="thread id (default: <pipeline>-<timestamp>)")
    parser.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, handlers=[log_handler()])
    load_env()

    thread_id = args.thread or f"{args.pipeline}-{datetime.now():%Y%m%dT%H%M%S}"
    try:
        result = run_pipeline(load_graph(args.pipeline), thread_id, parse_sets(args.sets))
    except InputError as exc:
        parser.error(str(exc))

    print(json.dumps(asdict(result), indent=2, default=str))
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
