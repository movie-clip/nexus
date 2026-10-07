import json
import logging
import subprocess
import sys

from nexus.run import RETRY_LOGGER, log_handler, run_pipeline
from tests.conftest import FAST_RETRY, ROOT


def fail(smoke_graph, node, kind="transient", times=0):
    inputs = {"fail_node": node, "fail_kind": kind, "fail_times": times}
    return run_pipeline(smoke_graph, "fail", inputs)


def retries_logged(caplog, node: str) -> int:
    return sum(
        1
        for r in caplog.records
        if r.name == RETRY_LOGGER and r.getMessage().startswith(f"Retrying task {node} ")
    )


def test_transient_failure_is_retried_then_node_succeeds(smoke_graph, caplog):
    with caplog.at_level(logging.INFO):
        result = fail(smoke_graph, "step_b", times=2)
    assert result.status == "completed"
    assert result.exit_code == 0
    assert result.state["visited"] == ["step_a", "step_b", "check", "finish"]
    assert retries_logged(caplog, "step_b") == 2


def test_persistent_transient_failure_fails_run_after_max_attempts(smoke_graph, caplog):
    with caplog.at_level(logging.INFO):
        result = fail(smoke_graph, "step_b")
    assert result.status == "failed"
    assert result.exit_code == 3
    assert result.failed_node == "step_b"
    assert result.error.startswith("InjectedTransient: ")
    assert retries_logged(caplog, "step_b") == FAST_RETRY.max_attempts - 1


def test_bug_is_not_retried(smoke_graph, caplog):
    with caplog.at_level(logging.INFO):
        result = fail(smoke_graph, "check", kind="bug")
    assert result.status == "failed"
    assert result.failed_node == "check"
    assert result.error == "InjectedBug: injected bug in check"
    assert retries_logged(caplog, "check") == 0


def test_failed_run_keeps_the_work_finished_before_the_failure(smoke_graph):
    result = fail(smoke_graph, "check", kind="bug")
    assert result.state["visited"] == ["step_a", "step_b"]


def test_retry_log_lines_carry_no_traceback():
    try:
        raise ConnectionError("down")
    except ConnectionError:
        record = logging.getLogger(RETRY_LOGGER).makeRecord(
            RETRY_LOGGER, logging.INFO, __file__, 1, "Retrying task x", (), sys.exc_info()
        )
    handler = log_handler()
    assert handler.filter(record)
    assert "\n" not in handler.format(record)


def test_cli_process_reports_failure_as_json_and_logs_traceback_once():
    # A real process: the only way to see what actually reaches stderr.
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "nexus.run",
            "--pipeline",
            "smoke",
            "--thread",
            "proc-fail",
            "--set",
            "fail_node=check",
            "--set",
            "fail_kind=bug",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=ROOT,
    )
    assert proc.returncode == 3
    out = json.loads(proc.stdout)
    assert out["status"] == "failed"
    assert out["failed_node"] == "check"
    assert out["error"] == "InjectedBug: injected bug in check"
    assert proc.stderr.count("Traceback") == 1
