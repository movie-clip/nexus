import pytest

from nexus.run import run_pipeline


@pytest.fixture
def run(smoke_graph):
    def _run(scenario: str, **extra):
        inputs = {"scenario": scenario, **extra}
        return run_pipeline(smoke_graph, f"test-{scenario}", inputs).state

    return _run


def test_ok_goes_straight_to_finish(run):
    final = run("ok")
    assert final["visited"] == ["step_a", "step_b", "check", "finish"]
    assert final["verdict"] == "ok"
    assert final.get("errors", []) == []


def test_breach_routes_to_finish_with_breach_verdict(run):
    final = run("breach")
    assert final["visited"] == ["step_a", "step_b", "check", "finish"]
    assert final["verdict"] == "breach"


def test_flaky_retries_once_then_passes(run):
    final = run("flaky")
    assert final["visited"] == ["step_a", "step_b", "check", "step_b", "check", "finish"]
    assert final["retries"] == {"check": 1}
    assert final["verdict"] == "ok"


@pytest.mark.parametrize("limit", [0, 1, 3])
def test_stuck_stops_at_retry_limit_and_records_error(run, limit):
    final = run("stuck", max_retries=limit)
    assert final["visited"].count("check") == limit + 1
    assert final.get("retries", {}).get("check", 0) == limit
    assert final["verdict"] == "give_up"
    assert final["errors"] == [f"check: retry limit ({limit}) reached"]
    assert final["visited"][-1] == "finish"
