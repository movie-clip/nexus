import json
import logging

import pytest

from nexus import registry
from nexus.run import InputError, main, parse_sets, run_pipeline
from nexus.state import GIVE_UP, RETRY, retry_update


def test_retry_update_bumps_until_limit():
    assert retry_update({}, 2, "n") == {"verdict": RETRY, "retries": {"n": 1}}
    assert retry_update({"retries": {"n": 1}}, 2, "n") == {"verdict": RETRY, "retries": {"n": 2}}
    assert retry_update({"retries": {"n": 2}}, 2, "n") == {
        "verdict": GIVE_UP,
        "errors": ["n: retry limit (2) reached"],
    }


def test_retry_counts_are_per_node():
    state = {"retries": {"verify": 2}}
    assert retry_update(state, 2, "fetch") == {"verdict": RETRY, "retries": {"fetch": 1}}


def test_registry_rejects_duplicates_and_unknown_names():
    with pytest.raises(ValueError, match="already registered"):
        registry.register("smoke")(lambda: None)
    with pytest.raises(KeyError, match="unknown pipeline 'nope'"):
        registry.get("nope")


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        ({"scenaro": "stuck"}, "unknown input"),  # typo must not fall back to defaults
        ({"scenario": "bogus"}, "scenario"),
        ({"max_retries": "many"}, "max_retries"),
        ({"errors": ["x"]}, "unknown input"),  # engine fields are not inputs
        ({"cost_usd": 5}, "unknown input"),
    ],
)
def test_bad_inputs_are_rejected(inputs, message):
    with pytest.raises(InputError, match=message):
        run_pipeline("smoke", "bad", inputs)


def test_cli_strings_are_converted_to_schema_types():
    final = run_pipeline("smoke", "conv", {"scenario": "stuck", "max_retries": "1"})
    assert final["retries"] == {"check": 1}


def test_parse_sets_keeps_values_as_strings():
    assert parse_sets(["a=3", "b=x=y", "c="]) == {"a": "3", "b": "x=y", "c": ""}
    with pytest.raises(InputError):
        parse_sets(["novalue"])


def test_progress_is_logged_per_node_without_values(caplog):
    with caplog.at_level(logging.INFO, logger="nexus.run"):
        run_pipeline("smoke", "log", {"scenario": "ok"})
    nodes = [r.getMessage() for r in caplog.records if r.getMessage().startswith("node=")]
    assert [n.split()[0] for n in nodes] == [
        "node=step_a",
        "node=step_b",
        "node=check",
        "node=finish",
    ]
    assert "'ok'" not in caplog.text  # key names are logged, values are not


def test_cli_ok_run_prints_state_and_exits_0(capsys):
    code = main(["--pipeline", "smoke", "--thread", "cli-ok", "--set", "scenario=ok"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["thread"] == "cli-ok"
    assert out["state"]["verdict"] == "ok"


def test_cli_exits_1_when_run_records_errors(capsys):
    code = main(["--pipeline", "smoke", "--set", "scenario=stuck", "--set", "max_retries=1"])
    out = json.loads(capsys.readouterr().out)
    assert code == 1
    assert out["thread"].startswith("smoke-")
    assert out["state"]["errors"]


def test_cli_bad_input_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--pipeline", "smoke", "--set", "scenaro=ok"])
    assert exc.value.code == 2
    assert "unknown input" in capsys.readouterr().err
