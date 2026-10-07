import json
import logging
import sys

import pytest

from nexus.errors import InputError
from nexus.pipelines import smoke
from nexus.retry import GIVE_UP, RETRY, retry_update
from nexus.run import graph_specs, load_graph, main, parse_sets, run_pipeline
from tests.conftest import ROOT


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


def test_every_graph_in_langgraph_json_loads_as_the_package_module():
    specs = graph_specs(ROOT / "langgraph.json")
    assert specs  # at least one pipeline
    for name in specs:
        graph = load_graph(name, ROOT / "langgraph.json")
        assert graph.builder.input_schema is not graph.builder.state_schema
    # Same module object the tests import, not a second copy under another name.
    assert load_graph("smoke", ROOT / "langgraph.json") is smoke.graph
    assert "src.nexus.pipelines.smoke" not in sys.modules


def test_unknown_pipeline_is_rejected():
    with pytest.raises(InputError, match="unknown pipeline 'nope'"):
        load_graph("nope", ROOT / "langgraph.json")


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        ({"scenaro": "stuck"}, "unknown input"),  # typo must not fall back to defaults
        ({"scenario": "bogus"}, "scenario"),
        ({"max_retries": "many"}, "max_retries"),
        ({"fail_node": "nowhere"}, "fail_node"),
        ({"errors": ["x"]}, "unknown input"),  # engine fields are not inputs
        ({"cost_usd": 5}, "unknown input"),
    ],
)
def test_bad_inputs_are_rejected(smoke_graph, inputs, message):
    with pytest.raises(InputError, match=message):
        run_pipeline(smoke_graph, "bad", inputs)


def test_cli_strings_are_converted_to_schema_types(smoke_graph):
    final = run_pipeline(smoke_graph, "conv", {"scenario": "stuck", "max_retries": "1"}).state
    assert final["retries"] == {"check": 1}


def test_parse_sets_keeps_values_as_strings():
    assert parse_sets(["a=3", "b=x=y", "c="]) == {"a": "3", "b": "x=y", "c": ""}
    with pytest.raises(InputError):
        parse_sets(["novalue"])


def test_progress_is_logged_per_node_without_values(smoke_graph, caplog):
    with caplog.at_level(logging.INFO, logger="nexus.run"):
        run_pipeline(smoke_graph, "log", {"scenario": "ok"})
    nodes = [r.getMessage() for r in caplog.records if r.getMessage().startswith("node=")]
    assert [n.split()[0] for n in nodes] == [
        "node=step_a",
        "node=step_b",
        "node=check",
        "node=finish",
    ]
    assert "'ok'" not in caplog.text  # key names are logged, values are not


def test_cli_ok_run_prints_result_and_exits_0(in_repo_root, capsys):
    code = main(["--pipeline", "smoke", "--thread", "cli-ok", "--set", "scenario=ok"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["thread_id"] == "cli-ok"
    assert out["status"] == "completed"
    assert out["state"]["verdict"] == "ok"


def test_cli_exits_1_when_run_records_errors(in_repo_root, capsys):
    code = main(["--pipeline", "smoke", "--set", "scenario=stuck", "--set", "max_retries=1"])
    out = json.loads(capsys.readouterr().out)
    assert code == 1
    assert out["thread_id"].startswith("smoke-")
    assert out["state"]["errors"]


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--pipeline", "smoke", "--set", "scenaro=ok"], "unknown input"),
        (["--pipeline", "nope"], "unknown pipeline"),
    ],
)
def test_cli_bad_arguments_are_usage_errors(in_repo_root, capsys, argv, message):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
    assert message in capsys.readouterr().err
