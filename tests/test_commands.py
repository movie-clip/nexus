"""The tester's command layer: config, checking values, running, summarizing."""

import re
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from nexus import commands
from nexus.commands import CommandError, Outcome, Prepared
from nexus.errors import InputError

# A stand-in for a build or test tool. Its first argument picks the behaviour.
FAKE_TOOL = textwrap.dedent(
    """\
    import os, subprocess, sys, time
    mode = sys.argv[1]
    if mode == "ok":
        print("build ok")
    elif mode == "fail":
        for _ in range(2):
            print("Assets/A.cs(3,5): error CS0103: The name 'x' does not exist")
        print("warning CS0168: unused")
        sys.exit(1)
    elif mode == "results":
        open(sys.argv[2], "w").write(sys.argv[3])
        sys.exit(2)
    elif mode == "touch":
        with open("tracked.txt", "a") as f:
            f.write("changed\\n")
    elif mode == "env":
        print("KEY=" + os.environ.get("ANTHROPIC_API_KEY", "<unset>"))
    elif mode == "hang":
        # A grandchild that would write a marker after 2 s, unless it is killed.
        subprocess.Popen([sys.executable, "-c",
            "import time; time.sleep(2); open('marker.txt', 'w').write('alive')"])
        time.sleep(30)
    """
)

NUNIT = """<?xml version="1.0" encoding="utf-8"?>
<test-run total="3" passed="1" failed="2" skipped="0">
  <test-suite>
    <test-case fullname="A.Tests.Passes" result="Passed" />
    <test-case fullname="A.Tests.Breaks" result="Failed">
      <failure><message><![CDATA[Expected 2
  But was 3]]></message></failure>
    </test-case>
    <test-case fullname="A.Tests.Throws" result="Failed">
      <failure><message>NullReferenceException</message></failure>
    </test-case>
  </test-suite>
</test-run>
"""


def toml_str(path) -> str:
    return "'" + str(path) + "'"  # TOML literal string: backslashes stay as they are


def write_config(projects_dir: Path, project: Path, body: str) -> None:
    projects_dir.mkdir(exist_ok=True)
    text = f"path = {toml_str(project.as_posix())}\n\n" + textwrap.dedent(body)
    (projects_dir / "p.toml").write_text(text.replace("PYTHON", toml_str(sys.executable)))


TOOL_CONFIG = """\
    [commands.tool]
    description = "Fake tool."
    run = [PYTHON, "tool.py", "{mode}", ["{extra}"]]
    timeout = 1
    approval = "auto"

    [commands.tool.params]
    mode = { pattern = "ok|fail|results|touch|env|hang", default = "ok" }
    extra = { pattern = "[a-z]+" }
"""


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "tool.py").write_text(FAKE_TOOL)
    return root


@pytest.fixture
def config(tmp_path, project):
    write_config(tmp_path / "projects", project, TOOL_CONFIG)
    return commands.find_config(project, tmp_path / "projects")


def prep(config, project, tmp_path, name="tool", **args) -> Prepared:
    return commands.prepare(config, project, name, args, tmp_path / "logs" / "1-tool")


# --- config ---------------------------------------------------------------------


def test_find_config_matches_the_project_folder(tmp_path, project, config):
    assert config is not None and "tool" in config.commands
    other = tmp_path / "other"
    other.mkdir()
    assert commands.find_config(other, tmp_path / "projects") is None
    if sys.platform == "win32":
        assert commands.find_config(str(project).upper(), tmp_path / "projects") is not None


@pytest.mark.parametrize(
    ("body", "error"),
    [
        ('[commands.x]\ndescription="d"\nrun=["a", "{nope}"]', "not a declared param"),
        ('[commands.x]\ndescription="d"', "exactly one of run or steps"),
        ('[commands.x]\ndescription="d"\nrun=["a"]\ncwd="../up"', "cwd must be inside"),
        ('[commands.x]\ndescription="d"\nrun=["{p}"]\nparams.p={pattern="a"}', "fixed program"),
        ('[commands.x]\ndescription="d"\nrun=["a"]\nparams.p={pattern="a",default="b"}', "default"),
        ('[commands.Bad_Name]\ndescription="d"\nrun=["a"]', "command name"),
        ('[commands.x]\ndescription="d"\nrun=["a"]\nshell=true', "Extra inputs"),
        ('[commands.x]\ndescription="d"\nrun=["a"]\napproval="never"', "approval"),
        ('[commands.x]\ndescription="d"\nrun=["a"]\nparams.p={pattern="("}', "bad pattern"),
        ('verify=["nope"]\n[commands.x]\ndescription="d"\nrun=["a"]', "verify: unknown command"),
        (
            'verify=["x", "x"]\n[commands.x]\ndescription="d"\nrun=["a"]',
            "verify: x is listed twice",
        ),
        (
            'verify=["x"]\n[commands.x]\ndescription="d"\nrun=["a", "{p}"]\nparams.p={pattern="a"}',
            "verify: x needs a value for p",
        ),
    ],
)
def test_bad_configs_are_refused(tmp_path, project, body, error):
    write_config(tmp_path / "projects", project, body)
    with pytest.raises(InputError, match=error):
        commands.find_config(project, tmp_path / "projects")


def test_the_impact_config_is_valid():
    from nexus.run import NEXUS_ROOT

    config = commands.load_config(NEXUS_ROOT / "projects" / "impact.toml")
    assert set(config.commands) == {"compile", "unity-tests"}
    assert config.verify == ["compile"]
    tests = config.commands["unity-tests"]
    assert tests.approval == "auto"
    pattern = tests.params["filter"].pattern
    for ok in ["A.B", "Impact.EditModeTests.Core.X;Impact.EditModeTests.Core.Y"]:
        assert re.fullmatch(pattern, ok)
    for bad in ["A;", ";A", "A; B", "A;-quit", "A;;B"]:
        assert not re.fullmatch(pattern, bad)


def test_verify_is_empty_unless_listed(config):
    assert config.verify == []


def test_a_verify_command_may_have_optional_params(tmp_path, project):
    body = """\
        verify = ["x"]
        [commands.x]
        description = "d"
        run = ["a", "{mode}", ["{extra}"]]
        params.mode = { pattern = "a", default = "a" }
        params.extra = { pattern = "b" }
    """
    write_config(tmp_path / "projects", project, body)
    config = commands.find_config(project, tmp_path / "projects")
    assert config is not None and config.verify == ["x"]


def test_describe_lists_commands_and_params(config):
    text = commands.describe(config)
    assert "- tool: Fake tool." in text
    assert "mode (must match 'ok|fail|results|touch|env|hang', default 'ok')" in text
    assert "can't run anything" in commands.describe(None)


# --- preparing ------------------------------------------------------------------


def test_prepare_fills_defaults_and_drops_unfilled_optional_groups(config, project, tmp_path):
    cmd = prep(config, project, tmp_path)
    assert cmd.steps == [[sys.executable, "tool.py", "ok"]]
    assert cmd.cwd == str(project.resolve())
    assert cmd.log_file.endswith("1-tool.log")
    assert cmd.results_file.endswith("1-tool-results.xml")


def test_prepare_includes_an_optional_group_with_a_value(config, project, tmp_path):
    assert prep(config, project, tmp_path, extra="abc").steps[0][-1] == "abc"


@pytest.mark.parametrize(
    ("name", "args", "error"),
    [
        ("tool", {"mode": "ok; del *"}, "doesn't match"),
        ("tool", {"mode": "ok --evil"}, "doesn't match"),
        ("tool", {"extra": "a b"}, "doesn't match"),
        ("tool", {"other": "x"}, "has no param other"),
        ("nope", {}, "unknown command 'nope'; available: tool"),
    ],
)
def test_prepare_refuses_bad_requests(config, project, tmp_path, name, args, error):
    with pytest.raises(CommandError, match=error):
        commands.prepare(config, project, name, args, tmp_path / "1-x")


def test_prepare_needs_values_for_required_params(tmp_path, project):
    body = '[commands.x]\ndescription="d"\nrun=[PYTHON, "{p}"]\nparams.p={pattern="a"}'
    write_config(tmp_path / "projects", project, body)
    config = commands.find_config(project, tmp_path / "projects")
    with pytest.raises(CommandError, match="needs a value for p"):
        commands.prepare(config, project, "x", {}, tmp_path / "1-x")


def test_missing_programs_and_batch_files_are_refused(tmp_path, project):
    (project / "build.cmd").write_text("@echo off\n")
    body = """\
        [commands.missing]
        description = "d"
        run = ["no-such-program-xyz"]

        [commands.batch]
        description = "d"
        run = ["./build.cmd"]
    """
    write_config(tmp_path / "projects", project, body)
    config = commands.find_config(project, tmp_path / "projects")
    with pytest.raises(CommandError, match="not found"):
        commands.prepare(config, project, "missing", {}, tmp_path / "1-x")
    with pytest.raises(CommandError, match="is a batch file"):
        commands.prepare(config, project, "batch", {}, tmp_path / "1-x")


def test_no_config_means_no_commands(project, tmp_path):
    with pytest.raises(CommandError, match="no commands are configured"):
        commands.prepare(None, project, "tool", {}, tmp_path / "1-x")


# --- running --------------------------------------------------------------------


def test_success_is_logged(config, project, tmp_path):
    cmd = prep(config, project, tmp_path)
    outcome = commands.run_prepared(cmd)
    assert outcome.exit_code == 0
    log = Path(cmd.log_file).read_text()
    assert log.startswith("$ ") and "build ok" in log


def test_failure_report_has_unique_errors_and_the_tail(config, project, tmp_path):
    cmd = prep(config, project, tmp_path, mode="fail")
    outcome = commands.run_prepared(cmd)
    report = commands.summarize(cmd, outcome, [])
    assert "Result: FAILED, exit code 1" in report
    assert "Errors (1 unique):" in report  # printed twice, reported once
    assert "error CS0103" in report
    assert "warning CS0168" in report  # the output tail
    assert "Files git reports as changed by this command: none" in report
    assert f"Full log: {cmd.log_file}" in report


def test_nunit_results_are_summarized(config, project, tmp_path):
    cmd = prep(config, project, tmp_path, mode="results")
    cmd = Prepared(**{**cmd.__dict__, "steps": [[*cmd.steps[0], cmd.results_file, NUNIT]]})
    report = commands.summarize(cmd, commands.run_prepared(cmd), [])
    assert "Tests: 3 run, 1 passed, 2 failed, 0 skipped" in report
    assert "FAILED A.Tests.Breaks: Expected 2 | But was 3" in report
    assert "FAILED A.Tests.Throws: NullReferenceException" in report
    summary = commands.nunit_summary(Path(cmd.results_file))
    assert summary is not None and "Passes" not in summary


def test_a_stale_results_file_is_never_reported(config, project, tmp_path):
    cmd = prep(config, project, tmp_path)
    Path(cmd.results_file).parent.mkdir(parents=True, exist_ok=True)
    Path(cmd.results_file).write_text(NUNIT)  # left over from an earlier run
    report = commands.summarize(cmd, commands.run_prepared(cmd), [])
    assert "Tests:" not in report


def test_timeout_kills_the_whole_process_tree(config, project, tmp_path):
    cmd = prep(config, project, tmp_path, mode="hang")
    start = time.monotonic()
    outcome = commands.run_prepared(cmd)
    assert outcome.exit_code is None
    assert time.monotonic() - start < 10
    time.sleep(2.5)  # the grandchild would have written its marker by now
    assert not (project / "marker.txt").exists()
    assert "TIMED OUT after 1 s" in commands.summarize(cmd, outcome, [])


def test_steps_stop_at_the_first_failure(tmp_path, project):
    body = """\
        [commands.two]
        description = "d"
        steps = [[PYTHON, "tool.py", "fail"], [PYTHON, "tool.py", "touch"]]
    """
    write_config(tmp_path / "projects", project, body)
    config = commands.find_config(project, tmp_path / "projects")
    cmd = commands.prepare(config, project, "two", {}, tmp_path / "1-two")
    outcome = commands.run_prepared(cmd)
    assert (outcome.exit_code, outcome.steps_run) == (1, 1)
    assert not (project / "tracked.txt").exists()
    assert "Steps run: 1 of 2" in commands.summarize(cmd, outcome, [])


def test_commands_never_see_nexus_secrets(config, project, tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    cmd = prep(config, project, tmp_path, mode="env")
    commands.run_prepared(cmd)
    assert "KEY=<unset>" in Path(cmd.log_file).read_text()


# --- git check ------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_git_check_reports_files_a_command_changed(config, project, tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=project, check=True, capture_output=True)

    git("init", "-q")
    (project / "tracked.txt").write_text("v1\n")
    (project / "other.txt").write_text("v1\n")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    (project / "tracked.txt").write_text("modified before the command\n")

    before = commands.git_snapshot(project)
    commands.run_prepared(prep(config, project, tmp_path, mode="touch"))
    after = commands.git_snapshot(project)
    # tracked.txt was already modified; the command changed it again.
    assert commands.changed_between(before, after) == ["tracked.txt"]


def test_git_check_is_skipped_outside_a_repository(tmp_path):
    folder = tmp_path / "not-a-repo"
    folder.mkdir()
    assert commands.changed_between(commands.git_snapshot(folder), {}) == []


def test_outcome_without_a_log_line_still_summarizes(tmp_path):
    log = tmp_path / "x.log"
    log.write_text("")
    cmd = Prepared("x", [["x"]], str(tmp_path), 5, "ask", str(tmp_path / "r.xml"), str(log))
    assert "Result: succeeded, exit code 0, after 0 s" in commands.summarize(
        cmd, Outcome(0, 0.2, 1), []
    )


def test_output_tail_is_short_on_success_and_long_on_failure(tmp_path):
    log = tmp_path / "x.log"
    log.write_text("\n".join(f"line {i}" for i in range(100)))
    cmd = Prepared("x", [["x"]], str(tmp_path), 5, "ask", str(tmp_path / "r.xml"), str(log))
    ok = commands.summarize(cmd, Outcome(0, 1, 1), [])
    failed = commands.summarize(cmd, Outcome(1, 1, 1), [])
    assert "Last 10 lines of output:" in ok and "line 89" not in ok and "line 90" in ok
    assert "Last 40 lines of output:" in failed and "line 60" in failed


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_git_check_reads_renames_as_one_entry(project):
    # With -z, a rename is "R  new" followed by the old path as its own field.
    def git(*args):
        subprocess.run(["git", *args], cwd=project, check=True, capture_output=True)

    git("init", "-q")
    (project / "old name.txt").write_text("v1\n")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    git("mv", "old name.txt", "new name.txt")
    snapshot = commands.git_snapshot(project)
    assert snapshot is not None
    assert list(snapshot) == ["new name.txt"]
    assert snapshot["new name.txt"].startswith("R ")


def test_a_step_with_no_time_left_never_starts(project, tmp_path):
    with open(tmp_path / "x.log", "wb") as log:
        argv = [sys.executable, "tool.py", "touch"]
        assert commands._run_one(argv, str(project), {}, log, remaining=0) is None
    assert not (project / "tracked.txt").exists()


def test_long_error_and_failure_lists_are_cut(tmp_path):
    log = tmp_path / "x.log"
    log.write_text("\n".join(f"A.cs({i},1): error CS{1000 + i}: e{i}" for i in range(25)))
    cases = "".join(
        f'<test-case fullname="T{i}" result="Failed"><failure><message>m</message></failure>'
        "</test-case>"
        for i in range(23)
    )
    results = tmp_path / "r.xml"
    results.write_text(
        f'<test-run total="23" passed="0" failed="23" skipped="0">{cases}</test-run>'
    )
    cmd = Prepared("x", [["x"]], str(tmp_path), 5, "ask", str(results), str(log))
    report = commands.summarize(cmd, Outcome(1, 1, 1), [])
    assert "Errors (25 unique):" in report and "... and 5 more" in report
    assert "FAILED T19: m" in report and "FAILED T20" not in report
    assert "... and 3 more failures" in report


def results_command(tmp_path, results: str | None) -> Prepared:
    """A command that passes {results_file}; ``results`` is what it left there."""
    log = tmp_path / "x.log"
    log.write_text("done\n")
    file = tmp_path / "r.xml"
    if results is not None:
        file.write_text(results)
    return Prepared("t", [["x", "-r", str(file)]], str(tmp_path), 5, "auto", str(file), str(log))


@pytest.mark.parametrize(
    ("results", "problem"),
    [
        # Unity's launcher returned before Unity had run the tests.
        (None, "no test results were written, so the tests did not run or did not finish"),
        # Unity exits 0 when the filter matches no test.
        ('<test-run total="0" passed="0" failed="0" skipped="0"/>', "no tests ran"),
        ("<test-run", "the test results file could not be read"),
    ],
)
def test_exit_0_without_test_results_is_a_failure(tmp_path, results, problem):
    report = commands.summarize(results_command(tmp_path, results), Outcome(0, 1, 1), [])
    assert f"Result: FAILED, exit code 0, after 1 s: {problem}" in report
    assert "succeeded" not in report


def test_exit_0_with_tests_run_succeeds(tmp_path):
    results = '<test-run total="2" passed="2" failed="0" skipped="0"/>'
    report = commands.summarize(results_command(tmp_path, results), Outcome(0, 1, 1), [])
    assert "Result: succeeded, exit code 0, after 1 s" in report
    assert "Tests: 2 run, 2 passed" in report


def test_unreadable_or_foreign_results_files(tmp_path):
    broken = tmp_path / "broken.xml"
    broken.write_text("<test-run")
    summary = commands.nunit_summary(broken)
    assert summary is not None and summary.startswith("Test results file could not be read")
    other = tmp_path / "junit.xml"
    other.write_text("<testsuites/>")
    assert commands.nunit_summary(other) is None


def test_a_missing_working_folder_is_refused(tmp_path, project):
    write_config(
        tmp_path / "projects", project, '[commands.x]\ndescription="d"\nrun=["a"]\ncwd="gone"'
    )
    config = commands.find_config(project, tmp_path / "projects")
    with pytest.raises(CommandError, match="working folder .* doesn't exist"):
        commands.prepare(config, project, "x", {}, tmp_path / "1-x")
