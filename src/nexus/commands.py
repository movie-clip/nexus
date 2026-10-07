"""Project commands the tester may run: builds and tests the user listed, nothing else.

Each project's commands live in nexus, not in the project:
``projects/<name>.toml`` holds the project's ``path`` and its ``[commands.<name>]``
tables. ``find_config`` picks the file whose ``path`` is the ``--project`` folder.

    [commands.compile]
    description = "What it checks, how long it takes, when to use it."
    run = ["dotnet", "build", "App.sln", ["-p:Filter={filter}"]]
    cwd = "src"          # relative to the project; default "."
    timeout = 600        # seconds for the whole command; default 600
    approval = "ask"     # "ask": the run pauses for the user's OK; "auto": runs at once

    [commands.compile.params]
    filter = { pattern = "[A-Za-z0-9_.]+", description = "..." }

A command is an argument list, never a shell line. ``{param}`` placeholders are
filled from values the orchestrator chose; each value must fully match its
param's ``pattern``, so it can't smuggle in extra arguments or commands. A
nested list is an optional group, left out unless every placeholder in it has
a value. ``{results_file}`` is a path nexus provides outside the project for a
test-results file (NUnit XML is summarized). ``steps = [[...], [...]]`` runs
several commands in order, stopping at the first that fails.

Commands run with nexus's own secrets (the names in nexus's ``.env`` and any
``ANTHROPIC_*`` variable) removed from their environment, with no input, and
with the whole process tree killed when the time limit is reached. The full
output goes to a log file under ``runs/logs/``; the tester's report carries a
short summary.
"""

import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import tomllib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from nexus.errors import InputError
from nexus.run import NEXUS_ROOT, runs_db_path

PROJECTS_DIR = NEXUS_ROOT / "projects"
BUILTINS = frozenset({"results_file"})
PLACEHOLDER = re.compile(r"\{(\w+)\}")
COMMAND_NAME = r"[a-z0-9][a-z0-9-]*"

Arg = str | list[str]


class CommandError(ValueError):
    """A command can't be prepared: unknown name, bad value, missing program."""


# --- the config file ----------------------------------------------------------


class ParamSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: str
    default: str | None = None
    description: str = ""

    @model_validator(mode="after")
    def _check(self) -> "ParamSpec":
        try:
            compiled = re.compile(self.pattern)
        except re.error as exc:
            raise ValueError(f"bad pattern {self.pattern!r}: {exc}") from exc
        if self.default is not None and not compiled.fullmatch(self.default):
            raise ValueError(f"default {self.default!r} doesn't match {self.pattern!r}")
        return self


class CommandSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str
    run: list[Arg] | None = None
    steps: list[list[Arg]] | None = None
    cwd: str = "."
    timeout: int = Field(600, gt=0, le=4 * 3600)
    approval: Literal["ask", "auto"] = "ask"
    params: dict[str, ParamSpec] = {}

    @model_validator(mode="after")
    def _check(self) -> "CommandSpec":
        if (self.run is None) == (self.steps is None):
            raise ValueError("give exactly one of run or steps")
        for step in self.all_steps:
            if not step or not isinstance(step[0], str) or PLACEHOLDER.search(step[0]):
                raise ValueError("each step must start with a fixed program name")
            for arg in step:
                for part in [arg] if isinstance(arg, str) else arg:
                    for name in PLACEHOLDER.findall(part):
                        if name not in self.params and name not in BUILTINS:
                            raise ValueError(f"placeholder {{{name}}} is not a declared param")
        rel = PurePosixPath(self.cwd.replace("\\", "/"))
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"cwd must be inside the project, got {self.cwd!r}")
        return self

    @property
    def all_steps(self) -> list[list[Arg]]:
        return [self.run] if self.run is not None else list(self.steps or [])


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    commands: dict[str, CommandSpec] = {}

    @model_validator(mode="after")
    def _check(self) -> "ProjectConfig":
        for name in self.commands:
            if not re.fullmatch(COMMAND_NAME, name):
                raise ValueError(f"command name {name!r} must match {COMMAND_NAME}")
        return self


def load_config(file: Path) -> ProjectConfig:
    try:
        with open(file, "rb") as f:
            return ProjectConfig.model_validate(tomllib.load(f))
    except (tomllib.TOMLDecodeError, ValidationError) as exc:
        raise InputError(f"{file}: {exc}") from exc


def find_config(project: str | Path, projects_dir: Path = PROJECTS_DIR) -> ProjectConfig | None:
    """The config whose ``path`` is this project folder, or None."""
    target = os.path.normcase(Path(project).resolve())  # case-insensitive on Windows
    for file in sorted(Path(projects_dir).glob("*.toml")):
        config = load_config(file)
        if os.path.normcase(Path(config.path).resolve()) == target:
            return config
    return None


def describe(config: ProjectConfig | None) -> str:
    """The command list as the orchestrator sees it."""
    if config is None or not config.commands:
        return "No commands are configured for this project, so the tester can't run anything."
    lines = []
    for name, spec in config.commands.items():
        lines.append(f"- {name}: {spec.description}")
        for pname, p in spec.params.items():
            default = f", default {p.default!r}" if p.default is not None else ""
            lines.append(f"    {pname} (must match {p.pattern!r}{default}): {p.description}")
    return "\n".join(lines)


# --- preparing a run ------------------------------------------------------------


@dataclass(frozen=True)
class Prepared:
    """A command ready to run: every value filled in and checked."""

    name: str
    steps: list[list[str]]
    cwd: str
    timeout: int
    approval: str
    results_file: str
    log_file: str

    def display(self) -> str:
        return "\n".join(_join(step) for step in self.steps)


def prepare(
    config: ProjectConfig | None,
    project: str | Path,
    name: str,
    args: dict[str, str],
    log_stem: Path,
) -> Prepared:
    if config is None or not config.commands:
        raise CommandError("no commands are configured for this project")
    spec = config.commands.get(name)
    if spec is None:
        raise CommandError(f"unknown command {name!r}; available: {', '.join(config.commands)}")
    unknown = sorted(set(args) - set(spec.params))
    if unknown:
        raise CommandError(f"{name} has no param {', '.join(unknown)}")
    values = {k: p.default for k, p in spec.params.items() if p.default is not None}
    for key, value in args.items():
        if not re.fullmatch(spec.params[key].pattern, value):
            raise CommandError(
                f"value {value!r} for {key} doesn't match {spec.params[key].pattern!r}"
            )
        values[key] = value
    values["results_file"] = str(log_stem.with_name(log_stem.name + "-results.xml"))

    cwd = (Path(project).resolve() / spec.cwd).resolve()
    if not cwd.is_dir():
        raise CommandError(f"working folder {cwd} doesn't exist")
    steps = []
    for step in spec.all_steps:
        argv = []
        for arg in step:
            if isinstance(arg, list):  # optional group
                if all(n in values for part in arg for n in PLACEHOLDER.findall(part)):
                    argv.extend(_fill(part, values) for part in arg)
            else:
                missing = [n for n in PLACEHOLDER.findall(arg) if n not in values]
                if missing:
                    raise CommandError(f"{name} needs a value for {', '.join(missing)}")
                argv.append(_fill(arg, values))
        argv[0] = _program(argv[0], cwd)
        steps.append(argv)
    return Prepared(
        name=name,
        steps=steps,
        cwd=str(cwd),
        timeout=spec.timeout,
        approval=spec.approval,
        results_file=values["results_file"],
        log_file=str(log_stem.with_name(log_stem.name + ".log")),
    )


def log_stem(thread_id: str, n: int, name: str) -> Path:
    """runs/logs/<thread>/<n>-<command>; next to the runs database."""
    safe_thread = re.sub(r"[^A-Za-z0-9_.-]", "_", thread_id)
    return runs_db_path().parent / "logs" / safe_thread / f"{n}-{name}"


def _fill(arg: str, values: dict[str, str]) -> str:
    return PLACEHOLDER.sub(lambda m: values[m.group(1)], arg)


def _program(program: str, cwd: Path) -> str:
    """The full path of the program, refusing batch files.

    Windows runs .bat/.cmd files through cmd.exe, which re-parses the
    arguments as a shell line; that would undo the protection of passing a list.
    """
    if "/" in program or "\\" in program:
        found = str((cwd / program).resolve()) if (cwd / program).is_file() else None
    else:
        found = shutil.which(program)
    if found is None:
        raise CommandError(f"program {program!r} not found")
    if Path(found).suffix.lower() in (".bat", ".cmd"):
        raise CommandError(f"{found} is a batch file; list the program it runs instead")
    return found


def _join(argv: list[str]) -> str:
    return subprocess.list2cmdline(argv) if sys.platform == "win32" else shlex.join(argv)


# --- running -------------------------------------------------------------------


@dataclass
class Outcome:
    exit_code: int | None  # None when it timed out
    seconds: float
    steps_run: int


def run_prepared(cmd: Prepared) -> Outcome:
    log_path = Path(cmd.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    Path(cmd.results_file).unlink(missing_ok=True)  # never summarize a stale file
    env = child_env()
    start = time.monotonic()
    exit_code: int | None = 0
    steps_run = 0
    with open(log_path, "wb") as log:
        for argv in cmd.steps:
            log.write(f"$ {_join(argv)}\n".encode())
            log.flush()
            steps_run += 1
            exit_code = _run_one(argv, cmd.cwd, env, log, cmd.timeout - (time.monotonic() - start))
            if exit_code != 0:
                break
    return Outcome(exit_code, time.monotonic() - start, steps_run)


def _run_one(argv: list[str], cwd: str, env: dict, log, remaining: float) -> int | None:
    if remaining <= 0:
        return None
    windows = sys.platform == "win32"
    proc = subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if windows else 0,
        start_new_session=not windows,
    )
    try:
        return proc.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        return None


def _kill_tree(proc: subprocess.Popen) -> None:
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    else:
        import signal

        os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()


def child_env() -> dict[str, str]:
    """This process's environment without nexus's own secrets."""
    secret = {name.upper() for name in dotenv_values(NEXUS_ROOT / ".env")}
    return {
        k: v
        for k, v in os.environ.items()
        if k.upper() not in secret and not k.upper().startswith("ANTHROPIC_")
    }


# --- did the command change the project? --------------------------------------


def git_snapshot(project: str | Path) -> dict[str, str] | None:
    """Each changed or untracked file git reports, with a hash of its content.

    None when the project isn't a git repository (or git isn't installed).
    Comparing a snapshot from before and after a command shows which files it
    changed, even ones that were already modified before it ran.
    """
    git = shutil.which("git")
    if git is None:
        return None
    result = subprocess.run(
        [git, "status", "--porcelain=v1", "-z", "-uall"],
        cwd=project,
        capture_output=True,
        stdin=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        return None
    root = Path(project).resolve()
    top = subprocess.run(
        [git, "rev-parse", "--show-toplevel"], cwd=project, capture_output=True, text=True
    )
    if top.returncode == 0:
        root = Path(top.stdout.strip()).resolve()
    snapshot = {}
    entries = result.stdout.decode("utf-8", "replace").split("\0")
    i = 0
    while i < len(entries):
        entry = entries[i]
        i += 1
        if len(entry) < 4:
            continue
        status, path = entry[:2], entry[3:]
        if status[0] in "RC":
            i += 1  # the original path of a rename/copy follows
        file = root / path
        digest = hashlib.sha1(file.read_bytes()).hexdigest() if file.is_file() else "missing"
        snapshot[path] = f"{status}:{digest}"
    return snapshot


def changed_between(before: dict[str, str] | None, after: dict[str, str] | None) -> list[str]:
    if before is None or after is None:
        return []
    return sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))


# --- the tester's report ---------------------------------------------------------

# MSBuild/C# style, as dotnet and Unity print it: "file(1,2): error CS0103: ..."
ERROR_LINE = re.compile(r"\berror [A-Z]+\d+\s*:")
# The output tail: enough to diagnose a failure; on success only the closing lines.
MAX_ERRORS, MAX_FAILURES, TAIL_LINES, TAIL_LINES_OK, MAX_LINE = 20, 20, 40, 10, 400


def summarize(cmd: Prepared, outcome: Outcome, changed: list[str]) -> str:
    log_text = Path(cmd.log_file).read_bytes().decode("utf-8", "replace")
    lines = log_text.splitlines()
    out = [f"Command: {cmd.name}", f"Ran in: {cmd.cwd}", cmd.display()]
    if outcome.exit_code is None:
        out.append(f"Result: TIMED OUT after {cmd.timeout} s (process tree killed)")
    else:
        verdict = "succeeded" if outcome.exit_code == 0 else "FAILED"
        out.append(
            f"Result: {verdict}, exit code {outcome.exit_code}, after {outcome.seconds:.0f} s"
        )
    if len(cmd.steps) > 1:
        out.append(f"Steps run: {outcome.steps_run} of {len(cmd.steps)}")

    errors = list(dict.fromkeys(line.strip() for line in lines if ERROR_LINE.search(line)))
    if errors:
        out.append(f"\nErrors ({len(errors)} unique):")
        out.extend(f"  {_cut(e)}" for e in errors[:MAX_ERRORS])
        if len(errors) > MAX_ERRORS:
            out.append(f"  ... and {len(errors) - MAX_ERRORS} more")

    tests = nunit_summary(Path(cmd.results_file))
    if tests:
        out.append(f"\n{tests}")

    out.append(
        "\nFiles git reports as changed by this command: "
        + (", ".join(changed) if changed else "none")
    )
    tail = lines[-(TAIL_LINES_OK if outcome.exit_code == 0 else TAIL_LINES) :]
    out.append(f"\nLast {len(tail)} lines of output:")
    out.extend(_cut(line) for line in tail)
    out.append(f"\nFull log: {cmd.log_file}")
    return "\n".join(out)


def nunit_summary(file: Path) -> str | None:
    """Counts and failures from an NUnit 3 results file (Unity, dotnet test)."""
    if not file.is_file():
        return None
    try:
        root = ET.parse(file).getroot()
    except ET.ParseError as exc:
        return f"Test results file could not be read: {exc}"
    if root.tag != "test-run":
        return None
    a = root.attrib
    head = (
        f"Tests: {a.get('total', '?')} run, {a.get('passed', '?')} passed, "
        f"{a.get('failed', '?')} failed, {a.get('skipped', '?')} skipped"
        f" (results file: {file})"
    )
    failed = [c for c in root.iter("test-case") if c.get("result") == "Failed"]
    out = [head]
    for case in failed[:MAX_FAILURES]:
        message = (case.findtext("failure/message") or "").strip().splitlines()
        first = " | ".join(line.strip() for line in message[:5])
        out.append(f"  FAILED {case.get('fullname')}: {_cut(first)}")
    if len(failed) > MAX_FAILURES:
        out.append(f"  ... and {len(failed) - MAX_FAILURES} more failures")
    return "\n".join(out)


def _cut(line: str) -> str:
    return line if len(line) <= MAX_LINE else line[:MAX_LINE] + " ..."
