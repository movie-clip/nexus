# nexus

An engineering assistant built on LangGraph and Deep Agents. You send it a
request in plain text about a project; agents explore the project and answer.
You start every run by hand. Background: `research/langgraph-standalone-app.md`.
What "done" means: `DONE.md`.

## Ask

```sh
# once: put your key in nexus's .env (git-ignored)
echo ANTHROPIC_API_KEY=sk-ant-... > .env

uv run python -m nexus "how does the character controller handle jumping?" --project C:\projects\impact
```

The answer is printed as Markdown; progress, the thread id and the run's cost
in US dollars go to stderr. `--json` prints the full result instead.

By default a **team** answers (`pipelines/team.py`):

```
request -> orchestrator -+- scout -----------------------------------+
              ^          +- reviewer --------------------------------+
              |          +- engineer -> (pause: you approve) -> verify +-> report back
              |          +- tester -> (pause unless "auto") -> run --+
              +----------+- finish -> answer
```

- The **orchestrator** (no file access) reads the request and every
  report so far, and each round picks: send the scout or the reviewer a
  self-contained brief, or finish with the answer. Its choice is JSON enforced
  by the API's structured output; LangGraph routes on it in plain code, and
  each decision and its reason is logged.
- The **scout** finds what is actually in the repo; the **reviewer** checks a
  draft, claims, or an applied change against the code. Each sees only its brief.
- The **engineer** changes the project, but only through staging: its edits
  never touch the project directly. When it is done the run **pauses** (exit
  code 4) and prints the whole batch as one diff. Nothing is written until you
  decide:

  ```sh
  uv run python -m nexus.resume --thread <id> --decision approve
  uv run python -m nexus.resume --thread <id> --decision reject --note "what to change"
  ```

  Approve writes the batch, skipping any file that changed on disk since it was
  staged. Reject writes nothing and passes your note to the orchestrator.
  Line endings are kept line by line, and the diff warns if a change would
  alter endings it can't show. The engineer cannot write inside `.git`, touch
  `.env` files, run commands, or commit. It writes tests only when the
  request asks for them.
- **Verify**: after you approve a batch, the project's `verify` commands (for
  impact: `compile`) run at once, without asking, in order, stopping at the
  first failure. The orchestrator reads their reports and can send the
  engineer to fix a failure. They don't use up rounds.
- The **tester** runs a build or test command and reports the result. It is
  not a model: the orchestrator picks a command by name from the project's
  config and fills in its params; see [Commands](#commands). Tests are run
  only when the request asks for them (write tests at milestones, e.g.
  "write EditMode tests for the crate pipeline and run them").
- Every run is saved in `runs/nexus.sqlite` (git-ignored), so a paused or
  failed run continues later from a new terminal with `nexus.resume`.
- Spending is capped three ways:
  - at most `--max-rounds` specialist and tester rounds (default 6);
  - at most `--budget` US dollars per run (default 2), checked before each
    round, so a run can only go over by the round that crossed it;
  - each specialist gets a number of model calls (scout and reviewer 30,
    engineer 60, `--pipeline ask` 30). On the last one it can't use tools and
    must write its report, which then says it was cut short.

  When rounds or budget run out, the only choice offered is "finish".
- Every model call is logged with its cost as it ends, e.g.
  `model call 7 (scout): $0.0312, run so far $0.4123`. A failed run also shows
  what the failed step spent, which isn't part of the run's recorded cost.
- `--pipeline ask` uses a single agent instead (`pipelines/ask.py`).

The scout, the reviewer and `--pipeline ask` are **read-only**: they can list,
read, glob and grep inside `--project` and nothing else. Write tools are not offered, writes are denied
by permission rules anyway, paths outside the project are blocked, `.env`
files are unreadable, and there is no shell. The project's own `AGENTS.md` /
`CLAUDE.md` are loaded when present.

LangGraph and Deep Agents are free and open source; the Claude API calls are
billed per token to your key. Install [ripgrep](https://github.com/BurntSushi/ripgrep)
(`winget install BurntSushi.ripgrep.MSVC`) so searches are fast and skip
`.gitignore`d folders.

The layout follows LangGraph's [application structure](https://docs.langchain.com/oss/python/langgraph/application-structure):
each pipeline is a module exporting a compiled `graph`, listed in `langgraph.json`.
The same graphs work in two ways:

| | Command | Use it for |
|---|---|---|
| **Ask** | `uv run python -m nexus "<request>" --project <path>` | a request in plain text (team by default) |
| **Resume** | `uv run python -m nexus.resume --thread <id> [--decision approve\|reject]` | answer a paused run, or retry a failed one |
| **Run** | `uv run python -m nexus.run --pipeline <name>` | any pipeline, inputs via `--set`, result as JSON |
| **Inspect** | `uv run langgraph dev` | developing: local API and Studio UI to watch and step through runs (in-memory, dev only) |

## Commands

The tester can only run commands you list for a project, in nexus (not in the
project): `projects/<name>.toml`, matched by its `path`. `projects/impact.toml`
is the example: `compile` (`dotnet build`) and `unity-tests` (Unity in
batchmode, only when a request asks for tests), both run without asking.

```toml
path = "C:/projects/impact"
verify = ["compile"]  # run after every approved batch, without asking

[commands.unity-tests]
description = "What it checks and when to use it (the orchestrator reads this)."
run = ["powershell", "-File", "Tools/Run-UnityTests.ps1", "-ResultsFile", "{results_file}",
       ["-Filter", "{filter}"]]
cwd = "Impact"
timeout = 1800
approval = "ask"    # or "auto": run without pausing

[commands.unity-tests.params]
filter = { pattern = "[A-Za-z_][A-Za-z0-9_.]*", description = "..." }
```

- A command is an argument list, never a shell line, and `.bat`/`.cmd`
  programs are refused (Windows would re-parse them as a shell line). Each
  param value must fully match its `pattern`, or the command is refused before
  it is shown to you. A nested list is left out unless its params have values.
- `approval = "ask"` pauses the run and prints the exact command, folder and
  time limit; answer with `nexus.resume --decision approve|reject`.
- `verify` lists commands to run after every approved batch. Listing one is
  your approval, so it runs without asking even if set to `"ask"`. Each must
  be runnable without values from a model: all its required params have
  defaults.
- The command runs without nexus's secrets in its environment, with no input,
  and its whole process tree is killed at the time limit. A command's failure
  is its result: it is never retried. If nexus itself dies while a command
  runs, `nexus.resume` runs the approved command again.
- The full output goes to `runs/logs/<thread>/<n>-<command>.log`. The report
  carries the result, unique compiler errors (`error CS0103: ...`), failed
  tests from an NUnit results file (`{results_file}`, written outside the
  project), the files `git status` shows the command changed, and the end of
  the output (40 lines after a failure, 10 after a success).
- A command given `{results_file}` must write it. If it exits 0 but wrote no
  results file, an unreadable one, or one with 0 tests run, the report says
  FAILED and why: the tests didn't run or didn't finish (a launcher that
  returns early), or the filter matched nothing (Unity exits 0 for that too).

## Tracing

Runs can be traced to a Langfuse that runs on this PC in Docker: every
orchestrator decision, every specialist, each model call and tool call, with
its input and output, tokens and time (a run's dollar cost is what nexus
prints). Setting it up, starting and stopping it:
`C:\projects\tools\langfuse\README.md`.

- Tracing is on when `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and
  `LANGFUSE_BASE_URL` are in nexus's `.env`; comment them out to turn it off.
  Only `nexus` and `nexus.run`/`nexus.resume` runs are traced, not runs
  started in `langgraph dev`.
- Each start or resume is one trace, named after the pipeline (`team`,
  `team (resume)`). A run and its resumes are grouped under **Sessions** by
  thread id.
- nexus's secrets (anything in its `.env` named like a key, secret, token or
  password, and any `ANTHROPIC_*` variable) are replaced with `[secret]`
  before a trace leaves the process. If Langfuse is down, runs work the same,
  just without traces.

## Checks

Every change to nexus must pass one command, which runs lint, formatting,
type checks (mypy) and the tests with a coverage floor (98%), and stops at the
first failure:

```sh
uv run python scripts/check.py
```

## Run any pipeline

```sh
uv sync
uv run python -m nexus.run --pipeline smoke --set scenario=flaky
uv run python -m nexus.run --pipeline smoke --set fail_node=step_b --set fail_times=1
```

`--set key=value` sets a run input. Unknown keys and wrong types are refused
before anything runs. Progress goes to stderr: one line per finished node or
retry, plus the traceback once if the run fails. The result (`thread_id`,
`status`, `state`, and `failed_node`/`error` on failure) goes to stdout as JSON.

| Exit | Meaning |
|---|---|
| 0 | completed |
| 1 | completed, but the run recorded errors in state |
| 2 | bad arguments or inputs; nothing ran |
| 3 | failed: a node raised and its retries ran out; `nexus.resume` retries the failed step |
| 4 | waiting: paused for your decision; continue with `nexus.resume` |

## Layout

```
langgraph.json    graph name -> ./src/nexus/pipelines/<module>.py:graph
src/nexus/
  state.py        BaseState: errors, cost_usd, artifacts, retries, verdict
  retry.py        DEFAULT_RETRY (transient errors) and retry_update (capped "try again" routes)
  models.py       Claude model defaults (effort, refusal fallbacks) and cost_usd from token usage
  agents.py       shared Deep Agent pieces: read_only_agent, staged_engineer, permissions, run_agent
  staging.py      StagedBackend (the engineer's writes go here) and apply_changes
  commands.py     the tester's commands: per-project config, checks, running, reports
  resume.py       `python -m nexus.resume`
  __main__.py     `python -m nexus "<request>"`
  run.py          the CLI: loads a graph from langgraph.json, validates inputs, runs, reports
  tracing.py      optional Langfuse tracing, with nexus's secrets masked
  errors.py       InputError
  pipelines/
    team.py       orchestrator + scout + reviewer + engineer + tester (default for `python -m nexus`)
    ask.py        a single read-only Deep Agent
    smoke.py      the engine's own test pipeline: routing scenarios and failure drills
projects/         per-project commands the tester may run (<name>.toml)
scripts/check.py  every check nexus must pass
tests/
```

## Adding a pipeline

1. Create `src/nexus/pipelines/<name>.py`.
2. Define `<Name>Input(TypedDict)` with the run inputs, and
   `<Name>State(BaseState, <Name>Input)` with the working fields.
3. Build a plain `StateGraph(<Name>State, input_schema=<Name>Input)`, add each
   node with `retry_policy=DEFAULT_RETRY`, and export `graph = build().compile(name="<name>")`
   **without** a checkpointer. `langgraph dev` and `nexus.run` each add their own.
4. Add `"<name>": "./src/nexus/pipelines/<name>.py:graph"` to `langgraph.json`.

Rules:
- Use plain code for deterministic steps, and LLM nodes only for judgment or wording.
- Transient failures (timeouts, 5xx) are retried by `DEFAULT_RETRY`: 3 attempts,
  about 1s then 2s apart. Programming errors (`ValueError`, `KeyError`, ...) are
  not retried. Model "try again" decisions as a route capped with `retry_update`.
- Keep state small: store large data as files and keep only their paths in
  `artifacts`. Never put secrets in state.
- Put secrets in `.env` (git-ignored).
