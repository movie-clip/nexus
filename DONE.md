# Definition of done

Agreed before building (see `research/langgraph-standalone-app.md` §6).
Nexus is an engine first, so the criteria are what the engine promises. Each
promise is proven by a test, not by a demo.

## Engine promises

| Promise | Proven by | Slice |
|---|---|---|
| Conditional routing goes where the state says | routing tests on the `smoke` pipeline | 1 |
| Retries are capped per node in state, and a run that hits the cap ends with the error recorded | `stuck` scenario test, per-node retry test | 1 |
| Bad inputs (unknown key, wrong type, engine field) are refused before the run starts | input validation tests | 1 |
| Unattended runs log progress per node, key names only | progress log test | 1 |
| Pipelines follow LangGraph's application structure: every graph in `langgraph.json` loads in `nexus.run` and in `langgraph dev` | loading test; manual `langgraph dev` check | 2 |
| A node that raises a transient error is retried with backoff, one log line per retry; programming errors are not retried | failure drills via the smoke pipeline's `fail_*` inputs | 2 |
| A failed run returns `status=failed`, the failed node and the error as JSON, keeps the work finished before the failure, logs the traceback once, and exits 3 | failure-report tests, subprocess test | 2 |
| `python -m nexus "<request>" --project P` answers about P, with cost in US dollars | ask pipeline test (scripted model); first live run | A |
| The ask agent cannot change the project: no write tools offered, writes denied, traversal blocked, `.env` unreadable, no shell | agent permission tests (exact error per case) | A |
| Cost counts every model call, including subagents and cache reads and writes; unknown models are never counted as free | cost tests | A |
| The orchestrator routes each round to scout, reviewer or finish, from structured output (no forced tool use) | team routing tests (scripted orchestrator) | B |
| Specialists see only their brief; the orchestrator sees every brief and report | context isolation tests | B |
| The round cap holds: with no rounds left, only "finish" is offered (verified against the live API) | round-cap tests; live schema call | B |
| On the same question the team matches the single agent's answer quality at lower cost (live: $0.22 vs $0.34, both correct) | live comparison, 2026-10-07 | B |
| A failed run resumes from the failed step; finished steps are not run again | resume test counting node calls | C |
| `run` refuses a thread id that already exists; only `resume` continues one | existing-thread test | C |
| The engineer's edits are staged, never written directly; the run pauses with one diff for the whole batch | engineer flow tests; live run on impact (project untouched while paused) | C |
| Approve applies the batch from a new process; reject writes nothing and passes the note on | approve/reject tests over a fresh SQLite connection; live reject | C |
| A file changed on disk while waiting is not overwritten | conflict test | C |
| Line endings are kept line by line, and the diff warns about ending changes it can't show | mixed-ending tests incl. a replay of a real impact file; live re-stage | C |
| The engineer cannot write `.git` or `.env`, read `.env`, escape the project, or run commands | engineer limit tests (exact errors) | C |
| The tester runs only commands listed in the project's config; values that don't fully match their pattern, unknown commands and params, batch files and missing programs are refused before anything is shown or run | command config and prepare tests; pipeline refusal tests | D |
| An "ask" command pauses with the exact command line and runs only on approve; "auto" runs at once; a rejection runs nothing and doesn't use a round | approval flow tests over a fresh SQLite connection | D |
| A command can't outlive its time limit (the whole process tree is killed), never sees nexus's secrets, and reports the files it changed | timeout test with a grandchild process; env test; git check test | D |
| The tester's report has the result, unique compiler errors, failed tests from NUnit XML, changed files and the output tail; the full log is kept | summary tests; live compile of impact | D |
| Every run reports its LLM cost in `cost_usd` | LLM-node test with a fake model | A |
| Without Langfuse keys nothing is traced; with them, a run and every step inside it (orchestrator, specialists, model and tool calls) is one trace, and a run and its resumes share a session | tracing tests with an in-memory exporter; first live trace, 2026-10-07 | E |
| nexus's secrets (its `.env` keys and tokens, any `ANTHROPIC_*`) are masked before a span leaves the process | masking test with a secret in a tool result; live trace searched for every secret | E |
| A Langfuse that is down or unreachable never fails or stalls a run | unreachable-endpoint test | E |
| Deep Agents' own files (oversized messages and tool results, saved history) never reach the project or the engineer's batch: they are kept in memory, where the agent can still read them | offload tests with a 210k-character brief (failed before the fix) | pre-F |
| A malformed approval answer asks the question again; it never fails the run or gets replayed | malformed-answer tests for batches and commands | pre-F |

## Pilot metrics (once a real pipeline exists)

- **Cost per run:** stays at cents per run for a mostly plain-code pipeline.
- **Human interventions:** share of runs that stop at a gate, plus how long they wait.
- **Correctness:** the output matches a known expected result or a manual check.
  Every disagreement is logged during the shadow run.

## Decision point

After slice F (the first real task on impact), check whether the graph earns
its keep. If the flows are linear
with one pause, plain Python plus a state file would do; switch then.
