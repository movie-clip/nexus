# LangGraph: research notes and a plan for a standalone app

Researched 2026-10-07. Prices, billing policy and benchmarks change often,
so re-check anything marked *(verify)* before you rely on it.

## 1. Verdict

| Option | Verdict | Why |
|---|---|---|
| Replace `.agentic` with LangGraph | **No** | It would mean rebuilding Claude Code's harness (tools, permissions, hooks, MCP, UI) and paying API prices for a result you already have |
| A Python script, run by an agent, that runs LangGraph | **No** | Two orchestrators; nodes become subprocess sessions or bare API calls; human gates are awkward; slower; more to maintain |
| Script-held orchestration inside Claude Code | **Later, only on evidence** | Use Claude Code **dynamic workflows** (native), not LangGraph. Wait until the orchestrator eval set (ROADMAP 0a / TD-18) shows repeated misrouting |
| **A separate, standalone LangGraph app** | **Yes, for the right job** | Good fit for unattended, scheduled, non-coding pipelines over investment data. Start with one pipeline, not a copy of the network |
| LangSmith tracing of Claude Code | Optional | No code changes needed (a plugin or environment variables). Try it only if the evals can't explain run cost |

## 2. The tools

### LangGraph
- A low-level Python/JS library from LangChain Inc for long-running, stateful
  agents. MIT licence, ~43k GitHub stars, **v1.0 stable**.
- You write a typed **StateGraph**:
  - nodes are functions (LLM calls, tools, plain code);
  - edges, including conditional ones, route between them;
  - `START` and `END` mark the entry and exit.
- Headline features:
  - **Durable execution:** a checkpointer saves state after each step, so a
    run resumes after a crash or restart.
  - **Human-in-the-loop:** `interrupt()` pauses a node, and
    `Command(resume=value)` continues it, even from a different process
    later, given a persistent checkpointer (e.g. SQLite/Postgres) and the
    same `thread_id`.
  - **Memory:** short-term (thread state) and long-term (a store across
    threads).
  - Streaming.
- **Gotcha:** on resume, the interrupted node **re-runs from its start**.
  Keep side effects (LLM calls, writes) out of nodes that interrupt, or make
  them idempotent.
- Higher layers built on it:
  - **LangChain agents:** prebuilt tool-calling loops.
  - **Deep Agents:** a Claude Code-style harness with a todo planner,
    subagents, filesystem tools (`read_file`, `write_file`, `edit_file`,
    `ls`, `glob`, `grep`) and context management. Model-agnostic and MIT.
    It has a **Deep Agents CLI** coding agent. On Terminal-Bench 2.0 with
    Sonnet 4.5, LangChain measured it at roughly Claude Code's level;
    third-party comparisons vary by the model they used *(verify)*.
- Production users named by LangChain: Klarna (support assistant), LinkedIn
  (recruiter agent), Uber (developer-productivity agents), Replit (coding
  agent).
- Criticism, mainly from the yeos.ai write-up:
  - debugging an "execution graph" instead of a function;
  - typed state turning into a maintenance burden;
  - orchestration overhead;
  - most real flows being sequential anyway, so the graph adds little;
  - a learning curve (state, reducers, edges, checkpointers).

### LangSmith
- LangChain's tracing and evaluation SaaS. It works without LangGraph.
- **Tracing:** every run, node, LLM call and tool call, with tokens and
  latency.
- **Evals:** datasets of examples with optional reference outputs. Evaluator
  kinds:
  - code checks;
  - LLM-as-judge;
  - pairwise comparison of two versions;
  - human annotation queues.
  It runs offline experiments (compare versions) and online monitoring, and
  can score an agent's trajectory (right tools, right arguments, right
  steps).
- **Claude Code integration:** the `langsmith-tracing` plugin, or the
  environment variables `TRACE_TO_LANGSMITH`, `CC_LANGSMITH_API_KEY` and
  `CC_LANGSMITH_PROJECT`.
  - It traces messages, tool calls, subagents, compaction, tokens and timing.
    System prompts are not traced, and an interrupted subagent is lost.
  - Secret redaction runs locally but only catches known credential shapes.
    Cloud only.
- **Pricing** *(verify)*:
  - Developer: free, 1 seat, 5k traces/month.
  - Plus: $39/seat/month plus per-trace fees (~$2.50 per 1k base traces).
- **Alternatives:** Claude Code's built-in OpenTelemetry export (any OTLP
  backend), or Langfuse (self-hostable).

### Claude Code dynamic workflows (the native alternative to "script orchestration")
- A JavaScript script that orchestrates subagents: "who decides what runs
  next: the script".
- Primitives:
  - `agent()` starts one subagent;
  - `pipeline()` and `parallel()` fan agents out;
  - `phase()` groups agents in the progress view;
  - `log()` shows a message in the progress view;
  - the `args` global carries input;
  - a per-agent JSON `schema` returns validated structured output.
- **Where scripts live:** saved in `.claude/workflows/`, or shipped in a
  plugin (`workflows/` directory, run as `/<plugin>:<name>`).
- **What it uses:** your subscription, permission rules, agents and shared
  prompt cache. You watch it in `/workflows`, with per-agent tokens.
- **Resume:** within the session, or after `claude --resume`.
- **Limits:**
  - **no mid-run user input** (run one workflow per stretch between human
    gates);
  - no shell or filesystem access from the script itself;
  - 16 concurrent agents by default;
  - 1,000 agents per run.

## 3. Why not inside `.agentic`

1. **Billing and tools.** A node can call Claude only through the API (API
   key, pay per token, no Claude Code tools) or through `claude -p` / the
   Agent SDK.
   - `--bare` is recommended for scripts, but it **needs
     `ANTHROPIC_API_KEY`**.
   - Without `--bare`, today it draws on your subscription limits. Anthropic
     paused a change to that on 2026-06-15 and promised notice before any
     change takes effect *(verify)*.
2. **Two orchestrators.** Lanes, phases and the round rule would live in both
   the plugin and the Python graph and drift apart.
3. **Human gates.** A `-p` session can't ask you anything, so the script has
   to exit at the interrupt, the outer agent asks you, then the script
   resumes.
4. **Visibility and speed.**
   - Foreground Bash times out after 10 minutes, so runs go to the
     background and the outer agent can't see them.
   - Each node pays Claude Code's startup cost and gets no shared prompt
     cache.
5. **Lab guidance.** Anthropic's *Building Effective Agents* says frameworks
   "often create extra layers of abstraction that can obscure the underlying
   prompts and responses, making them harder to debug"; start simple. This
   also matches the network's "no new mechanisms without proof" rule.

## 4. When a standalone LangGraph app fits

**Fits:**
- Unattended or scheduled runs where nobody is in a chat.
- Durable waits for approval: hours or days, then resume.
- Retries and resume after failures.
- Mixing models from different providers per node.
- Mostly data and analysis work, with little open-ended code editing.

**Doesn't fit:**
- Editing code in your repos. Claude Code / `.agentic` does that better at
  no extra cost.
- Flows that are really one linear prompt chain. Plain Python is enough.

### Candidate pipelines (investments workspace)

| Pipeline | Shape | Human stop |
|---|---|---|
| Portfolio health check (daily/weekly) | fetch positions → compute metrics → independent quant re-check → threshold rules → report | alert or approve only when a threshold breaks |
| TradingView signal scan | fetch data → run indicator logic → LLM summary per symbol → rank → digest | none, or approve a watchlist change |
| Budget categorisation | import transactions → rule pass → LLM classifies the leftovers → confidence gate → write | review low-confidence items |

Pick **one** for the pilot. The portfolio check has the clearest value and an
easy way to judge it: the numbers either reconcile or they don't.

## 5. Reference architecture (pilot)

```
scheduler (cron / Task Scheduler / GitHub Actions)
   └─ python -m app.run --pipeline portfolio_check --thread <date>
        StateGraph[PipelineState]
          fetch ─▶ compute ─▶ verify ─┬─(ok)──────▶ report ─▶ END
                                      └─(breach)──▶ approve (interrupt) ─▶ report
        checkpointer: SQLite (one thread_id per run)
        tracing: LangSmith (free tier) or OTel/Langfuse
```

- **State:** one typed object (TypedDict or Pydantic) holding inputs, the
  computed numbers, verification verdicts, report text, `cost_usd` and
  `errors`. Keep it small, and store large blobs as files with only their
  paths in state.
- **Nodes:** prefer plain code for anything deterministic (fetching, maths,
  thresholds). Use LLM nodes only for judgment or wording. Use one model per
  node: a cheap model for extraction and summaries, a strong one for
  verification and judgment.
- **Edges:** conditional edges return a route name from the state, e.g.
  verify → `ok` / `breach` / `retry`. Cap retries in state, the same idea as
  the network's two-round rule.
- **Human gate:** `interrupt()` in its own node with no side effects. Notify
  through email or Telegram, and resume through a small CLI:
  `app.resume --thread X --decision approve`.
- **Durability:** SQLite checkpointer for the pilot; Postgres only if it ever
  runs on a server with concurrent runs.
- **Errors:** retries with backoff on API nodes; a failed node leaves the
  checkpoint so `resume` continues from it.
- **Secrets:** keep API keys in environment variables or a secrets store,
  never in state, because state gets traced. Redact account numbers before
  tracing.
- **Tests:**
  - unit-test deterministic nodes like normal functions;
  - test the graph's routing with stubbed LLM nodes;
  - run a small eval set (10–20 cases) in LangSmith for the LLM nodes.

### Model and cost sketch *(verify prices)*

API prices per million input/output tokens: Opus 5.5 $4/$20, Sonnet 5.5
$2/$10, Haiku 4.5 $1/$5. Prompt caching cuts the cost of repeated prefixes a
lot. A pilot pipeline that is mostly plain code, with 2–3 short LLM nodes,
should cost cents per run. A 5M-input / 0.5M-output Opus run would be about
$30, which is why the coding network stays on the subscription.

## 6. Pilot plan and success criteria

1. **Define done before building:** cost per run, how often a human has to
   step in, and whether the output matches a manual check.
2. **Build:** the graph with 4–6 nodes, a SQLite checkpointer, a resume CLI,
   tracing, and tests for the deterministic nodes and routing.
3. **Shadow run:** 2–4 weeks next to your current manual process, logging
   every disagreement.
4. **Decide:** expand to a second pipeline, simplify to plain Python (if the
   graph added nothing), or stop.

## 7. Open questions for the next round of research

- [ ] Current LangGraph version and API details: `interrupt`, `Command`, the
      streaming API version, the SQLite checkpointer package name.
- [ ] Is LangGraph actually needed, or does plain Python plus a small state
      file do the pilot? Check the yeos.ai argument against the chosen
      pipeline's shape.
- [ ] LangSmith vs Langfuse (self-hosted) vs OTel, given financial data
      privacy.
- [ ] Data sources and APIs for the chosen pipeline (broker export,
      TradingView data access and its terms).
- [ ] Where it runs: local Task Scheduler vs a small VPS vs GitHub Actions,
      and the notification channel for human gates.
- [ ] Agent SDK billing status, if any node should use Claude Code tools
      instead of the plain API.
- [ ] LangGraph Platform / deployment offering: needed or not for one user.

## 8. Sources

- LangGraph repo: https://github.com/langchain-ai/langgraph
- LangGraph overview: https://docs.langchain.com/oss/python/langgraph/overview
- LangGraph interrupts: https://docs.langchain.com/oss/python/langgraph/interrupts
- LangGraph 1.0: https://changelog.langchain.com/announcements/ann_lxOZRCoFw2Qn9
- Production users: https://blog.langchain.com/is-langgraph-used-in-production
- Deep Agents: https://docs.langchain.com/labs/deep-agents/overview
- Deep Agents CLI on Terminal-Bench 2.0: https://www.blog.langchain.com/evaluating-deepagents-cli-on-terminal-bench-2-0
- Claude Code vs Deep Agents CLI: https://terminaltrove.com/compare/ai-coding-agents/claude-code-vs-deep-agents-cli/
- Why we removed LangGraph (yeos.ai): https://dev.to/nicolaric/why-we-removed-langgraph-from-our-ai-platform-i9o
- Why I stopped using LangGraph: https://dev.to/deadlocker/why-i-stopped-using-langgraph-4jo2
- LangGraph + Claude Agent SDK nodes: https://www.khaledelfakharany.com/articles/langgraph-claude-sdk-integration
- Claude Agent SDK vs LangGraph: https://www.developersdigest.tech/blog/claude-agent-sdk-vs-langgraph
- LangSmith, trace Claude Code: https://docs.langchain.com/langsmith/trace-claude-code
- LangSmith evaluation concepts: https://docs.langchain.com/langsmith/evaluation-concepts
- LangSmith pricing overview: https://www.costbench.com/software/ai-observability/langsmith/
- Claude Code dynamic workflows: https://code.claude.com/docs/en/workflows
- Claude Code headless / `--bare`: https://code.claude.com/docs/en/headless
- Claude Code best practices: https://code.claude.com/docs/en/best-practices
- Claude Code monitoring (OTel): https://code.claude.com/docs/en/monitoring-usage
- Agent SDK with your Claude plan: https://support.claude.com/en/articles/15036540-use-the-claude-agent-sdk-with-your-claude-plan
- Claude API pricing: https://platform.claude.com/docs/en/about-claude/pricing
- Anthropic, Building Effective Agents: https://www.anthropic.com/engineering/building-effective-agents
