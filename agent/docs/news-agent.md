# News Research Agent

`src/news/agent.py` demonstrates building a ReAct-style agent loop from raw LangGraph
primitives — no `create_react_agent`, no `create_supervisor`. The graph is wired by hand:

```
START → agent → [should_continue] → tools → agent (loop)
                       ↓ (no tool_calls)
                      END
```

For each topic you provide, the agent searches for recent news and calls `store_result`
once per article. Results are written to a local SQLite database (`news_results.db` by default).

## Key design choices

- `DuckDuckGoSearchResults(backend="news", output_format="list")` returns per-article structured
  data (not a single string blob), making it easier for the model to call `store_result` once
  per item.
- Today's date is injected into the system prompt so the model can reason about recency.
- `recursion_limit=50` guards against infinite tool-call loops.
- Topics are searched in small batches (`BATCH_SIZE = 2`). With `qwen3.5:latest`,
  a batch larger than ~3 topics fills the context with search results and the
  model stops calling `store_result` (drops to zero stored at 4+); large combined
  queries also crash the news backend. Each batch is an independent run, so a
  flaky batch (e.g. intermittent malformed tool-call XML) is logged and skipped
  rather than aborting the whole run. See the [README](../README.md#topic-batching-and-model-reliability-batch_size) for the full rationale.
- `store_result` uses a module-level `DB_PATH` (set from `NEWS_DB_PATH` env var) so it's
  patchable in tests without monkey-patching sqlite3 itself.
- `should_continue` is a module-level function (not a closure inside `build_graph`) so it can
  be unit-tested directly.

## Run manually

```bash
make news                                                  # defaults: ~65 AI/LLM topics
PYTHONPATH=src uv run python src/news/agent.py "rust lang" "webassembly"
```

## Ad-hoc day summary

A normal run summarizes the day's articles into the `news_summaries` table as its
final step. To (re)generate that summary for a single day on demand — without
running the research pipeline:

```bash
PYTHONPATH=src uv run python src/news/agent.py --summarize --day 2026-06-21
PYTHONPATH=src uv run python src/news/agent.py --summarize   # defaults to today
```

The summary is upserted (re-running overwrites the existing row for that date) and
printed to stdout. `--day` selects articles by the date they were **collected**
(`fetched_at`), not their publication date, and defaults to today in **UTC** to match
how `fetched_at` is stored. The date must be zero-padded `YYYY-MM-DD`.

## Polling mode

Set `NEWS_INTERVAL` to a number of seconds to run on a fixed interval instead of once:

```bash
NEWS_INTERVAL=300 make news   # re-run every 5 minutes
```

The loop runs → prints → sleeps → repeats until interrupted (`Ctrl-C`). Unset or `0` runs once and exits.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `NEWS_DB_PATH` | `news_results.db` | Path to the SQLite database for storing results. |
| `NEWS_INTERVAL` | `0` | Polling interval in seconds. `0` (or unset) runs once and exits. |

Search tool follows the same Tavily / DuckDuckGo logic as the other agents: set
`TAVILY_API_KEY` to use Tavily; otherwise falls back to DuckDuckGo's news backend.
