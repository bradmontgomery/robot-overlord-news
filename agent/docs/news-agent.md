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
- Each `DEFAULT_TOPICS` entry (a comma-separated group of related labels) is searched in its
  own independent run. With `qwen3.5:latest`, a run covering more than ~3 topics fills the
  context with search results and the model stops calling `store_result`; large combined
  queries also crash the news backend. A flaky run (e.g. intermittent malformed tool-call
  XML) is logged and skipped rather than aborting the whole job. See the
  [README](../README.md#one-topic-entry-per-agent-run) for the full rationale.
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
final step, then renders that briefing to the terminal (as a "Daily Briefing"
markdown block) after the article-count line. To (re)generate that summary for a
single day on demand — without running the research pipeline:

```bash
PYTHONPATH=src uv run python src/news/agent.py --summarize --day 2026-06-21
PYTHONPATH=src uv run python src/news/agent.py --summarize   # defaults to today

# Experiment with a different model / context window for this run only:
PYTHONPATH=src uv run python src/news/agent.py --summarize --day 2026-06-21 \
  --model qwen3.5:latest --num-ctx 24576
```

The summary is upserted (re-running overwrites the existing row for that date) and
printed to stdout. `--day` selects articles by the date they were **collected**
(`fetched_at`), not their publication date, and defaults to today in **UTC** to match
how `fetched_at` is stored. The date must be zero-padded `YYYY-MM-DD`.

The summary model defaults to `SUMMARY_MODEL` (`ministral-3:8b`) with context window
`SUMMARY_NUM_CTX` (`49152`). `ministral-3:8b` (6 GB, 256k native context) holds 48k
tokens at ~11 GB fully on a 16 GB GPU — enough for even the largest news days in one
pass. `--model` and `--num-ctx` override these for a single run without touching the
environment — handy for experimenting. If a day ever exceeds the context window it is
truncated; raise `--num-ctx` only while it still fits on the GPU (check `ollama ps`
reads `100% GPU`).

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

## Model choices

Evaluations of candidate models for the agent's three LLM workloads — and why the
current ones were kept — live in [news-agent-model-research.md](news-agent-model-research.md).
