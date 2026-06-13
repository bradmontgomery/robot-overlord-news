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
- `store_result` uses a module-level `DB_PATH` (set from `NEWS_DB_PATH` env var) so it's
  patchable in tests without monkey-patching sqlite3 itself.
- `should_continue` is a module-level function (not a closure inside `build_graph`) so it can
  be unit-tested directly.

## Run manually

```bash
make news                                                  # defaults: AI, climate
PYTHONPATH=src uv run python src/news/agent.py "rust lang" "webassembly"
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `NEWS_DB_PATH` | `news_results.db` | Path to the SQLite database for storing results. |

Search tool follows the same Tavily / DuckDuckGo logic as the other agents: set
`TAVILY_API_KEY` to use Tavily; otherwise falls back to DuckDuckGo's news backend.
