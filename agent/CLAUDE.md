# News Agent

Python/LangGraph agent that researches AI news and writes it to the SQLite database the Go web
app (`../web/`) serves. Imported from `langchain_simple_demo` @ `112b5ed` with history.

Simplicity first: the agent is built from raw LangGraph primitives (no prebuilt agents) so the
loop is easy to read.

## Stack

- Python 3.14, managed by `uv` (`pyproject.toml` / `uv.lock` here, independent of the Go side).
- Local Ollama: `qwen3.5:latest` for the ReAct loop, `ministral-3:8b` for daily summaries,
  `nimble:latest` (Noul decision model via `typesafe-sdk`) as the relevance judge.
- Runs on a GPU box, **not** the EC2 web host (t3.micro can't run these models).

## Commands

`make news` · `make report` · `make test` (from `agent/`, or `make news` / `make agent-test` from
the repo root). Ad-hoc summary: `PYTHONPATH=src uv run python src/news/agent.py --summarize --day YYYY-MM-DD`.

## Graph

```
START → agent → [should_continue] → tools → agent (loop)
                       ↓ (no tool_calls)
                      END
tools: web search (DuckDuckGo news / Tavily), store_result (SQLite)
```

## Key Files

| Path | Purpose |
|---|---|
| `src/news/agent.py` | Self-contained news agent (raw primitives + SQLite). Relevance judge is a Noul decision model (`nimble:latest`) called via `typesafe-sdk` against Ollama's `/v1/systemone`: one yes/no request per article, delete if probability < `JUDGE_THRESHOLD`. The same pass deletes stale articles (`published_on` older than `MAX_ARTICLE_AGE_DAYS`) with a plain date comparison, logging each as an `article_excluded` metric. **No imports from shared/** |
| `src/news/metrics_report.py` | `make report` — renders `news_metrics.jsonl` to a static self-contained HTML page (stdlib only, hand-written inline SVG; **no plotting deps**). Groups runs into cohorts by the model config logged on their agent steps, and derives its failure breakdown from `error_type` rather than asserting a cause |
| `tests/` | pytest suite — **maintain 100% statement coverage** |
| `docs/` | `news-agent.md` (design, config), `news-agent-model-research.md` (model/context tuning) |

## Database — this side owns the schema

`init_db()` in `src/news/agent.py` creates and migrates the schema. The canonical contract is
`../schema.sql`; `tests/test_schema_contract.py` fails if `init_db()` drifts from it. Any schema
change must update `init_db()`, `schema.sql`, and the web app's queries together, in one PR.
The web app only ever reads (`mode=ro`).

## Environment Variables

Copy `.envrc.example` to `.envrc` (git-ignored at the repo root). Never commit secrets.

| Variable | Default | Required | Description |
|---|---|---|---|
| `OLLAMA_MODEL` | `qwen3.5:latest` | No | Ollama model used for the ReAct loop. `qwen3.5:latest` is the default for its tool-call reliability; gemma models emit malformed tool-call XML. |
| `SUMMARY_MODEL` | `ministral-3:8b` | No | Model used only for `summarize_day`. `ministral-3:8b` (6 GB, 256k native context) holds the largest news days in one pass on a 16 GB GPU; avoid `gemma4:26b` (17 GB), which spills to CPU and returns empty summaries. |
| `SUMMARY_NUM_CTX` | `49152` | No | Context window for the summary model. 48k fits `ministral-3:8b` at ~11 GB fully on a 16 GB GPU; override per-run with `--num-ctx`. |
| `NEWS_NUM_CTX` | `16384` | No | Context window for the news agent's ReAct loop. Left unset, Ollama defaults to 4096, which truncated ~15% of agent steps mid-reasoning — 88% of those emitted no tool call at all. ~8.3 GB total on a 16 GB GPU. Watch `load_duration_ns` in the metrics: sustained spikes mean it is spilling to CPU. |
| `TYPESAFE_BASE_URL` | _(unset)_ | For judge | Where the relevance judge's decision endpoint lives; set to your Ollama server, `http://localhost:11434`. Read by `typesafe-sdk`. |
| `TYPESAFE_API_KEY` | _(unset)_ | For judge | API key for the decision endpoint. Ollama ignores the value; use `ollama`. |
| `TYPESAFE_DEFAULT_MODEL` | `nimble:latest` | No | Noul decision model the news judge uses (`JUDGE_MODEL` in `agent.py`). The judge makes one `/v1/systemone` request per article (≈0.2 s each) and deletes articles whose relevance probability is below `JUDGE_THRESHOLD`; a failed call keeps the article. |
| `JUDGE_THRESHOLD` | `0.5` | No | Relevance cutoff for the judge, 0–1. The model returns a probability that an article is AI/ML-relevant; below this it is deleted. Raise it to prune harder. |
| `NEWS_MAX_ARTICLE_AGE_DAYS` | `90` | No | Recency cutoff (`MAX_ARTICLE_AGE_DAYS`). After judging, articles whose `published_on` is more than this many days before the run are deleted, printed to the console, and logged as `article_excluded` metrics. Code-only date check; a missing or unparseable date keeps the article. |
| `OLLAMA_HOST` | `http://localhost:11434` | No | Ollama server URL. |
| `LANGSMITH_TRACING` | _(unset)_ | No | Set to `true` to enable LangSmith tracing. |
| `LANGSMITH_API_KEY` | _(unset)_ | No | LangSmith API key. Required if tracing is enabled. |
| `LANGSMITH_PROJECT` | _(unset)_ | No | LangSmith project name for grouping runs. |
| `TAVILY_API_KEY` | _(unset)_ | No | If set, the news agent uses Tavily instead of DuckDuckGo. |
| `NEWS_DB_PATH` | `news_data.db` | No | SQLite database path for the news research agent. |
| `NEWS_METRICS_PATH` | `news_metrics.jsonl` | No | JSONL file the news agent appends timing/token metrics to: one record per LLM call (`llm_call`), per pipeline step (`step`), and per tool invocation (`tool_call`). Every record carries the model config it ran under, so `make report` can cohort runs by config. Tool logging roughly doubles the growth rate; there is no rotation yet. |
| `NEWS_FAILURE_DIR` | `news_failures` | No | Directory where a failed agent step dumps the conversation that produced it. Ollama discards the malformed completion server-side, so the inputs are the only recoverable evidence — dumps are written in `messages_to_dict` form and replay through `messages_from_dict`. |
| `NEWS_FAILURE_KEEP` | `50` | No | How many failure dumps to retain. Older ones are pruned on each new dump. |
| `NEWS_INTERVAL` | `0` | No | Polling interval in seconds for the news agent. `0` runs once. |
