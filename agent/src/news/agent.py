"""
news/agent.py — self-contained LangGraph news research agent.

Builds a ReAct-style loop from raw LangGraph primitives:
  StateGraph + MessagesState + ToolNode + hand-written conditional routing.

No create_react_agent, no create_supervisor.

Graph:
  START → agent → [should_continue] → tools → agent (loop)
                         ↓ (no tool_calls)
                        END
"""

import argparse
import contextvars
import json
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, messages_to_dict
from langchain_core.tools import StructuredTool, tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from rich import print as rprint
from rich.markdown import Markdown

MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen3.5:latest")
SUMMARY_MODEL_NAME = os.getenv("SUMMARY_MODEL", "ministral-3:8b")
# Context window for the summary model. ministral-3:8b (6 GB, 256k native context)
# holds 48k tokens at ~11 GB fully on a 16 GB GPU — enough for even the largest news
# days in one pass. Override per-run with --num-ctx or the SUMMARY_NUM_CTX env.
SUMMARY_NUM_CTX = int(os.getenv("SUMMARY_NUM_CTX", "49152"))
# Context window for the ReAct loop. Left unset, Ollama defaults to 4096 — and the
# metrics show what that costs: 15% of agent_step calls stopped at exactly 4096
# tokens, and 88% of those emitted zero tool calls, so the whole loop iteration was
# wasted mid-reasoning. Prompts run to ~4.1k and complete responses to ~1.8k, so the
# real floor is ~6k; 16k leaves room for conversations that grow once they stop
# being clipped. Costs ~1.7 GB of KV cache on top of qwen3.5's 6.6 GB — roughly
# 8.3 GB of a 16 GB card. If load_duration_ns starts spiking in the metrics, this is
# too large for the GPU and is spilling to CPU.
AGENT_NUM_CTX = int(os.getenv("NEWS_NUM_CTX", "16384"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DB_PATH = os.getenv("NEWS_DB_PATH", "news_results.db")
METRICS_PATH = os.getenv("NEWS_METRICS_PATH", "news_metrics.jsonl")
# Topics are searched in small batches. A single huge query crashes the news
# backend ("Could not extract vqd"); just as important, once a batch fills the
# context with many search results the local model stops following through on
# the store_result step (empirically reliable at 2-3 topics, drops to zero
# storing at 4+). Keep this small so every topic's articles actually get stored.
BATCH_SIZE = 2
# Articles judged per LLM call in judge_articles(). 10 articles' title+summary is
# ~500-700 tokens, comfortably inside the model's default 4096-token context; cuts
# judge_articles' call count ~10x versus one call per article.
JUDGE_BATCH_SIZE = 10
# Hard cap on judge output. The judge is not context-starved — its prompts peak at
# ~1.2k tokens, well inside the 4096 default — but its output is bimodal: 41 tokens
# for a normal verdict, or a runaway that burns every remaining token and stops at
# the limit with empty .content, which fails open and keeps every article (see the
# reasoning=False note in judge_articles). A bigger window would only let a runaway
# burn more; capping it ends one in ~256 tokens instead of ~3000.
JUDGE_NUM_PREDICT = 256
# Sampling parameters. Ollama's own defaults (temperature 0.8, top_k 40, top_p 0.9)
# are tuned for open-ended chat; these three call sites want different things.
# The ReAct loop needs the same tool call for the same state, not variety — but not
# pure greedy decoding either, which can lock a stuck loop into repeating one bad
# search verbatim. 0.2 keeps tool selection stable with just enough slack to break
# out. Note this is low temperature, not determinism: reproducing a run exactly
# would also require pinning `seed`.
AGENT_TEMPERATURE = 0.2
# The judge emits a handful of YES/NO tokens and nothing else, so any sampling
# entropy is pure downside — see the runaway described at JUDGE_NUM_PREDICT.
JUDGE_TEMPERATURE = 0.0
# The summary is prose meant to be read, so it gets room to vary word choice.
# top_k/top_p are set explicitly at Ollama's defaults: at 0.5 the tail is live
# (unlike the greedy judge, where they would be inert), so the values that shape
# it belong in the source rather than in whatever the server happens to default to.
SUMMARY_TEMPERATURE = 0.5
SUMMARY_TOP_K = 40
SUMMARY_TOP_P = 0.9
# Every knob that shapes a model's output, defined once and used twice: to build the
# ChatOllama and to stamp the metrics record. Passing the same dict to both makes it
# structurally impossible for the logged config to drift from the config actually used,
# which is what lets metrics_report.py cohort runs by config and trust the answer.
AGENT_MODEL_KWARGS = {"num_ctx": AGENT_NUM_CTX, "temperature": AGENT_TEMPERATURE}
JUDGE_MODEL_KWARGS = {
    "num_predict": JUDGE_NUM_PREDICT,
    "temperature": JUDGE_TEMPERATURE,
    "reasoning": False,
}
# num_ctx is per-call for the summary (see summarize_day), so it is merged in there
# rather than fixed here.
SUMMARY_MODEL_KWARGS = {
    "temperature": SUMMARY_TEMPERATURE,
    "top_k": SUMMARY_TOP_K,
    "top_p": SUMMARY_TOP_P,
}
# Where agent-step failures dump the conversation that produced them. Ollama discards
# the malformed completion server-side (its ResponseError carries only a message and a
# status code), so the inputs are the only recoverable evidence — dumping them is what
# makes a failure replayable. Kept out of the metrics file because a failing prompt runs
# to ~4k tokens; the JSONL records only the path.
FAILURE_DIR = os.getenv("NEWS_FAILURE_DIR", "news_failures")
FAILURE_KEEP = int(os.getenv("NEWS_FAILURE_KEEP", "50"))
DEFAULT_TOPICS = (
    "anthropic",  # AI companies
    "openai",
    "ollama",
    "google",
    "xai",
    "spacex",
    "z.ai",
    "nvidia",
    "microsoft",
    "perplexity",
    "moonshot",
    "minimax",
    "hugging face",
    "groq",
    "claude",  # Models
    "gemini",
    "chatgpt",
    "gpt-5",
    "nova",
    "phi",
    "sora",
    "veo",
    "midjourney",
    "stable diffusion",
    "grok",
    "deepseek",
    "meta llama",
    "glm",
    "kimi",
    "qwen",
    "minimax",
    "lfm",
    "gemma",
    "nemotron",
    "mistral",
    "cohere",
    "llama",
    "astra",
    "fable",
    "mistral",
    "muse spark",
    "claude code",  # tools, harnesses
    "codex",
    "muse code",
    "opencode",
    "cursor",
    "langchain",
    "copilot",
    "kiro",
    "openclaw",
    "cline",
    "windsurf",
    "replit",
    "zed",
    "local llm",  # broader topics
    "vllm",
    "llamacpp",
    "sglang",
    "mlx",
    "transformers",
    "mcp",
    "model context protocol",
    "a2a",
    "coding agents",
    "ai harness",
    "artificial intelligence",
    "llm",
    "large language model",
    "foundation model",
    "ai safety",
    "ai policy",
    "reasoning models",
    "open weights",
    "agi",
    "vibe coding",
    "rag",
)

_SYSTEM_PROMPT = """\
You are an AI/ML technology news research agent. Today's date is {today}.

The user gives you one or more topics — companies, products, models, or concepts from
the AI industry. Treat each provided topic as a short, fixed label.

For each topic:
1. Search for recent news using the search tool. Use a SHORT query — just the
   topic label itself (optionally add the word "news"). Search ONE topic per
   query: never combine multiple topics into a single search, and never add
   site: filters or URLs to the query. Both return zero results.
2. For each article found, only store it if it is genuinely about AI/ML technology
   (a model, AI product/feature, AI research, AI tooling, or AI industry strategy).
   Company topics like "nvidia", "microsoft", "google", or "spacex" also generate
   plenty of ordinary corporate, financial, or product news with no AI angle — stock
   moves, earnings, unrelated product launches, layoffs, legal rulings, rocket
   launches, gaming. Skip those even though the company name matches the topic.
   For each article that passes this check, call store_result ONCE with:
   - topic: the topic label(s) this article belongs to (see rules below)
   - title: article headline
   - url: article URL
   - summary: 1-2 sentence summary of the article
   - published_on: publication date if visible in results (ISO format YYYY-MM-DD), else omit

Rules for the `topic` field — keep topics tight so they stay easy to group:
- Reuse the SAME short label the user gave you. Do not expand the label or invent longer wording,
  and do NOT turn the label into a descriptive phrase. You may shorten or make the label more succint.
- Labels are lowercase and one or two words each (e.g. "anthropic", "local llm", "ai coding").
- Multiple labels on a news article are allowed: if one article is genuinely relevant to several of the
  provided topics, set topic to a comma-separated list of those labels
  (e.g. "anthropic, claude code"). Otherwise use the single matching label.

After you have searched and stored results for ALL topics, stop.
"""

_SUMMARIZE_SYSTEM_PROMPT = """\
You are a news summarization assistant. Given a list of news articles for a single date, \
write a tight briefing of the day's major themes and most notable events.

Requirements:
- Open with ONE short overview sentence, then 3-5 themed groups of brief bullets. Favor \
brevity over completeness: merge related items and drop minor or duplicate ones.
- Use markdown for skimmability (bold the key entities, short bullets). Do NOT use tables.
- Report only what the articles say. Do NOT speculate, add "impact"/analysis commentary, \
or invent details not present in the articles.
- Output ONLY the briefing. No preamble (e.g. "Here's a summary..."), no closing remarks, \
and no questions to the reader.
"""

_JUDGE_SYSTEM_PROMPT = """\
You are a relevance judge for an AI/ML technology news digest. You will be given a \
numbered list of articles and a set of topics (companies, products, models, or \
concepts from the AI industry). For EACH numbered article, decide YES only if it is \
BOTH:
  1. genuinely about artificial intelligence / machine learning technology
     (a model, AI product/feature, AI research, AI tooling, or AI industry strategy), AND
  2. relevant to at least one of the given topics.

Many topics are company names (e.g. "nvidia", "microsoft", "google", "spacex") that \
also generate large volumes of ordinary corporate, financial, or product news with no \
AI angle — stock price moves, earnings, unrelated product launches, layoffs, legal/
antitrust rulings, rocket launches, gaming, or general company gossip. Answer NO for \
those even though the company name matches the topic. Only answer YES when the AI/ML \
angle is actually present in the title or summary.

Respond with exactly one line per article, in this format and no other text:
N: YES
N: NO

where N is the article's number. Do not add explanations, headers, or commentary."""

SEARCH_PROMPT = """Research recent news for these topics: {topic_list}.

Search each topic separately, using a short keyword query — just the topic label
itself (optionally with the word "news"). Do NOT combine multiple topics into a
single query, and do NOT add site: filters or URLs to the query; both return no
results.

Cast a broad net and store every relevant, recent article you find.
"""


def _write_metric(record: dict) -> None:
    record["logged_at"] = datetime.now(timezone.utc).isoformat()
    with open(METRICS_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


# Which run and which topic batch is currently executing. build_graph() is called once
# per run but invoked once per batch, so batch_index cannot be closed over — and tools
# receive only their own arguments, so it cannot be threaded through the graph state
# either. A ContextVar reaches both the agent node and the tool wrappers with one
# mechanism, and stays correct if the loop is ever made async.
_BATCH_CONTEXT: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "news_batch_context", default={}
)


def _log_tool_call(tool_name: str, elapsed_s: float, **extra) -> None:
    """Record one tool invocation — success or failure.

    Without this, a tool failure existed only in stdout: `_handle_tool_error` printed
    and returned a string to the model, writing nothing. Logging every call (not only
    failures) is what supplies the denominator — an error count alone cannot say
    whether a failure rate is rising or falling.
    """
    ctx = _BATCH_CONTEXT.get()
    _write_metric(
        {
            "type": "tool_call",
            "run_id": ctx.get("run_id"),
            "batch_index": ctx.get("batch_index"),
            "tool": tool_name,
            "elapsed_s": round(elapsed_s, 3),
            **extra,
        }
    )


def _dump_failed_conversation(messages: list, tools: list | None = None) -> str | None:
    """Serialize *messages* to FAILURE_DIR and return the path, or None on failure.

    Written in langchain's own `messages_to_dict` form so a dump round-trips through
    `messages_from_dict` straight back into a model call — the point is a case that can
    be replayed against Ollama, not a log line to read.

    The tool schemas are captured too, and they are not optional for the failure this
    exists to diagnose: the XML errors come from Ollama parsing a *tool call*, so a
    replay that skips `.bind_tools()` will not reproduce them. Replay with:

        model = ChatOllama(model=d["model"], base_url=..., **d["model_kwargs"])
        model.bind_tools(d["tools"]).invoke(messages_from_dict(d["messages"]))

    Never raises: a failure while recording a failure must not mask the original error.
    """
    ctx = _BATCH_CONTEXT.get()
    run_id = (ctx.get("run_id") or "adhoc").replace(":", "").replace("+", "")
    # Nanoseconds, not milliseconds: consecutive failures in one batch land inside the
    # same millisecond and would overwrite each other.
    name = f"{run_id}-batch{ctx.get('batch_index')}-{time.time_ns()}.json"
    try:
        os.makedirs(FAILURE_DIR, exist_ok=True)
        path = os.path.join(FAILURE_DIR, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "model": MODEL_NAME,
                    "base_url": OLLAMA_BASE_URL,
                    "model_kwargs": AGENT_MODEL_KWARGS,
                    "tools": [convert_to_openai_tool(t) for t in tools or []],
                    "messages": messages_to_dict(messages),
                },
                fh,
            )
    except Exception:
        return None
    # Pruning happens after the path is secured. A file vanishing between scandir and
    # remove (two runs overlapping) must not throw away a dump that was written fine.
    try:
        _prune_failure_dumps()
    except Exception:
        pass
    return path


def _prune_failure_dumps() -> None:
    """Keep only the newest FAILURE_KEEP dumps; ~12% of batches fail, so this grows."""
    entries = sorted(
        (e for e in os.scandir(FAILURE_DIR) if e.is_file()),
        key=lambda e: e.stat().st_mtime,
        reverse=True,
    )
    for entry in entries[FAILURE_KEEP:]:
        os.remove(entry.path)


def _llm_perf_fields(response) -> dict:
    """Best-effort extraction of token/timing fields from an AIMessage-like response.

    Tolerant of test doubles: a plain ``MagicMock()`` auto-creates non-dict
    attributes, so both fields are coerced to ``{}`` unless they're real dicts.
    """
    usage = getattr(response, "usage_metadata", None)
    meta = getattr(response, "response_metadata", None)
    usage = usage if isinstance(usage, dict) else {}
    meta = meta if isinstance(meta, dict) else {}
    output_tokens = usage.get("output_tokens")
    eval_duration = meta.get("eval_duration")
    tokens_per_sec = (
        round(output_tokens / (eval_duration / 1e9), 2)
        if output_tokens and eval_duration
        else None
    )
    return {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": output_tokens,
        "total_tokens": usage.get("total_tokens"),
        "load_duration_ns": meta.get("load_duration"),
        "prompt_eval_duration_ns": meta.get("prompt_eval_duration"),
        "eval_duration_ns": eval_duration,
        "total_duration_ns": meta.get("total_duration"),
        "tokens_per_sec": tokens_per_sec,
        "done_reason": meta.get("done_reason"),
    }


def _log_llm_call(run_id, step, model_name, response, elapsed_s, **extra) -> None:
    _write_metric(
        {
            "type": "llm_call",
            "run_id": run_id,
            "step": step,
            "model": model_name,
            "elapsed_s": round(elapsed_s, 3),
            **_llm_perf_fields(response),
            **extra,
        }
    )


def _log_step(run_id, step, elapsed_s, **extra) -> None:
    _write_metric(
        {
            "type": "step",
            "run_id": run_id,
            "step": step,
            "elapsed_s": round(elapsed_s, 3),
            **extra,
        }
    )


def init_db(path: str) -> None:
    """Create tables and indexes if they do not exist; migrate existing DBs."""
    with sqlite3.connect(path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS news_results (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                topic        TEXT NOT NULL,
                title        TEXT,
                url          TEXT,
                summary      TEXT,
                fetched_at   TEXT NOT NULL,
                published_on TEXT,
                model_name   TEXT
            )
        """)
        # Migrate existing DBs that lack the published_on/model_name columns.
        existing_cols = [
            r[1] for r in conn.execute("PRAGMA table_info(news_results)").fetchall()
        ]
        if "published_on" not in existing_cols:
            conn.execute("ALTER TABLE news_results ADD COLUMN published_on TEXT")
        if "model_name" not in existing_cols:
            conn.execute("ALTER TABLE news_results ADD COLUMN model_name TEXT")
        # Remove duplicate URLs before enforcing the unique index, keeping the
        # earliest row (lowest id) for each URL so existing databases migrate cleanly.
        conn.execute("""
            DELETE FROM news_results
            WHERE id NOT IN (
                SELECT MIN(id) FROM news_results GROUP BY url
            )
        """)
        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_news_url
            ON news_results (url)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS news_summaries (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                date       TEXT NOT NULL UNIQUE,
                summary    TEXT NOT NULL,
                model_name TEXT,
                created_at TEXT NOT NULL
            )
        """)
        existing_summary_cols = [
            r[1] for r in conn.execute("PRAGMA table_info(news_summaries)").fetchall()
        ]
        if "model_name" not in existing_summary_cols:
            conn.execute("ALTER TABLE news_summaries ADD COLUMN model_name TEXT")


def _normalize_topic(topic: str) -> str:
    """Lowercase and trim each comma-separated topic label so the same topic
    (e.g. "Nvidia" vs "nvidia") always collapses to one row in GROUP BY topic."""
    return ", ".join(part.strip().lower() for part in topic.split(","))


@tool
def store_result(
    topic: str, title: str, url: str, summary: str, published_on: str = ""
) -> str:
    """Store a single news article to the database.

    Args:
        topic: short topic label(s) this article belongs to — reuse the user's
            lowercase one-or-two-word label; for an article spanning several
            topics, a comma-separated list (e.g. "anthropic, claude code") is fine
        title: article headline
        url: article URL
        summary: brief 1-2 sentence summary
        published_on: publication date in YYYY-MM-DD format if known, else omit
    """
    start = time.perf_counter()
    fetched_at = datetime.now(timezone.utc).isoformat()
    pub_date = published_on.strip() or None
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO news_results"
                " (topic, title, url, summary, fetched_at, published_on, model_name)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    _normalize_topic(topic),
                    title,
                    url,
                    summary,
                    fetched_at,
                    pub_date,
                    MODEL_NAME,
                ),
            )
    except Exception as e:
        # Without this, a locked or missing database would raise before the success
        # log below and store_result's error rate would sit at a structural 0% —
        # exactly the blind spot this instrumentation exists to close.
        _log_tool_call(
            "store_result",
            time.perf_counter() - start,
            success=False,
            error=str(e),
            error_type=type(e).__name__,
        )
        raise
    # The rowcount branch already knew whether this was a new article or a duplicate;
    # it just threw the fact away. Recording it separates "the agent found nothing new"
    # from "the agent stopped calling store_result".
    stored = bool(cursor.rowcount)
    _log_tool_call(
        "store_result",
        time.perf_counter() - start,
        success=True,
        outcome="stored" if stored else "duplicate",
        topic=_normalize_topic(topic),
    )
    if stored:
        return f"Stored: {title}"
    return f"Skipped (duplicate URL): {title}"


def _instrument_tool(underlying):
    """Wrap a tool so every call is logged, without changing what the model sees.

    name, description and args_schema are copied verbatim, so the schema bound to the
    model is identical to the unwrapped tool's. That matters: renaming the search tool
    could change tool-call reliability by itself, which would confound the very failure
    rate this instrumentation exists to measure.

    Exceptions are logged and re-raised, leaving ToolNode's existing `_handle_tool_error`
    recovery path intact.
    """

    def call(**kwargs):
        start = time.perf_counter()
        try:
            result = underlying.invoke(kwargs)
        except Exception as e:
            _log_tool_call(
                underlying.name,
                time.perf_counter() - start,
                success=False,
                error=str(e),
                error_type=type(e).__name__,
                **kwargs,
            )
            raise
        # A search that returns zero results is not an error, and until now it was
        # indistinguishable from one that was never called. Counting it separately is
        # the difference between "the backend is failing" and "there is no news".
        _log_tool_call(
            underlying.name,
            time.perf_counter() - start,
            success=True,
            result_count=len(result) if isinstance(result, (list, tuple)) else None,
            **kwargs,
        )
        return result

    return StructuredTool.from_function(
        func=call,
        name=underlying.name,
        description=underlying.description,
        args_schema=underlying.args_schema,
    )


def _make_search_tool():
    if os.getenv("TAVILY_API_KEY"):
        from langchain_community.tools.tavily_search import TavilySearchResults

        return _instrument_tool(TavilySearchResults(max_results=5))
    from langchain_community.tools import DuckDuckGoSearchResults

    return _instrument_tool(
        DuckDuckGoSearchResults(backend="news", output_format="list", max_results=5)
    )


def _handle_tool_error(e: Exception) -> str:
    """Log a tool failure and return a message so the ReAct loop continues.

    The ``e: Exception`` annotation routes ALL tool exceptions here (e.g. a
    DuckDuckGo ``DDGSException: No results found.``), so an empty search for one
    topic is logged and treated as "no results" instead of crashing the run.
    """
    # A cross-check, not the primary record: _instrument_tool already logged this call
    # with its name and arguments, which ToolNode's handler signature cannot see. Kept
    # as the safety net for any tool that is not wrapped. Rates come from `tool_call`
    # records; counting both types would double-count every wrapped failure.
    _write_metric(
        {
            "type": "tool_error",
            "run_id": _BATCH_CONTEXT.get().get("run_id"),
            "batch_index": _BATCH_CONTEXT.get().get("batch_index"),
            "error": str(e),
            "error_type": type(e).__name__,
            "handled_by": "ToolNode",
        }
    )
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rprint(f"[dim][{now}][/dim] [yellow]Search/tool error (skipping): {e}[/yellow]")
    return f"No results available (tool error: {e}). Continue with the next topic."


def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
    """Route to tools if the last message has tool calls, otherwise end."""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


def build_graph(run_id: str | None = None):
    """Compile and return the news research StateGraph."""
    search_tool = _make_search_tool()
    tools = [search_tool, store_result]

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    system_message = SystemMessage(_SYSTEM_PROMPT.format(today=today))

    model = ChatOllama(
        model=MODEL_NAME, base_url=OLLAMA_BASE_URL, **AGENT_MODEL_KWARGS
    ).bind_tools(tools)
    tool_node = ToolNode(tools, handle_tool_errors=_handle_tool_error)

    def agent_node(state: MessagesState) -> dict:
        start = time.perf_counter()
        messages = [system_message] + state["messages"]
        try:
            response = model.invoke(messages)
        except Exception as e:
            # Until this existed, a call that raised never reached _log_llm_call, so
            # every parse failure was simply absent from the agent_step data — and the
            # calls that did survive were the ones that worked. Every per-call figure
            # in the report was computed over survivors only.
            _log_llm_call(
                run_id,
                "agent_step",
                MODEL_NAME,
                None,
                time.perf_counter() - start,
                batch_index=_BATCH_CONTEXT.get().get("batch_index"),
                success=False,
                error=str(e),
                error_type=type(e).__name__,
                message_count=len(messages),
                dump_path=_dump_failed_conversation(messages, tools),
                **AGENT_MODEL_KWARGS,
            )
            # Re-raise: run() still logs the search_batch failure, so the batch-level
            # and call-level records stay reconcilable.
            raise
        _log_llm_call(
            run_id,
            "agent_step",
            MODEL_NAME,
            response,
            time.perf_counter() - start,
            batch_index=_BATCH_CONTEXT.get().get("batch_index"),
            tool_call_count=len(getattr(response, "tool_calls", None) or []),
            success=True,
            **AGENT_MODEL_KWARGS,
        )
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


_JUDGE_LINE_RE = re.compile(r"(\d+)\s*[:.)]\s*(YES|NO)", re.IGNORECASE)


def _parse_judge_verdicts(response_text: str) -> dict[int, str]:
    """Parse a judge response into {article_number: "YES"|"NO"}.

    If an article number appears more than once (e.g. a reasoning model restates
    it before its final answer), the LAST occurrence wins.
    """
    verdicts: dict[int, str] = {}
    for number, verdict in _JUDGE_LINE_RE.findall(response_text):
        verdicts[int(number)] = verdict.upper()
    return verdicts


def judge_articles(topics: list[str], run_start: str) -> int:
    """Delete articles fetched since run_start that are irrelevant to topics.

    Articles are judged in batches of JUDGE_BATCH_SIZE per LLM call. If a batch's
    response can't be parsed for a given article, or the LLM call itself fails,
    that article is kept (fail-open) rather than deleted, so a malformed local-model
    response never silently destroys data.

    Returns the count of removed rows.
    """
    judge_start = time.perf_counter()
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT id, title, summary FROM news_results WHERE fetched_at >= ?",
            (run_start,),
        ).fetchall()
    if not rows:
        return 0

    topic_str = ", ".join(topics)
    # reasoning=False: qwen3.5 is a reasoning model and its <think> traces reliably
    # blow past the response length limit on a 10-article batch, causing done_reason
    # "length" with empty .content — which fails open and keeps every article,
    # including the false positives this judge exists to catch. The judge only needs
    # short YES/NO tokens, so reasoning adds cost without adding judgment quality.
    model = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, **JUDGE_MODEL_KWARGS)
    to_delete: list[int] = []
    for i in range(0, len(rows), JUDGE_BATCH_SIZE):
        batch = rows[i : i + JUDGE_BATCH_SIZE]
        articles = "\n\n".join(
            f"{n}. Article title: {title}\nArticle summary: {summary}"
            for n, (_, title, summary) in enumerate(batch, start=1)
        )
        batch_index = i // JUDGE_BATCH_SIZE
        call_start = time.perf_counter()
        try:
            response = model.invoke(
                [
                    SystemMessage(_JUDGE_SYSTEM_PROMPT),
                    HumanMessage(f"Topics: {topic_str}\n\n{articles}"),
                ]
            )
        except Exception as e:
            _log_llm_call(
                run_start,
                "judge_batch",
                MODEL_NAME,
                None,
                time.perf_counter() - call_start,
                batch_index=batch_index,
                batch_size=len(batch),
                success=False,
                error=str(e),
                error_type=type(e).__name__,
                **JUDGE_MODEL_KWARGS,
            )
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            rprint(
                f"[dim][{now}][/dim] [yellow]Judge call failed (keeping batch "
                f"unfiltered): {e}[/yellow]"
            )
            continue
        _log_llm_call(
            run_start,
            "judge_batch",
            MODEL_NAME,
            response,
            time.perf_counter() - call_start,
            batch_index=batch_index,
            batch_size=len(batch),
            success=True,
            **JUDGE_MODEL_KWARGS,
        )
        verdicts = _parse_judge_verdicts(response.content)
        for n, (row_id, _title, _summary) in enumerate(batch, start=1):
            if verdicts.get(n) == "NO":
                to_delete.append(row_id)

    if to_delete:
        placeholders = ",".join("?" * len(to_delete))
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                f"DELETE FROM news_results WHERE id IN ({placeholders})", to_delete
            )
    _log_step(
        run_start,
        "judge_articles",
        time.perf_counter() - judge_start,
        articles_judged=len(rows),
        articles_deleted=len(to_delete),
    )
    return len(to_delete)


def _log_summary_warning(message: str, style: str = "yellow") -> None:
    """Print a timestamped warning about the summary step (mirrors _handle_tool_error)."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rprint(f"[dim][{now}][/dim] [{style}]{message}[/{style}]")


def _invoke_summary_model(
    model_name: str, messages: list, num_ctx: int, run_id: str | None, step: str
) -> str:
    """Invoke a summary model and return its stripped text, raising on empty output.

    Treating an empty/whitespace response as a failure lets ``summarize_day`` fall
    back to ``MODEL_NAME`` (and ultimately store nothing) instead of silently
    persisting a blank summary — the failure mode seen with oversized models that
    spill to CPU.
    """
    kwargs = {**SUMMARY_MODEL_KWARGS, "num_ctx": num_ctx}
    start = time.perf_counter()
    try:
        response = ChatOllama(
            model=model_name, base_url=OLLAMA_BASE_URL, **kwargs
        ).invoke(messages)
    except Exception as exc:
        _log_llm_call(
            run_id,
            step,
            model_name,
            None,
            time.perf_counter() - start,
            success=False,
            error=str(exc),
            error_type=type(exc).__name__,
            **kwargs,
        )
        raise
    text = (response.content or "").strip()
    _log_llm_call(
        run_id,
        step,
        model_name,
        response,
        time.perf_counter() - start,
        success=bool(text),
        **kwargs,
    )
    if not text:
        raise ValueError(f"{model_name} returned an empty summary")
    return text


def summarize_day(
    date: str,
    model: str | None = None,
    num_ctx: int | None = None,
    run_id: str | None = None,
) -> str | None:
    """Summarize all news articles stored for *date* (YYYY-MM-DD) and upsert into news_summaries.

    *model* overrides the summary model (default ``SUMMARY_MODEL_NAME``); *num_ctx*
    overrides the context window (default ``SUMMARY_NUM_CTX``). Both default to None
    so positional ``summarize_day(date)`` calls keep working. *run_id* correlates
    metrics with the enclosing ``run()`` loop iteration, if any.

    Returns the summary text, or None if no articles exist for that date.
    """
    primary_model = model or SUMMARY_MODEL_NAME
    ctx = num_ctx or SUMMARY_NUM_CTX
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT title, summary FROM news_results WHERE fetched_at LIKE ?",
            (f"{date}%",),
        ).fetchall()
    if not rows:
        return None
    digest = "\n\n".join(f"- {title}: {summary}" for title, summary in rows)
    messages = [
        SystemMessage(_SUMMARIZE_SYSTEM_PROMPT),
        HumanMessage(f"Summarize the news for {date}:\n\n{digest}"),
    ]
    summarize_start = time.perf_counter()
    try:
        summary_text = _invoke_summary_model(
            primary_model, messages, ctx, run_id, "summarize_primary"
        )
        used_model = primary_model
    except Exception as exc:
        # A failed or empty response from the only model is fatal; otherwise warn
        # loudly and retry once with MODEL_NAME so a misconfigured SUMMARY_MODEL
        # surfaces instead of silently degrading.
        if primary_model == MODEL_NAME:
            _log_step(
                run_id,
                "summarize_day",
                time.perf_counter() - summarize_start,
                success=False,
            )
            raise
        _log_summary_warning(
            f"summary model {primary_model!r} failed ({exc}); "
            f"falling back to {MODEL_NAME!r}"
        )
        try:
            summary_text = _invoke_summary_model(
                MODEL_NAME, messages, ctx, run_id, "summarize_fallback"
            )
            used_model = MODEL_NAME
        except Exception as exc2:
            _log_summary_warning(
                f"fallback model {MODEL_NAME!r} also failed ({exc2}); "
                f"no summary stored for {date}",
                style="red",
            )
            _log_step(
                run_id,
                "summarize_day",
                time.perf_counter() - summarize_start,
                success=False,
            )
            return None
    created_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO news_summaries (date, summary, model_name, created_at)"
            " VALUES (?, ?, ?, ?)",
            (date, summary_text, used_model, created_at),
        )
    _log_step(
        run_id,
        "summarize_day",
        time.perf_counter() - summarize_start,
        used_model=used_model,
        success=True,
    )
    return summary_text


def run(topics: list[str]) -> tuple[int, str | None]:
    """Research news for each topic and store to SQLite.

    Returns ``(new_article_count, daily_briefing)`` where the briefing is the
    summary text produced by ``summarize_day`` (or ``None`` if no articles exist
    for today).
    """
    run_loop_start = time.perf_counter()
    init_db(DB_PATH)
    run_start = datetime.now(timezone.utc).isoformat()
    today = run_start[:10]
    app = build_graph(run_id=run_start)
    # Search topics in small batches: each invoke gets a fresh recursion budget,
    # and a failed batch is logged without abandoning the remaining topics.
    for i in range(0, len(topics), BATCH_SIZE):
        batch = topics[i : i + BATCH_SIZE]
        topic_list = ", ".join(batch)
        batch_start = time.perf_counter()
        # Stamp every agent step and tool call in this batch with the batch it belongs
        # to. A run spans ~64 batches; without this, a failed search_batch could not be
        # traced back to the LLM calls and tool calls that produced it except by
        # guessing from timestamp order.
        _BATCH_CONTEXT.set({"run_id": run_start, "batch_index": i // BATCH_SIZE})
        try:
            app.invoke(
                {
                    "messages": [
                        HumanMessage(SEARCH_PROMPT.format(topic_list=topic_list))
                    ]
                },
                config={"recursion_limit": 50},
            )
        except Exception as e:
            _log_step(
                run_start,
                "search_batch",
                time.perf_counter() - batch_start,
                batch_index=i // BATCH_SIZE,
                topics=topic_list,
                success=False,
                error=str(e),
                # Separates GraphRecursionError from ollama's ResponseError without
                # string-matching the message — the distinction the report's old
                # "every failure is the same failure" caption missed.
                error_type=type(e).__name__,
                model=MODEL_NAME,
            )
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            rprint(
                f"[dim][{now}][/dim] [red]News research failed for batch "
                f"{topic_list}: {e}[/red]"
            )
            continue
        _log_step(
            run_start,
            "search_batch",
            time.perf_counter() - batch_start,
            batch_index=i // BATCH_SIZE,
            topics=topic_list,
            success=True,
            model=MODEL_NAME,
        )
    with sqlite3.connect(DB_PATH) as conn:
        (new_count,) = conn.execute(
            "SELECT COUNT(*) FROM news_results WHERE fetched_at >= ?", (run_start,)
        ).fetchone()
    judge_articles(topics, run_start)
    summary = summarize_day(today, run_id=run_start)
    _log_step(
        run_start,
        "run",
        time.perf_counter() - run_loop_start,
        topics=len(topics),
        new_articles=new_count,
    )
    return new_count, summary


def _print_summary(new_count: int, topics: list[str]) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    topic_str = ", ".join(topics)
    if new_count:
        noun = "article" if new_count == 1 else "articles"
        rprint(
            f"[dim][{now}][/dim] [bold green]{new_count} new {noun} found[/bold green] for: {topic_str}"
        )
    else:
        rprint(f"[dim][{now}][/dim] [yellow]No new articles[/yellow] for: {topic_str}")


def _print_briefing(summary: str | None) -> None:
    """Render the day's briefing as markdown, or do nothing if there is none."""
    if not summary:
        return
    rprint("\n[bold cyan]Daily Briefing[/bold cyan]")
    rprint(Markdown(summary))


def _summarize_cli(
    day: str | None, model: str | None = None, num_ctx: int | None = None
) -> None:
    """Ad-hoc: (re)generate and store the summary for a single day.

    *day* is a YYYY-MM-DD string, or None to default to today (UTC, matching how
    `fetched_at` is stored). The date is validated up front so a malformed value
    fails loudly instead of silently matching no articles via the LIKE filter.
    *model* and *num_ctx* override the summary model and context window for
    experimentation (both default to None → use the configured defaults).
    """
    if day is None:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    # strptime is lenient about zero-padding ("2026-6-21" parses), but the stored
    # fetched_at values are canonical "2026-06-21", so a non-padded date would
    # silently match no rows. Round-trip to require the exact YYYY-MM-DD form.
    try:
        parsed = datetime.strptime(day, "%Y-%m-%d")
    except ValueError:
        raise SystemExit(f"Invalid --day {day!r}: expected YYYY-MM-DD")
    if parsed.strftime("%Y-%m-%d") != day:
        raise SystemExit(f"Invalid --day {day!r}: expected zero-padded YYYY-MM-DD")
    init_db(DB_PATH)
    summary = summarize_day(day, model=model, num_ctx=num_ctx)
    if summary is None:
        rprint(f"[yellow]No articles stored for {day}[/yellow]")
    else:
        rprint(summary)


def run_cli() -> None:
    """CLI entry point.

    Default: read topics from argv and research news, polling on a fixed interval
    (set NEWS_INTERVAL seconds to repeat; unset or 0 runs once).

    With --summarize: (re)generate the stored summary for --day (default today)
    instead of researching.
    """
    parser = argparse.ArgumentParser(description="News research agent.")
    parser.add_argument("topics", nargs="*", help="topics to research")
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="summarize a stored day instead of researching",
    )
    parser.add_argument(
        "--day",
        help="day to summarize as YYYY-MM-DD (default: today, UTC)",
    )
    parser.add_argument(
        "--model",
        help="override the summary model (default: $SUMMARY_MODEL)",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        help="override the summary context window (default: $SUMMARY_NUM_CTX)",
    )
    args = parser.parse_args()

    if args.summarize:
        _summarize_cli(args.day, model=args.model, num_ctx=args.num_ctx)
        return

    topics = args.topics or list(DEFAULT_TOPICS)
    interval = int(os.getenv("NEWS_INTERVAL", "0"))
    while True:
        new_count, summary = run(topics)
        _print_summary(new_count, topics)
        _print_briefing(summary)
        if not interval:
            break
        rprint(f"[dim]Sleeping {interval}s...[/dim]")
        time.sleep(interval)


if __name__ == "__main__":  # pragma: no cover
    run_cli()
