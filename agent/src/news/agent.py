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
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
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
OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DB_PATH = os.getenv("NEWS_DB_PATH", "news_results.db")
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
DEFAULT_TOPICS = (
    "anthropic",  # AI companies
    "openai",
    "ollama",
    "Google",
    "xai",
    "SpaceX",
    "Z.ai",
    "Nvidia",
    "Microsoft",
    "perplexity",
    "Moonshot",
    "hugging face",
    "Groq",
    "claude",  # Models
    "Gemini",
    "chatgpt",
    "GPT-5",
    "nova",
    "Phi",
    "Sora",
    "veo",
    "Midjourney",
    "Stable Diffusion",
    "grok",
    "deepseek",
    "meta llama",
    "GLM",
    "Kimi",
    "Qwen",
    "Minimax",
    "LFM",
    "gemma",
    "nemotron",
    "mistral",
    "Cohere",
    "claude code",  # tools, harnesses
    "codex",
    "opencode",
    "Cursor",
    "LangChain",
    "copilot",
    "kiro",
    "openclaw",
    "cline",
    "windsurf",
    "replit",
    "zed",
    "local llm",  # broader topics
    "vllm",
    "MCP",
    "model context protocol",
    "A2A",
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
    "AGI",
    "vibe coding",
    "RAG",
)

_SYSTEM_PROMPT = """\
You are a news research agent. Today's date is {today}.

The user gives you one or more topics. Treat each provided topic as a short, fixed label.

For each topic:
1. Search for recent news using the search tool. Use a SHORT query — just the
   topic label itself (optionally add the word "news"). Search ONE topic per
   query: never combine multiple topics into a single search, and never add
   site: filters or URLs to the query. Both return zero results.
2. For each relevant article found, call store_result ONCE with:
   - topic: the topic label(s) this article belongs to (see rules below)
   - title: article headline
   - url: article URL
   - summary: 1-2 sentence summary of the article
   - published_on: publication date if visible in results (ISO format YYYY-MM-DD), else omit

Rules for the `topic` field — keep topics tight so they stay easy to group:
- Reuse the SAME short label the user gave you. Do not expand the label or invent longer wording,
  and do NOT turn the label into a descriptive phrase. You may shorten or make the label more succint.
- Labels are lowercase and one or two words each (e.g. "anthropic", "local llm", "ai coding").
- Multiple topics are allowed: if one article is genuinely relevant to several of the
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
You are a relevance judge. You will be given a numbered list of articles and a set \
of topics. For EACH numbered article, decide if it is genuinely relevant to at \
least one topic.

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
                published_on TEXT
            )
        """)
        # Migrate existing DBs that lack the published_on column.
        existing_cols = [
            r[1] for r in conn.execute("PRAGMA table_info(news_results)").fetchall()
        ]
        if "published_on" not in existing_cols:
            conn.execute("ALTER TABLE news_results ADD COLUMN published_on TEXT")
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
                created_at TEXT NOT NULL
            )
        """)


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
    fetched_at = datetime.now(timezone.utc).isoformat()
    pub_date = published_on.strip() or None
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "INSERT OR IGNORE INTO news_results (topic, title, url, summary, fetched_at, published_on)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (topic, title, url, summary, fetched_at, pub_date),
        )
    if cursor.rowcount:
        return f"Stored: {title}"
    return f"Skipped (duplicate URL): {title}"


def _make_search_tool():
    if os.getenv("TAVILY_API_KEY"):
        from langchain_community.tools.tavily_search import TavilySearchResults

        return TavilySearchResults(max_results=5)
    from langchain_community.tools import DuckDuckGoSearchResults

    return DuckDuckGoSearchResults(backend="news", output_format="list", max_results=5)


def _handle_tool_error(e: Exception) -> str:
    """Log a tool failure and return a message so the ReAct loop continues.

    The ``e: Exception`` annotation routes ALL tool exceptions here (e.g. a
    DuckDuckGo ``DDGSException: No results found.``), so an empty search for one
    topic is logged and treated as "no results" instead of crashing the run.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rprint(f"[dim][{now}][/dim] [yellow]Search/tool error (skipping): {e}[/yellow]")
    return f"No results available (tool error: {e}). Continue with the next topic."


def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
    """Route to tools if the last message has tool calls, otherwise end."""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


def build_graph():
    """Compile and return the news research StateGraph."""
    search_tool = _make_search_tool()
    tools = [search_tool, store_result]

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    system_message = SystemMessage(_SYSTEM_PROMPT.format(today=today))

    model = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL).bind_tools(tools)
    tool_node = ToolNode(tools, handle_tool_errors=_handle_tool_error)

    def agent_node(state: MessagesState) -> dict:
        response = model.invoke([system_message] + state["messages"])
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
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT id, title, summary FROM news_results WHERE fetched_at >= ?",
            (run_start,),
        ).fetchall()
    if not rows:
        return 0

    topic_str = ", ".join(topics)
    model = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)
    to_delete: list[int] = []
    for i in range(0, len(rows), JUDGE_BATCH_SIZE):
        batch = rows[i : i + JUDGE_BATCH_SIZE]
        articles = "\n\n".join(
            f"{n}. Article title: {title}\nArticle summary: {summary}"
            for n, (_, title, summary) in enumerate(batch, start=1)
        )
        try:
            response = model.invoke(
                [
                    SystemMessage(_JUDGE_SYSTEM_PROMPT),
                    HumanMessage(f"Topics: {topic_str}\n\n{articles}"),
                ]
            )
        except Exception as e:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            rprint(
                f"[dim][{now}][/dim] [yellow]Judge call failed (keeping batch "
                f"unfiltered): {e}[/yellow]"
            )
            continue
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
    return len(to_delete)


def _log_summary_warning(message: str, style: str = "yellow") -> None:
    """Print a timestamped warning about the summary step (mirrors _handle_tool_error)."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rprint(f"[dim][{now}][/dim] [{style}]{message}[/{style}]")


def _invoke_summary_model(model_name: str, messages: list, num_ctx: int) -> str:
    """Invoke a summary model and return its stripped text, raising on empty output.

    Treating an empty/whitespace response as a failure lets ``summarize_day`` fall
    back to ``MODEL_NAME`` (and ultimately store nothing) instead of silently
    persisting a blank summary — the failure mode seen with oversized models that
    spill to CPU.
    """
    response = ChatOllama(
        model=model_name, base_url=OLLAMA_BASE_URL, num_ctx=num_ctx
    ).invoke(messages)
    text = (response.content or "").strip()
    if not text:
        raise ValueError(f"{model_name} returned an empty summary")
    return text


def summarize_day(
    date: str, model: str | None = None, num_ctx: int | None = None
) -> str | None:
    """Summarize all news articles stored for *date* (YYYY-MM-DD) and upsert into news_summaries.

    *model* overrides the summary model (default ``SUMMARY_MODEL_NAME``); *num_ctx*
    overrides the context window (default ``SUMMARY_NUM_CTX``). Both default to None
    so positional ``summarize_day(date)`` calls keep working.

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
    try:
        summary_text = _invoke_summary_model(primary_model, messages, ctx)
    except Exception as exc:
        # A failed or empty response from the only model is fatal; otherwise warn
        # loudly and retry once with MODEL_NAME so a misconfigured SUMMARY_MODEL
        # surfaces instead of silently degrading.
        if primary_model == MODEL_NAME:
            raise
        _log_summary_warning(
            f"summary model {primary_model!r} failed ({exc}); "
            f"falling back to {MODEL_NAME!r}"
        )
        try:
            summary_text = _invoke_summary_model(MODEL_NAME, messages, ctx)
        except Exception as exc2:
            _log_summary_warning(
                f"fallback model {MODEL_NAME!r} also failed ({exc2}); "
                f"no summary stored for {date}",
                style="red",
            )
            return None
    created_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO news_summaries (date, summary, created_at) VALUES (?, ?, ?)",
            (date, summary_text, created_at),
        )
    return summary_text


def run(topics: list[str]) -> tuple[int, str | None]:
    """Research news for each topic and store to SQLite.

    Returns ``(new_article_count, daily_briefing)`` where the briefing is the
    summary text produced by ``summarize_day`` (or ``None`` if no articles exist
    for today).
    """
    init_db(DB_PATH)
    run_start = datetime.now(timezone.utc).isoformat()
    today = run_start[:10]
    app = build_graph()
    # Search topics in small batches: each invoke gets a fresh recursion budget,
    # and a failed batch is logged without abandoning the remaining topics.
    for i in range(0, len(topics), BATCH_SIZE):
        batch = topics[i : i + BATCH_SIZE]
        topic_list = ", ".join(batch)
        try:
            app.invoke(
                {"messages": [HumanMessage(SEARCH_PROMPT.format(topic_list=topic_list))]},
                config={"recursion_limit": 50},
            )
        except Exception as e:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            rprint(
                f"[dim][{now}][/dim] [red]News research failed for batch "
                f"{topic_list}: {e}[/red]"
            )
    with sqlite3.connect(DB_PATH) as conn:
        (new_count,) = conn.execute(
            "SELECT COUNT(*) FROM news_results WHERE fetched_at >= ?", (run_start,)
        ).fetchone()
    judge_articles(topics, run_start)
    summary = summarize_day(today)
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
