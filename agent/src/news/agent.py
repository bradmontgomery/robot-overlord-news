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

import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from typing import Literal

from rich import print as rprint

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

MODEL_NAME = os.getenv("OLLAMA_MODEL", "gemma4:latest")
SUMMARY_MODEL_NAME = os.getenv("SUMMARY_MODEL", "gemma4:26b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DB_PATH = os.getenv("NEWS_DB_PATH", "news_results.db")

_SYSTEM_PROMPT = """\
You are a news research agent. Today's date is {today}.

For each topic the user provides:
1. Search for recent news using the search tool.
2. For each relevant article found, call store_result ONCE with:
   - topic: the topic you searched for
   - title: article headline
   - url: article URL
   - summary: 1-2 sentence summary of the article
   - published_on: publication date if visible in results (ISO format YYYY-MM-DD), else omit

After you have searched and stored results for ALL topics, stop.
"""

_SUMMARIZE_SYSTEM_PROMPT = """\
You are a news summarization assistant. Given a list of news articles for a specific date, \
write a concise high-level summary (2-4 paragraphs) covering the major themes and notable events.
"""

_JUDGE_SYSTEM_PROMPT = "You are a relevance judge. Answer only YES or NO — no explanation."


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
        existing_cols = [r[1] for r in conn.execute("PRAGMA table_info(news_results)").fetchall()]
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
def store_result(topic: str, title: str, url: str, summary: str, published_on: str = "") -> str:
    """Store a single news article to the database.

    Args:
        topic: the search topic this article belongs to
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
    tool_node = ToolNode(tools)

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


def judge_articles(topics: list[str], run_start: str) -> int:
    """Delete articles fetched since run_start that are irrelevant to topics.

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
    removed = 0
    for row_id, title, summary in rows:
        response = model.invoke([
            SystemMessage(_JUDGE_SYSTEM_PROMPT),
            HumanMessage(
                f"Topics: {topic_str}\n\n"
                f"Article title: {title}\n"
                f"Article summary: {summary}\n\n"
                f"Is this article genuinely relevant to at least one of the topics?"
            ),
        ])
        if "yes" not in response.content.lower():
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("DELETE FROM news_results WHERE id = ?", (row_id,))
            removed += 1
    return removed


def summarize_day(date: str) -> str | None:
    """Summarize all news articles stored for *date* (YYYY-MM-DD) and upsert into news_summaries.

    Returns the summary text, or None if no articles exist for that date.
    """
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
        response = ChatOllama(model=SUMMARY_MODEL_NAME, base_url=OLLAMA_BASE_URL).invoke(messages)
    except Exception:
        if SUMMARY_MODEL_NAME == MODEL_NAME:
            raise
        response = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL).invoke(messages)
    summary_text = response.content
    created_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO news_summaries (date, summary, created_at) VALUES (?, ?, ?)",
            (date, summary_text, created_at),
        )
    return summary_text


def run(topics: list[str]) -> int:
    """Research news for each topic, store to SQLite, and return count of new articles."""
    init_db(DB_PATH)
    run_start = datetime.now(timezone.utc).isoformat()
    today = run_start[:10]
    app = build_graph()
    topic_list = ", ".join(topics)
    app.invoke(
        {"messages": [HumanMessage(f"Research recent news for: {topic_list}")]},
        config={"recursion_limit": 50},
    )
    with sqlite3.connect(DB_PATH) as conn:
        (new_count,) = conn.execute(
            "SELECT COUNT(*) FROM news_results WHERE fetched_at >= ?", (run_start,)
        ).fetchone()
    judge_articles(topics, run_start)
    summarize_day(today)
    return new_count


def _print_summary(new_count: int, topics: list[str]) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    topic_str = ", ".join(topics)
    if new_count:
        noun = "article" if new_count == 1 else "articles"
        rprint(f"[dim][{now}][/dim] [bold green]{new_count} new {noun} found[/bold green] for: {topic_str}")
    else:
        rprint(f"[dim][{now}][/dim] [yellow]No new articles[/yellow] for: {topic_str}")


def run_cli() -> None:
    """CLI entry point: read topics from argv and poll on a fixed interval.

    Set NEWS_INTERVAL (seconds) to run repeatedly; unset or 0 runs once.
    """
    topics = sys.argv[1:] or ["AI", "climate"]
    interval = int(os.getenv("NEWS_INTERVAL", "0"))
    while True:
        new_count = run(topics)
        _print_summary(new_count, topics)
        if not interval:
            break
        rprint(f"[dim]Sleeping {interval}s...[/dim]")
        time.sleep(interval)


if __name__ == "__main__":  # pragma: no cover
    run_cli()
