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

After you have searched and stored results for ALL topics, stop.
"""


def init_db(path: str) -> None:
    """Create the news_results table and unique URL index if they do not exist."""
    with sqlite3.connect(path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS news_results (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                topic      TEXT NOT NULL,
                title      TEXT,
                url        TEXT,
                summary    TEXT,
                fetched_at TEXT NOT NULL
            )
        """)
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


@tool
def store_result(topic: str, title: str, url: str, summary: str) -> str:
    """Store a single news article to the database.

    Args:
        topic: the search topic this article belongs to
        title: article headline
        url: article URL
        summary: brief 1-2 sentence summary
    """
    fetched_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "INSERT OR IGNORE INTO news_results (topic, title, url, summary, fetched_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (topic, title, url, summary, fetched_at),
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


def run(topics: list[str]) -> int:
    """Research news for each topic, store to SQLite, and return count of new articles."""
    init_db(DB_PATH)
    run_start = datetime.now(timezone.utc).isoformat()
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
