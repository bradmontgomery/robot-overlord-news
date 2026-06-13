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
from datetime import datetime, timezone
from typing import Literal

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
    """Create the news_results table if it does not exist."""
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
        conn.execute(
            "INSERT INTO news_results (topic, title, url, summary, fetched_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (topic, title, url, summary, fetched_at),
        )
    return f"Stored: {title}"


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


def run(topics: list[str]) -> list[dict]:
    """Research news for each topic, store to SQLite, and return stored rows."""
    init_db(DB_PATH)
    app = build_graph()
    topic_list = ", ".join(topics)
    app.invoke(
        {"messages": [HumanMessage(f"Research recent news for: {topic_list}")]},
        config={"recursion_limit": 50},
    )
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT topic, title, url, summary, fetched_at"
            " FROM news_results ORDER BY fetched_at DESC LIMIT 100"
        ).fetchall()
    return [
        {"topic": r[0], "title": r[1], "url": r[2], "summary": r[3], "fetched_at": r[4]}
        for r in rows
    ]


def run_cli() -> None:
    """CLI entry point: read topics from argv and print results."""
    topics = sys.argv[1:] or ["AI", "climate"]
    results = run(topics)
    if not results:
        print("No results stored.")
        return
    current_topic = None
    for r in reversed(results):
        if r["topic"] != current_topic:
            current_topic = r["topic"]
            print(f"\n## {current_topic}")
        print(f"  {r['title']}")
        print(f"  {r['url']}")
        print(f"  {r['summary']}")
        print()


if __name__ == "__main__":  # pragma: no cover
    run_cli()
