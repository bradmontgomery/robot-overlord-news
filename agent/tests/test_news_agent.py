import os
import sqlite3
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------


def test_init_db_creates_table(tmp_path):
    from news.agent import init_db

    db_file = str(tmp_path / "test.db")
    init_db(db_file)
    with sqlite3.connect(db_file) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='news_results'"
        ).fetchall()
    assert tables == [("news_results",)]


def test_init_db_idempotent(tmp_path):
    from news.agent import init_db

    db_file = str(tmp_path / "test.db")
    init_db(db_file)
    init_db(db_file)  # second call must not raise


# ---------------------------------------------------------------------------
# store_result tool
# ---------------------------------------------------------------------------


def test_store_result_inserts_row(tmp_path):
    from news import agent as news_agent

    db_file = str(tmp_path / "test.db")
    news_agent.init_db(db_file)
    with patch.object(news_agent, "DB_PATH", db_file):
        result = news_agent.store_result.invoke(
            {"topic": "AI", "title": "Test Article", "url": "http://example.com", "summary": "Short summary."}
        )
    assert "Test Article" in result
    with sqlite3.connect(db_file) as conn:
        rows = conn.execute("SELECT topic, title, url, summary FROM news_results").fetchall()
    assert rows == [("AI", "Test Article", "http://example.com", "Short summary.")]


def test_store_result_sets_fetched_at(tmp_path):
    from news import agent as news_agent

    db_file = str(tmp_path / "test.db")
    news_agent.init_db(db_file)
    with patch.object(news_agent, "DB_PATH", db_file):
        news_agent.store_result.invoke(
            {"topic": "t", "title": "T", "url": "u", "summary": "s"}
        )
    with sqlite3.connect(db_file) as conn:
        fetched_at = conn.execute("SELECT fetched_at FROM news_results").fetchone()[0]
    assert fetched_at  # non-empty ISO string


# ---------------------------------------------------------------------------
# _make_search_tool
# ---------------------------------------------------------------------------


def test_make_search_tool_duckduckgo():
    from news.agent import _make_search_tool

    env = {k: v for k, v in os.environ.items() if k != "TAVILY_API_KEY"}
    with patch.dict(os.environ, env, clear=True):
        tool = _make_search_tool()
    from langchain_community.tools import DuckDuckGoSearchResults

    assert isinstance(tool, DuckDuckGoSearchResults)
    assert tool.backend == "news"
    assert tool.output_format == "list"


def test_make_search_tool_tavily():
    from news.agent import _make_search_tool

    mock_cls = MagicMock()
    mock_instance = MagicMock()
    mock_cls.return_value = mock_instance
    with patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}), \
         patch("langchain_community.tools.tavily_search.TavilySearchResults", mock_cls):
        tool = _make_search_tool()
    mock_cls.assert_called_once_with(max_results=5)
    assert tool is mock_instance


# ---------------------------------------------------------------------------
# should_continue
# ---------------------------------------------------------------------------


def test_should_continue_with_tool_calls():
    from langgraph.graph import END
    from news.agent import should_continue

    msg = AIMessage(content="", tool_calls=[{"name": "store_result", "args": {}, "id": "1", "type": "tool_call"}])
    result = should_continue({"messages": [msg]})
    assert result == "tools"


def test_should_continue_without_tool_calls():
    from langgraph.graph import END
    from news.agent import should_continue

    msg = AIMessage(content="Done.")
    result = should_continue({"messages": [msg]})
    assert result == END


def test_should_continue_empty_tool_calls():
    from langgraph.graph import END
    from news.agent import should_continue

    msg = AIMessage(content="Done.", tool_calls=[])
    result = should_continue({"messages": [msg]})
    assert result == END


# ---------------------------------------------------------------------------
# build_graph
# ---------------------------------------------------------------------------


def test_build_graph_duckduckgo():
    from news.agent import build_graph

    mock_model = MagicMock()
    mock_model.bind_tools.return_value = mock_model
    with patch("news.agent.ChatOllama", return_value=mock_model), \
         patch("news.agent.ToolNode"), \
         patch.dict(os.environ, {k: v for k, v in os.environ.items() if k != "TAVILY_API_KEY"}, clear=True):
        graph = build_graph()
    assert graph is not None


def test_build_graph_tavily():
    from news.agent import build_graph

    mock_model = MagicMock()
    mock_model.bind_tools.return_value = mock_model
    mock_tavily_cls = MagicMock()
    mock_tavily_cls.return_value = MagicMock()
    with patch("news.agent.ChatOllama", return_value=mock_model), \
         patch("news.agent.ToolNode"), \
         patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}), \
         patch("langchain_community.tools.tavily_search.TavilySearchResults", mock_tavily_cls):
        graph = build_graph()
    assert graph is not None


def test_build_graph_agent_node_invokes_model():
    """Verify agent_node prepends system message and calls model.invoke."""
    from news.agent import build_graph

    mock_response = AIMessage(content="done")
    mock_model = MagicMock()
    mock_model.bind_tools.return_value = mock_model
    mock_model.invoke.return_value = mock_response

    with patch("news.agent.ChatOllama", return_value=mock_model), \
         patch.dict(os.environ, {k: v for k, v in os.environ.items() if k != "TAVILY_API_KEY"}, clear=True):
        graph = build_graph()

    # Invoke the graph — model returns AIMessage with no tool_calls so it terminates
    result = graph.invoke({"messages": [HumanMessage("test")]})
    mock_model.invoke.assert_called_once()
    call_args = mock_model.invoke.call_args[0][0]
    # First message must be the system prompt
    from langchain_core.messages import SystemMessage
    assert isinstance(call_args[0], SystemMessage)
    assert "news research agent" in call_args[0].content


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


def test_run_calls_init_db_and_returns_rows(tmp_path):
    from news import agent as news_agent

    db_file = str(tmp_path / "run_test.db")

    mock_app = MagicMock()
    mock_app.invoke.return_value = {}

    with patch.object(news_agent, "DB_PATH", db_file), \
         patch("news.agent.build_graph", return_value=mock_app):
        results = news_agent.run(["topic1"])

    mock_app.invoke.assert_called_once()
    # DB was initialised (table exists) but no rows stored by mocked app
    assert results == []


def test_run_returns_stored_rows(tmp_path):
    from news import agent as news_agent

    db_file = str(tmp_path / "run_rows.db")
    news_agent.init_db(db_file)
    with sqlite3.connect(db_file) as conn:
        conn.execute(
            "INSERT INTO news_results (topic, title, url, summary, fetched_at)"
            " VALUES (?, ?, ?, ?, ?)",
            ("AI", "Headline", "http://x.com", "Summary text.", "2026-01-01T00:00:00"),
        )

    mock_app = MagicMock()
    mock_app.invoke.return_value = {}

    with patch.object(news_agent, "DB_PATH", db_file), \
         patch("news.agent.build_graph", return_value=mock_app):
        results = news_agent.run(["AI"])

    assert len(results) == 1
    assert results[0]["topic"] == "AI"
    assert results[0]["title"] == "Headline"


# ---------------------------------------------------------------------------
# run_cli()
# ---------------------------------------------------------------------------


def test_run_cli_prints_results(capsys):
    from news import agent as news_agent

    fake_results = [
        {"topic": "AI", "title": "Title A", "url": "http://a.com", "summary": "Summary A.", "fetched_at": "2026-01-01"},
        {"topic": "AI", "title": "Title B", "url": "http://b.com", "summary": "Summary B.", "fetched_at": "2026-01-02"},
    ]
    with patch("news.agent.run", return_value=fake_results), \
         patch("sys.argv", ["news_agent", "AI"]):
        news_agent.run_cli()

    out = capsys.readouterr().out
    assert "## AI" in out
    assert "Title A" in out
    assert "Title B" in out


def test_run_cli_no_results(capsys):
    from news import agent as news_agent

    with patch("news.agent.run", return_value=[]), \
         patch("sys.argv", ["news_agent", "AI"]):
        news_agent.run_cli()

    out = capsys.readouterr().out
    assert "No results stored." in out


def test_run_cli_default_topics():
    from news import agent as news_agent

    with patch("news.agent.run", return_value=[]) as mock_run, \
         patch("sys.argv", ["news_agent"]):
        news_agent.run_cli()

    mock_run.assert_called_once_with(["AI", "climate"])


def test_run_cli_polls_on_interval(capsys):
    from news import agent as news_agent

    # sleep returns None on first call, then raises to exit the loop
    with patch("news.agent.run", return_value=[]) as mock_run, \
         patch("news.agent.time.sleep", side_effect=[None, KeyboardInterrupt]) as mock_sleep, \
         patch("sys.argv", ["news_agent", "AI"]), \
         patch.dict(os.environ, {"NEWS_INTERVAL": "60"}):
        try:
            news_agent.run_cli()
        except KeyboardInterrupt:
            pass

    assert mock_run.call_count == 2
    mock_sleep.assert_called_with(60)


def test_run_cli_interval_zero_runs_once():
    from news import agent as news_agent

    with patch("news.agent.run", return_value=[]) as mock_run, \
         patch("news.agent.time.sleep") as mock_sleep, \
         patch("sys.argv", ["news_agent", "AI"]), \
         patch.dict(os.environ, {"NEWS_INTERVAL": "0"}):
        news_agent.run_cli()

    mock_run.assert_called_once()
    mock_sleep.assert_not_called()


def test_run_cli_interval_unset_runs_once():
    from news import agent as news_agent

    env = {k: v for k, v in os.environ.items() if k != "NEWS_INTERVAL"}
    with patch("news.agent.run", return_value=[]) as mock_run, \
         patch("news.agent.time.sleep") as mock_sleep, \
         patch("sys.argv", ["news_agent"]), \
         patch.dict(os.environ, env, clear=True):
        news_agent.run_cli()

    mock_run.assert_called_once()
    mock_sleep.assert_not_called()
