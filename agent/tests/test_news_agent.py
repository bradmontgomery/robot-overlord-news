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


def test_init_db_deduplicates_existing_rows(tmp_path):
    from news.agent import init_db

    db_file = str(tmp_path / "test.db")
    # Simulate a pre-existing DB without the unique index (plain inserts, duplicates allowed)
    with sqlite3.connect(db_file) as conn:
        conn.execute("""
            CREATE TABLE news_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL, title TEXT, url TEXT, summary TEXT, fetched_at TEXT NOT NULL
            )
        """)
        conn.executemany(
            "INSERT INTO news_results (topic, title, url, summary, fetched_at) VALUES (?,?,?,?,?)",
            [
                ("AI", "Article", "http://dup.com", "s", "2026-01-01T00:00:00"),
                ("AI", "Article", "http://dup.com", "s", "2026-01-02T00:00:00"),  # duplicate
                ("AI", "Unique", "http://unique.com", "s", "2026-01-01T00:00:00"),
            ],
        )
    init_db(db_file)  # must not raise
    with sqlite3.connect(db_file) as conn:
        count = conn.execute("SELECT COUNT(*) FROM news_results").fetchone()[0]
    assert count == 2  # duplicate removed, unique row kept


def test_init_db_creates_unique_index(tmp_path):
    from news.agent import init_db

    db_file = str(tmp_path / "test.db")
    init_db(db_file)
    with sqlite3.connect(db_file) as conn:
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_news_url'"
        ).fetchall()
    assert indexes == [("idx_news_url",)]


def test_init_db_adds_published_on_to_existing_db(tmp_path):
    from news.agent import init_db

    db_file = str(tmp_path / "test.db")
    # Simulate a DB without published_on (old schema)
    with sqlite3.connect(db_file) as conn:
        conn.execute("""
            CREATE TABLE news_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL, title TEXT, url TEXT, summary TEXT, fetched_at TEXT NOT NULL
            )
        """)
    init_db(db_file)
    with sqlite3.connect(db_file) as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(news_results)").fetchall()]
    assert "published_on" in cols


def test_init_db_creates_news_summaries_table(tmp_path):
    from news.agent import init_db

    db_file = str(tmp_path / "test.db")
    init_db(db_file)
    with sqlite3.connect(db_file) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='news_summaries'"
        ).fetchall()
    assert tables == [("news_summaries",)]


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


def test_store_result_stores_published_on(tmp_path):
    from news import agent as news_agent

    db_file = str(tmp_path / "test.db")
    news_agent.init_db(db_file)
    with patch.object(news_agent, "DB_PATH", db_file):
        news_agent.store_result.invoke(
            {"topic": "AI", "title": "Dated Article", "url": "http://dated.com",
             "summary": "s", "published_on": "2026-06-16"}
        )
    with sqlite3.connect(db_file) as conn:
        pub = conn.execute("SELECT published_on FROM news_results").fetchone()[0]
    assert pub == "2026-06-16"


def test_store_result_published_on_none_when_omitted(tmp_path):
    from news import agent as news_agent

    db_file = str(tmp_path / "test.db")
    news_agent.init_db(db_file)
    with patch.object(news_agent, "DB_PATH", db_file):
        news_agent.store_result.invoke(
            {"topic": "AI", "title": "No Date", "url": "http://nodate.com", "summary": "s"}
        )
    with sqlite3.connect(db_file) as conn:
        pub = conn.execute("SELECT published_on FROM news_results").fetchone()[0]
    assert pub is None


def test_store_result_deduplicates_by_url(tmp_path):
    from news import agent as news_agent

    db_file = str(tmp_path / "test.db")
    news_agent.init_db(db_file)
    with patch.object(news_agent, "DB_PATH", db_file):
        news_agent.store_result.invoke(
            {"topic": "AI", "title": "Article", "url": "http://example.com", "summary": "s"}
        )
        result = news_agent.store_result.invoke(
            {"topic": "AI", "title": "Article", "url": "http://example.com", "summary": "s"}
        )
    assert "Skipped" in result
    with sqlite3.connect(db_file) as conn:
        count = conn.execute("SELECT COUNT(*) FROM news_results").fetchone()[0]
    assert count == 1


def test_store_result_allows_different_urls(tmp_path):
    from news import agent as news_agent

    db_file = str(tmp_path / "test.db")
    news_agent.init_db(db_file)
    with patch.object(news_agent, "DB_PATH", db_file):
        news_agent.store_result.invoke(
            {"topic": "AI", "title": "Article A", "url": "http://a.com", "summary": "s"}
        )
        news_agent.store_result.invoke(
            {"topic": "AI", "title": "Article B", "url": "http://b.com", "summary": "s"}
        )
    with sqlite3.connect(db_file) as conn:
        count = conn.execute("SELECT COUNT(*) FROM news_results").fetchone()[0]
    assert count == 2


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


def test_run_returns_zero_when_no_articles_stored(tmp_path):
    from news import agent as news_agent

    db_file = str(tmp_path / "run_test.db")

    mock_app = MagicMock()
    mock_app.invoke.return_value = {}

    with patch.object(news_agent, "DB_PATH", db_file), \
         patch("news.agent.build_graph", return_value=mock_app), \
         patch("news.agent.judge_articles"), \
         patch("news.agent.summarize_day"):
        count = news_agent.run(["topic1"])

    mock_app.invoke.assert_called_once()
    assert count == 0


def test_run_returns_count_of_new_articles(tmp_path):
    from news import agent as news_agent

    db_file = str(tmp_path / "run_rows.db")
    news_agent.init_db(db_file)

    def fake_invoke(messages, config=None):
        # Simulate the agent storing a row during the run (current timestamp)
        from datetime import datetime, timezone
        fetched_at = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(db_file) as conn:
            conn.execute(
                "INSERT INTO news_results (topic, title, url, summary, fetched_at)"
                " VALUES (?, ?, ?, ?, ?)",
                ("AI", "Headline", "http://x.com", "Summary.", fetched_at),
            )
        return {}

    mock_app = MagicMock()
    mock_app.invoke.side_effect = fake_invoke

    with patch.object(news_agent, "DB_PATH", db_file), \
         patch("news.agent.build_graph", return_value=mock_app), \
         patch("news.agent.judge_articles"), \
         patch("news.agent.summarize_day"):
        count = news_agent.run(["AI"])

    assert count == 1


def test_run_calls_summarize_day(tmp_path):
    from news import agent as news_agent

    db_file = str(tmp_path / "run_summarize.db")
    mock_app = MagicMock()
    mock_app.invoke.return_value = {}

    with patch.object(news_agent, "DB_PATH", db_file), \
         patch("news.agent.build_graph", return_value=mock_app), \
         patch("news.agent.judge_articles"), \
         patch("news.agent.summarize_day") as mock_summarize:
        news_agent.run(["AI"])

    mock_summarize.assert_called_once()
    date_arg = mock_summarize.call_args[0][0]
    assert len(date_arg) == 10  # YYYY-MM-DD


def test_run_calls_judge_articles(tmp_path):
    from news import agent as news_agent

    db_file = str(tmp_path / "run_judge.db")
    mock_app = MagicMock()
    mock_app.invoke.return_value = {}

    with patch.object(news_agent, "DB_PATH", db_file), \
         patch("news.agent.build_graph", return_value=mock_app), \
         patch("news.agent.judge_articles") as mock_judge, \
         patch("news.agent.summarize_day"):
        news_agent.run(["AI", "climate"])

    mock_judge.assert_called_once()
    topics_arg, run_start_arg = mock_judge.call_args[0]
    assert topics_arg == ["AI", "climate"]
    assert len(run_start_arg) > 10  # ISO timestamp


# ---------------------------------------------------------------------------
# run_cli()
# ---------------------------------------------------------------------------


def test_run_cli_calls_print_summary():
    from news import agent as news_agent

    with patch("news.agent.run", return_value=3), \
         patch("news.agent._print_summary") as mock_summary, \
         patch("news.agent.rprint"), \
         patch("sys.argv", ["news_agent", "AI"]), \
         patch.dict(os.environ, {"NEWS_INTERVAL": "0"}):
        news_agent.run_cli()

    mock_summary.assert_called_once_with(3, ["AI"])


def test_run_cli_no_results():
    from news import agent as news_agent

    with patch("news.agent.run", return_value=0), \
         patch("news.agent._print_summary") as mock_summary, \
         patch("news.agent.rprint"), \
         patch("sys.argv", ["news_agent", "AI"]), \
         patch.dict(os.environ, {"NEWS_INTERVAL": "0"}):
        news_agent.run_cli()

    mock_summary.assert_called_once_with(0, ["AI"])


def test_print_summary_new_articles():
    from news import agent as news_agent

    with patch("news.agent.rprint") as mock_print:
        news_agent._print_summary(3, ["AI", "climate"])

    mock_print.assert_called_once()
    output = mock_print.call_args[0][0]
    assert "3 new articles found" in output
    assert "AI, climate" in output


def test_print_summary_singular():
    from news import agent as news_agent

    with patch("news.agent.rprint") as mock_print:
        news_agent._print_summary(1, ["AI"])

    output = mock_print.call_args[0][0]
    assert "1 new article found" in output
    assert "articles" not in output


def test_print_summary_no_results():
    from news import agent as news_agent

    with patch("news.agent.rprint") as mock_print:
        news_agent._print_summary(0, ["AI"])

    output = mock_print.call_args[0][0]
    assert "No new articles" in output


def test_run_cli_default_topics():
    from news import agent as news_agent

    with patch("news.agent.run", return_value=0) as mock_run, \
         patch("news.agent._print_summary"), \
         patch("news.agent.rprint"), \
         patch("sys.argv", ["news_agent"]), \
         patch.dict(os.environ, {"NEWS_INTERVAL": "0"}):
        news_agent.run_cli()

    mock_run.assert_called_once_with(list(news_agent.DEFAULT_TOPICS))


def test_run_cli_polls_on_interval():
    from news import agent as news_agent

    # sleep returns None on first call, then raises to exit the loop
    with patch("news.agent.run", return_value=0) as mock_run, \
         patch("news.agent._print_summary"), \
         patch("news.agent.rprint"), \
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

    with patch("news.agent.run", return_value=0) as mock_run, \
         patch("news.agent._print_summary"), \
         patch("news.agent.rprint"), \
         patch("news.agent.time.sleep") as mock_sleep, \
         patch("sys.argv", ["news_agent", "AI"]), \
         patch.dict(os.environ, {"NEWS_INTERVAL": "0"}):
        news_agent.run_cli()

    mock_run.assert_called_once()
    mock_sleep.assert_not_called()


def test_run_cli_interval_unset_runs_once():
    from news import agent as news_agent

    env = {k: v for k, v in os.environ.items() if k != "NEWS_INTERVAL"}
    with patch("news.agent.run", return_value=0) as mock_run, \
         patch("news.agent._print_summary"), \
         patch("news.agent.rprint"), \
         patch("news.agent.time.sleep") as mock_sleep, \
         patch("sys.argv", ["news_agent"]), \
         patch.dict(os.environ, env, clear=True):
        news_agent.run_cli()

    mock_run.assert_called_once()
    mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# summarize_day()
# ---------------------------------------------------------------------------


def test_summarize_day_returns_none_when_no_articles(tmp_path):
    from news import agent as news_agent

    db_file = str(tmp_path / "sum.db")
    news_agent.init_db(db_file)

    with patch.object(news_agent, "DB_PATH", db_file):
        result = news_agent.summarize_day("2026-06-16")

    assert result is None


def test_summarize_day_stores_summary(tmp_path):
    from news import agent as news_agent

    db_file = str(tmp_path / "sum.db")
    news_agent.init_db(db_file)
    with sqlite3.connect(db_file) as conn:
        conn.execute(
            "INSERT INTO news_results (topic, title, url, summary, fetched_at) VALUES (?,?,?,?,?)",
            ("AI", "Headline", "http://h.com", "Short summary.", "2026-06-16T10:00:00+00:00"),
        )

    mock_model = MagicMock()
    mock_model.invoke.return_value = MagicMock(content="Today AI dominated the news.")

    with patch.object(news_agent, "DB_PATH", db_file), \
         patch("news.agent.ChatOllama", return_value=mock_model):
        result = news_agent.summarize_day("2026-06-16")

    assert result == "Today AI dominated the news."
    with sqlite3.connect(db_file) as conn:
        row = conn.execute(
            "SELECT date, summary FROM news_summaries WHERE date='2026-06-16'"
        ).fetchone()
    assert row == ("2026-06-16", "Today AI dominated the news.")


def test_summarize_day_upserts_on_rerun(tmp_path):
    from news import agent as news_agent

    db_file = str(tmp_path / "sum.db")
    news_agent.init_db(db_file)
    with sqlite3.connect(db_file) as conn:
        conn.execute(
            "INSERT INTO news_results (topic, title, url, summary, fetched_at) VALUES (?,?,?,?,?)",
            ("AI", "Article", "http://a.com", "Summary.", "2026-06-16T10:00:00+00:00"),
        )

    mock_model = MagicMock()
    mock_model.invoke.side_effect = [
        MagicMock(content="First summary."),
        MagicMock(content="Updated summary."),
    ]

    with patch.object(news_agent, "DB_PATH", db_file), \
         patch("news.agent.ChatOllama", return_value=mock_model):
        news_agent.summarize_day("2026-06-16")
        news_agent.summarize_day("2026-06-16")

    with sqlite3.connect(db_file) as conn:
        rows = conn.execute("SELECT summary FROM news_summaries WHERE date='2026-06-16'").fetchall()
    assert len(rows) == 1  # only one row due to UNIQUE constraint
    assert rows[0][0] == "Updated summary."


def test_summarize_day_passes_articles_to_model(tmp_path):
    from news import agent as news_agent
    from langchain_core.messages import HumanMessage as HM

    db_file = str(tmp_path / "sum.db")
    news_agent.init_db(db_file)
    with sqlite3.connect(db_file) as conn:
        conn.execute(
            "INSERT INTO news_results (topic, title, url, summary, fetched_at) VALUES (?,?,?,?,?)",
            ("AI", "Big Story", "http://b.com", "Details here.", "2026-06-16T12:00:00+00:00"),
        )

    mock_model = MagicMock()
    mock_model.invoke.return_value = MagicMock(content="Summary text.")

    with patch.object(news_agent, "DB_PATH", db_file), \
         patch("news.agent.ChatOllama", return_value=mock_model):
        news_agent.summarize_day("2026-06-16")

    call_messages = mock_model.invoke.call_args[0][0]
    human_msg = call_messages[1]
    assert "Big Story" in human_msg.content
    assert "Details here." in human_msg.content


def test_summarize_day_falls_back_to_model_name(tmp_path):
    from news import agent as news_agent

    db_file = str(tmp_path / "sum.db")
    news_agent.init_db(db_file)
    with sqlite3.connect(db_file) as conn:
        conn.execute(
            "INSERT INTO news_results (topic, title, url, summary, fetched_at) VALUES (?,?,?,?,?)",
            ("AI", "Story", "http://s.com", "Summary.", "2026-06-16T10:00:00+00:00"),
        )

    failing_model = MagicMock()
    failing_model.invoke.side_effect = Exception("model not found")
    working_model = MagicMock()
    working_model.invoke.return_value = MagicMock(content="Fallback summary.")

    with patch.object(news_agent, "DB_PATH", db_file), \
         patch.object(news_agent, "SUMMARY_MODEL_NAME", "missing-model"), \
         patch.object(news_agent, "MODEL_NAME", "fallback-model"), \
         patch("news.agent.ChatOllama", side_effect=[failing_model, working_model]):
        result = news_agent.summarize_day("2026-06-16")

    assert result == "Fallback summary."


def test_summarize_day_raises_when_only_model_fails(tmp_path):
    from news import agent as news_agent

    db_file = str(tmp_path / "sum.db")
    news_agent.init_db(db_file)
    with sqlite3.connect(db_file) as conn:
        conn.execute(
            "INSERT INTO news_results (topic, title, url, summary, fetched_at) VALUES (?,?,?,?,?)",
            ("AI", "Story", "http://s2.com", "Summary.", "2026-06-16T10:00:00+00:00"),
        )

    failing_model = MagicMock()
    failing_model.invoke.side_effect = Exception("model not found")

    with patch.object(news_agent, "DB_PATH", db_file), \
         patch.object(news_agent, "SUMMARY_MODEL_NAME", "only-model"), \
         patch.object(news_agent, "MODEL_NAME", "only-model"), \
         patch("news.agent.ChatOllama", return_value=failing_model):
        with pytest.raises(Exception, match="model not found"):
            news_agent.summarize_day("2026-06-16")


# ---------------------------------------------------------------------------
# judge_articles()
# ---------------------------------------------------------------------------


def test_judge_articles_no_rows_returns_zero(tmp_path):
    from news import agent as news_agent

    db_file = str(tmp_path / "judge.db")
    news_agent.init_db(db_file)

    with patch.object(news_agent, "DB_PATH", db_file):
        result = news_agent.judge_articles(["AI"], "2026-06-16T00:00:00+00:00")

    assert result == 0


def test_judge_articles_keeps_relevant_row(tmp_path):
    from news import agent as news_agent

    db_file = str(tmp_path / "judge.db")
    news_agent.init_db(db_file)
    run_start = "2026-06-16T10:00:00+00:00"
    with sqlite3.connect(db_file) as conn:
        conn.execute(
            "INSERT INTO news_results (topic, title, url, summary, fetched_at) VALUES (?,?,?,?,?)",
            ("AI", "AI Breakthrough", "http://ai.com", "About AI.", run_start),
        )

    mock_model = MagicMock()
    mock_model.invoke.return_value = MagicMock(content="YES")

    with patch.object(news_agent, "DB_PATH", db_file), \
         patch("news.agent.ChatOllama", return_value=mock_model):
        removed = news_agent.judge_articles(["AI"], run_start)

    assert removed == 0
    with sqlite3.connect(db_file) as conn:
        count = conn.execute("SELECT COUNT(*) FROM news_results").fetchone()[0]
    assert count == 1


def test_judge_articles_removes_irrelevant_row(tmp_path):
    from news import agent as news_agent

    db_file = str(tmp_path / "judge.db")
    news_agent.init_db(db_file)
    run_start = "2026-06-16T10:00:00+00:00"
    with sqlite3.connect(db_file) as conn:
        conn.execute(
            "INSERT INTO news_results (topic, title, url, summary, fetched_at) VALUES (?,?,?,?,?)",
            ("AI", "Celebrity Gossip", "http://gossip.com", "A celeb did something.", run_start),
        )

    mock_model = MagicMock()
    mock_model.invoke.return_value = MagicMock(content="NO")

    with patch.object(news_agent, "DB_PATH", db_file), \
         patch("news.agent.ChatOllama", return_value=mock_model):
        removed = news_agent.judge_articles(["AI"], run_start)

    assert removed == 1
    with sqlite3.connect(db_file) as conn:
        count = conn.execute("SELECT COUNT(*) FROM news_results").fetchone()[0]
    assert count == 0


def test_judge_articles_returns_count(tmp_path):
    from news import agent as news_agent

    db_file = str(tmp_path / "judge.db")
    news_agent.init_db(db_file)
    run_start = "2026-06-16T10:00:00+00:00"
    with sqlite3.connect(db_file) as conn:
        conn.executemany(
            "INSERT INTO news_results (topic, title, url, summary, fetched_at) VALUES (?,?,?,?,?)",
            [
                ("AI", "Good Article", "http://good.com", "Relevant.", run_start),
                ("AI", "Bad Article", "http://bad.com", "Irrelevant.", run_start),
            ],
        )

    mock_model = MagicMock()
    mock_model.invoke.side_effect = [
        MagicMock(content="YES"),
        MagicMock(content="NO"),
    ]

    with patch.object(news_agent, "DB_PATH", db_file), \
         patch("news.agent.ChatOllama", return_value=mock_model):
        removed = news_agent.judge_articles(["AI"], run_start)

    assert removed == 1
    with sqlite3.connect(db_file) as conn:
        remaining = conn.execute("SELECT title FROM news_results").fetchall()
    assert remaining == [("Good Article",)]


