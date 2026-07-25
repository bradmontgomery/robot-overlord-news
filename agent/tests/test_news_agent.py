import os
import sqlite3
from datetime import datetime, timezone
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
    assert rows == [("ai", "Test Article", "http://example.com", "Short summary.")]


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


def test_store_result_lowercases_topic(tmp_path):
    from news import agent as news_agent

    db_file = str(tmp_path / "test.db")
    news_agent.init_db(db_file)
    with patch.object(news_agent, "DB_PATH", db_file):
        news_agent.store_result.invoke(
            {"topic": "Nvidia", "title": "T", "url": "http://example.com", "summary": "s"}
        )
    with sqlite3.connect(db_file) as conn:
        topic = conn.execute("SELECT topic FROM news_results").fetchone()[0]
    assert topic == "nvidia"


def test_store_result_normalizes_multi_topic(tmp_path):
    from news import agent as news_agent

    db_file = str(tmp_path / "test.db")
    news_agent.init_db(db_file)
    with patch.object(news_agent, "DB_PATH", db_file):
        news_agent.store_result.invoke(
            {"topic": "Anthropic,  Claude Code", "title": "T", "url": "http://example.com", "summary": "s"}
        )
    with sqlite3.connect(db_file) as conn:
        topic = conn.execute("SELECT topic FROM news_results").fetchone()[0]
    assert topic == "anthropic, claude code"


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
# _handle_tool_error
# ---------------------------------------------------------------------------


def test_handle_tool_error_logs_and_returns_message():
    from news import agent as news_agent

    with patch("news.agent.rprint") as mock_print:
        result = news_agent._handle_tool_error(Exception("No results found."))

    assert "No results found." in result
    assert result  # non-empty message keeps the ReAct loop going
    mock_print.assert_called_once()
    logged = mock_print.call_args[0][0]
    assert "No results found." in logged


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
        count, _ = news_agent.run(["topic1"])

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
        count, _ = news_agent.run(["AI"])

    assert count == 1


def test_run_batches_topics(tmp_path):
    """Topics are searched in BATCH_SIZE-sized invokes, never one giant query."""
    from news import agent as news_agent

    db_file = str(tmp_path / "batch.db")
    mock_app = MagicMock()
    mock_app.invoke.return_value = {}
    topics = [f"topic{i}" for i in range(20)]

    with patch.object(news_agent, "DB_PATH", db_file), \
         patch.object(news_agent, "BATCH_SIZE", 8), \
         patch("news.agent.build_graph", return_value=mock_app), \
         patch("news.agent.judge_articles") as mock_judge, \
         patch("news.agent.summarize_day") as mock_summarize:
        news_agent.run(topics)

    # 20 topics / 8 per batch -> 3 invokes (8 + 8 + 4)
    assert mock_app.invoke.call_count == 3
    first_prompt = mock_app.invoke.call_args_list[0].args[0]["messages"][0].content
    assert "topic7" in first_prompt  # first batch
    assert "topic8" not in first_prompt  # second batch not included in one query
    # judge + summarize still run once over the whole topic set
    mock_judge.assert_called_once()
    assert mock_judge.call_args[0][0] == topics
    mock_summarize.assert_called_once()


def test_run_continues_after_batch_failure(tmp_path):
    """A failed batch is logged but the remaining batches still run."""
    from news import agent as news_agent

    db_file = str(tmp_path / "batch_fail.db")
    mock_app = MagicMock()
    mock_app.invoke.side_effect = [RuntimeError("boom"), {}, {}]
    topics = [f"topic{i}" for i in range(20)]

    with patch.object(news_agent, "DB_PATH", db_file), \
         patch.object(news_agent, "BATCH_SIZE", 8), \
         patch("news.agent.build_graph", return_value=mock_app), \
         patch("news.agent.judge_articles"), \
         patch("news.agent.summarize_day"), \
         patch("news.agent.rprint") as mock_print:
        count, _ = news_agent.run(topics)

    assert mock_app.invoke.call_count == 3  # all batches attempted despite failure
    assert count == 0
    assert any("boom" in str(c.args[0]) for c in mock_print.call_args_list)


def test_run_survives_invoke_failure(tmp_path):
    """A failed app.invoke is logged and does not crash run(); judge+summarize still run."""
    from news import agent as news_agent

    db_file = str(tmp_path / "run_fail.db")
    mock_app = MagicMock()
    mock_app.invoke.side_effect = RuntimeError("graph blew up")

    with patch.object(news_agent, "DB_PATH", db_file), \
         patch("news.agent.build_graph", return_value=mock_app), \
         patch("news.agent.judge_articles") as mock_judge, \
         patch("news.agent.summarize_day") as mock_summarize, \
         patch("news.agent.rprint") as mock_print:
        count, _ = news_agent.run(["AI"])

    assert count == 0
    mock_judge.assert_called_once()
    mock_summarize.assert_called_once()
    # the failure was logged
    assert any("graph blew up" in str(c.args[0]) for c in mock_print.call_args_list)


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

    with patch("news.agent.run", return_value=(3, "briefing")), \
         patch("news.agent._print_summary") as mock_summary, \
         patch("news.agent._print_briefing") as mock_briefing, \
         patch("news.agent.rprint"), \
         patch("sys.argv", ["news_agent", "AI"]), \
         patch.dict(os.environ, {"NEWS_INTERVAL": "0"}):
        news_agent.run_cli()

    mock_summary.assert_called_once_with(3, ["AI"])
    mock_briefing.assert_called_once_with("briefing")


def test_run_cli_no_results():
    from news import agent as news_agent

    with patch("news.agent.run", return_value=(0, None)), \
         patch("news.agent._print_summary") as mock_summary, \
         patch("news.agent._print_briefing"), \
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


def test_print_briefing_renders():
    from news import agent as news_agent
    from rich.markdown import Markdown

    with patch("news.agent.rprint") as mock_print:
        news_agent._print_briefing("# Headlines\n\n- one\n- two")

    # header line + the rendered Markdown
    assert mock_print.call_count == 2
    rendered = mock_print.call_args_list[1].args[0]
    assert isinstance(rendered, Markdown)


def test_print_briefing_none():
    from news import agent as news_agent

    with patch("news.agent.rprint") as mock_print:
        news_agent._print_briefing(None)

    mock_print.assert_not_called()


def test_run_returns_summary(tmp_path):
    from news import agent as news_agent

    db_file = str(tmp_path / "run_returns_summary.db")
    mock_app = MagicMock()
    mock_app.invoke.return_value = {}

    with patch.object(news_agent, "DB_PATH", db_file), \
         patch("news.agent.build_graph", return_value=mock_app), \
         patch("news.agent.judge_articles"), \
         patch("news.agent.summarize_day", return_value="briefing text"):
        count, summary = news_agent.run(["AI"])

    assert count == 0
    assert summary == "briefing text"


def test_run_cli_default_topics():
    from news import agent as news_agent

    with patch("news.agent.run", return_value=(0, None)) as mock_run, \
         patch("news.agent._print_summary"), \
         patch("news.agent.rprint"), \
         patch("sys.argv", ["news_agent"]), \
         patch.dict(os.environ, {"NEWS_INTERVAL": "0"}):
        news_agent.run_cli()

    mock_run.assert_called_once_with(list(news_agent.DEFAULT_TOPICS))


def test_run_cli_polls_on_interval():
    from news import agent as news_agent

    # sleep returns None on first call, then raises to exit the loop
    with patch("news.agent.run", return_value=(0, None)) as mock_run, \
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

    with patch("news.agent.run", return_value=(0, None)) as mock_run, \
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
    with patch("news.agent.run", return_value=(0, None)) as mock_run, \
         patch("news.agent._print_summary"), \
         patch("news.agent.rprint"), \
         patch("news.agent.time.sleep") as mock_sleep, \
         patch("sys.argv", ["news_agent"]), \
         patch.dict(os.environ, env, clear=True):
        news_agent.run_cli()

    mock_run.assert_called_once()
    mock_sleep.assert_not_called()


def test_run_cli_summarize_explicit_day():
    from news import agent as news_agent

    with patch("news.agent.summarize_day", return_value="The summary") as mock_sum, \
         patch("news.agent.init_db") as mock_init, \
         patch("news.agent.run") as mock_run, \
         patch("news.agent.rprint") as mock_print, \
         patch("sys.argv", ["news_agent", "--summarize", "--day", "2026-06-21"]):
        news_agent.run_cli()

    mock_sum.assert_called_once_with("2026-06-21", model=None, num_ctx=None)
    mock_init.assert_called_once()
    mock_run.assert_not_called()
    mock_print.assert_called_once_with("The summary")


def test_run_cli_summarize_passes_model_and_num_ctx():
    from news import agent as news_agent

    with patch("news.agent.summarize_day", return_value="s") as mock_sum, \
         patch("news.agent.init_db"), \
         patch("news.agent.rprint"), \
         patch("sys.argv", ["news_agent", "--summarize", "--day", "2026-06-21",
                            "--model", "qwen3.5:latest", "--num-ctx", "4096"]):
        news_agent.run_cli()

    mock_sum.assert_called_once_with("2026-06-21", model="qwen3.5:latest", num_ctx=4096)


def test_run_cli_summarize_defaults_to_today_utc():
    from news import agent as news_agent

    expected = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with patch("news.agent.summarize_day", return_value="x") as mock_sum, \
         patch("news.agent.init_db"), \
         patch("news.agent.rprint"), \
         patch("sys.argv", ["news_agent", "--summarize"]):
        news_agent.run_cli()

    mock_sum.assert_called_once_with(expected, model=None, num_ctx=None)


def test_run_cli_summarize_no_articles():
    from news import agent as news_agent

    with patch("news.agent.summarize_day", return_value=None), \
         patch("news.agent.init_db"), \
         patch("news.agent.rprint") as mock_print, \
         patch("sys.argv", ["news_agent", "--summarize", "--day", "2026-06-21"]):
        news_agent.run_cli()

    output = mock_print.call_args[0][0]
    assert "No articles stored for 2026-06-21" in output


def test_run_cli_summarize_unpadded_day():
    from news import agent as news_agent

    with patch("news.agent.summarize_day") as mock_sum, \
         patch("news.agent.init_db") as mock_init, \
         patch("sys.argv", ["news_agent", "--summarize", "--day", "2026-6-21"]):
        with pytest.raises(SystemExit):
            news_agent.run_cli()

    mock_sum.assert_not_called()
    mock_init.assert_not_called()


def test_run_cli_summarize_unparseable_day():
    from news import agent as news_agent

    with patch("news.agent.summarize_day") as mock_sum, \
         patch("news.agent.init_db") as mock_init, \
         patch("sys.argv", ["news_agent", "--summarize", "--day", "notadate"]):
        with pytest.raises(SystemExit):
            news_agent.run_cli()

    mock_sum.assert_not_called()
    mock_init.assert_not_called()


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


def test_summarize_day_scopes_to_fetched_at_date(tmp_path):
    """Only articles whose fetched_at falls on the target date are summarized."""
    from news import agent as news_agent

    db_file = str(tmp_path / "sum.db")
    news_agent.init_db(db_file)
    with sqlite3.connect(db_file) as conn:
        conn.executemany(
            "INSERT INTO news_results (topic, title, url, summary, fetched_at) VALUES (?,?,?,?,?)",
            [
                ("AI", "Today Story", "http://today.com", "Happened today.",
                 "2026-06-16T12:00:00+00:00"),
                ("AI", "Yesterday Story", "http://yest.com", "Happened yesterday.",
                 "2026-06-15T23:59:00+00:00"),
                ("AI", "Tomorrow Story", "http://tom.com", "Happens tomorrow.",
                 "2026-06-17T00:01:00+00:00"),
            ],
        )

    mock_model = MagicMock()
    mock_model.invoke.return_value = MagicMock(content="Summary text.")

    with patch.object(news_agent, "DB_PATH", db_file), \
         patch("news.agent.ChatOllama", return_value=mock_model):
        news_agent.summarize_day("2026-06-16")

    human_msg = mock_model.invoke.call_args[0][0][1]
    assert "Today Story" in human_msg.content
    # Adjacent days must be excluded by the fetched_at filter.
    assert "Yesterday Story" not in human_msg.content
    assert "Tomorrow Story" not in human_msg.content


def test_summarize_day_uses_default_num_ctx(tmp_path):
    from news import agent as news_agent

    db_file = str(tmp_path / "sum.db")
    news_agent.init_db(db_file)
    with sqlite3.connect(db_file) as conn:
        conn.execute(
            "INSERT INTO news_results (topic, title, url, summary, fetched_at) VALUES (?,?,?,?,?)",
            ("AI", "Story", "http://s.com", "Summary.", "2026-06-16T10:00:00+00:00"),
        )

    mock_model = MagicMock()
    mock_model.invoke.return_value = MagicMock(content="Summary text.")

    with patch.object(news_agent, "DB_PATH", db_file), \
         patch("news.agent.ChatOllama", return_value=mock_model) as mock_chat:
        news_agent.summarize_day("2026-06-16")

    _, kwargs = mock_chat.call_args
    assert kwargs["model"] == news_agent.SUMMARY_MODEL_NAME
    assert kwargs["num_ctx"] == news_agent.SUMMARY_NUM_CTX


def test_summarize_day_model_and_num_ctx_override(tmp_path):
    from news import agent as news_agent

    db_file = str(tmp_path / "sum.db")
    news_agent.init_db(db_file)
    with sqlite3.connect(db_file) as conn:
        conn.execute(
            "INSERT INTO news_results (topic, title, url, summary, fetched_at) VALUES (?,?,?,?,?)",
            ("AI", "Story", "http://s.com", "Summary.", "2026-06-16T10:00:00+00:00"),
        )

    mock_model = MagicMock()
    mock_model.invoke.return_value = MagicMock(content="Summary text.")

    with patch.object(news_agent, "DB_PATH", db_file), \
         patch("news.agent.ChatOllama", return_value=mock_model) as mock_chat:
        news_agent.summarize_day("2026-06-16", model="qwen3.5:latest", num_ctx=8192)

    _, kwargs = mock_chat.call_args
    assert kwargs["model"] == "qwen3.5:latest"
    assert kwargs["num_ctx"] == 8192


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


def test_summarize_day_empty_primary_falls_back(tmp_path):
    """An empty/whitespace primary response triggers the MODEL_NAME fallback + a warning."""
    from news import agent as news_agent

    db_file = str(tmp_path / "sum.db")
    news_agent.init_db(db_file)
    with sqlite3.connect(db_file) as conn:
        conn.execute(
            "INSERT INTO news_results (topic, title, url, summary, fetched_at) VALUES (?,?,?,?,?)",
            ("AI", "Story", "http://s.com", "Summary.", "2026-06-16T10:00:00+00:00"),
        )

    empty_model = MagicMock()
    empty_model.invoke.return_value = MagicMock(content="   \n")  # whitespace only
    working_model = MagicMock()
    working_model.invoke.return_value = MagicMock(content="Fallback summary.")

    with patch.object(news_agent, "DB_PATH", db_file), \
         patch.object(news_agent, "SUMMARY_MODEL_NAME", "primary-model"), \
         patch.object(news_agent, "MODEL_NAME", "fallback-model"), \
         patch("news.agent.ChatOllama", side_effect=[empty_model, working_model]), \
         patch("news.agent.rprint") as mock_print:
        result = news_agent.summarize_day("2026-06-16")

    assert result == "Fallback summary."
    assert any("falling back" in str(c.args[0]) for c in mock_print.call_args_list)
    # the blank is never persisted — the stored row is the fallback text
    with sqlite3.connect(db_file) as conn:
        stored = conn.execute(
            "SELECT summary FROM news_summaries WHERE date='2026-06-16'"
        ).fetchone()[0]
    assert stored == "Fallback summary."


def test_summarize_day_returns_none_when_all_models_empty(tmp_path):
    """If primary and fallback both return empty, store nothing and return None."""
    from news import agent as news_agent

    db_file = str(tmp_path / "sum.db")
    news_agent.init_db(db_file)
    with sqlite3.connect(db_file) as conn:
        conn.execute(
            "INSERT INTO news_results (topic, title, url, summary, fetched_at) VALUES (?,?,?,?,?)",
            ("AI", "Story", "http://s.com", "Summary.", "2026-06-16T10:00:00+00:00"),
        )

    empty_model = MagicMock()
    empty_model.invoke.return_value = MagicMock(content="")

    with patch.object(news_agent, "DB_PATH", db_file), \
         patch.object(news_agent, "SUMMARY_MODEL_NAME", "primary-model"), \
         patch.object(news_agent, "MODEL_NAME", "fallback-model"), \
         patch("news.agent.ChatOllama", return_value=empty_model), \
         patch("news.agent.rprint") as mock_print:
        result = news_agent.summarize_day("2026-06-16")

    assert result is None
    with sqlite3.connect(db_file) as conn:
        rows = conn.execute(
            "SELECT * FROM news_summaries WHERE date='2026-06-16'"
        ).fetchall()
    assert rows == []  # nothing persisted
    assert any("no summary stored" in str(c.args[0]) for c in mock_print.call_args_list)


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
    mock_model.invoke.return_value = MagicMock(content="1: YES")

    with patch.object(news_agent, "DB_PATH", db_file), \
         patch("news.agent.ChatOllama", return_value=mock_model):
        removed = news_agent.judge_articles(["AI"], run_start)

    assert removed == 0
    mock_model.invoke.assert_called_once()
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
    mock_model.invoke.return_value = MagicMock(content="1: NO")

    with patch.object(news_agent, "DB_PATH", db_file), \
         patch("news.agent.ChatOllama", return_value=mock_model):
        removed = news_agent.judge_articles(["AI"], run_start)

    assert removed == 1
    mock_model.invoke.assert_called_once()
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
    mock_model.invoke.return_value = MagicMock(content="1: YES\n2: NO")

    with patch.object(news_agent, "DB_PATH", db_file), \
         patch("news.agent.ChatOllama", return_value=mock_model):
        removed = news_agent.judge_articles(["AI"], run_start)

    assert removed == 1
    mock_model.invoke.assert_called_once()
    with sqlite3.connect(db_file) as conn:
        remaining = conn.execute("SELECT title FROM news_results").fetchall()
    assert remaining == [("Good Article",)]


def test_judge_articles_batches_large_sets(tmp_path):
    from news import agent as news_agent

    db_file = str(tmp_path / "judge.db")
    news_agent.init_db(db_file)
    run_start = "2026-06-16T10:00:00+00:00"
    total = news_agent.JUDGE_BATCH_SIZE + 1
    with sqlite3.connect(db_file) as conn:
        conn.executemany(
            "INSERT INTO news_results (topic, title, url, summary, fetched_at) VALUES (?,?,?,?,?)",
            [
                ("AI", f"Article {i}", f"http://a{i}.com", "Relevant.", run_start)
                for i in range(total)
            ],
        )

    mock_model = MagicMock()
    mock_model.invoke.return_value = MagicMock(content="1: YES")

    with patch.object(news_agent, "DB_PATH", db_file), \
         patch("news.agent.ChatOllama", return_value=mock_model):
        removed = news_agent.judge_articles(["AI"], run_start)

    assert removed == 0
    assert mock_model.invoke.call_count == 2
    second_call_prompt = mock_model.invoke.call_args_list[1][0][0][1].content
    assert f"Article {news_agent.JUDGE_BATCH_SIZE}" in second_call_prompt
    assert "Article 0" not in second_call_prompt


def test_judge_articles_fail_open_on_malformed_response(tmp_path):
    from news import agent as news_agent

    db_file = str(tmp_path / "judge.db")
    news_agent.init_db(db_file)
    run_start = "2026-06-16T10:00:00+00:00"
    with sqlite3.connect(db_file) as conn:
        conn.executemany(
            "INSERT INTO news_results (topic, title, url, summary, fetched_at) VALUES (?,?,?,?,?)",
            [
                ("AI", "Good Article", "http://good.com", "Relevant.", run_start),
                ("AI", "Missing Verdict Article", "http://missing.com", "Unclear.", run_start),
            ],
        )

    mock_model = MagicMock()
    # No line for article 2 — response is malformed/incomplete.
    mock_model.invoke.return_value = MagicMock(content="1: YES\nsome unrelated text")

    with patch.object(news_agent, "DB_PATH", db_file), \
         patch("news.agent.ChatOllama", return_value=mock_model):
        removed = news_agent.judge_articles(["AI"], run_start)

    assert removed == 0
    with sqlite3.connect(db_file) as conn:
        count = conn.execute("SELECT COUNT(*) FROM news_results").fetchone()[0]
    assert count == 2


def test_judge_articles_fail_open_on_invoke_error(tmp_path):
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
    mock_model.invoke.side_effect = RuntimeError("ollama unreachable")

    with patch.object(news_agent, "DB_PATH", db_file), \
         patch("news.agent.ChatOllama", return_value=mock_model):
        removed = news_agent.judge_articles(["AI"], run_start)

    assert removed == 0
    with sqlite3.connect(db_file) as conn:
        count = conn.execute("SELECT COUNT(*) FROM news_results").fetchone()[0]
    assert count == 1


