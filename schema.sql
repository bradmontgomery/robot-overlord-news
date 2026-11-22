-- Contract between agent/ (writer) and web/ (read-only reader).
--
-- agent/src/news/agent.py:init_db() creates and migrates this schema;
-- agent/tests/test_schema_contract.py fails if the two drift apart.
-- Change this file, init_db(), and web/main.go's queries together.

CREATE TABLE news_results (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    topic        TEXT NOT NULL,   -- comma-separated labels; web fans out with splitTopics()
    title        TEXT,
    url          TEXT,
    summary      TEXT,
    fetched_at   TEXT NOT NULL,   -- ISO-8601 UTC timestamp of the agent run
    published_on TEXT,            -- YYYY-MM-DD, may be NULL
    model_name   TEXT
);
CREATE UNIQUE INDEX idx_news_url ON news_results (url);

CREATE TABLE news_summaries (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    date       TEXT NOT NULL UNIQUE,  -- YYYY-MM-DD, matches date(fetched_at)
    summary    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    model_name TEXT
);
