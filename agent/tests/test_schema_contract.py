"""init_db() must produce exactly the schema in ../schema.sql — the web app's contract."""

import sqlite3
from pathlib import Path

from news import agent as news_agent

SCHEMA_SQL = Path(__file__).resolve().parents[2] / "schema.sql"


def _shape(conn: sqlite3.Connection) -> dict:
    """Columns (order-insensitive: migrations append) and indexes for each user table."""
    tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    shape = {}
    for t in tables:
        cols = {
            (name, ctype.upper(), notnull, pk)
            for _, name, ctype, notnull, _, pk in conn.execute(f"PRAGMA table_info({t})")
        }
        indexes = {
            (unique, tuple(c[2] for c in conn.execute(f"PRAGMA index_info({name})")))
            for _, name, unique, *_ in conn.execute(f"PRAGMA index_list({t})")
        }
        shape[t] = {"columns": cols, "indexes": indexes}
    return shape


def test_init_db_matches_schema_contract(tmp_path):
    agent_db = tmp_path / "agent.db"
    news_agent.init_db(str(agent_db))

    with sqlite3.connect(agent_db) as a, sqlite3.connect(":memory:") as c:
        c.executescript(SCHEMA_SQL.read_text())
        assert _shape(a) == _shape(c)
