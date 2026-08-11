# 🤖 Robot Overlord News

A fast, minimal web app for browsing news articles fetched by a LangChain agent and stored in
SQLite. Live at [robot-overlord.news](https://robot-overlord.news).

Server-rendered HTML, vanilla JS, no build step. Templates and static assets are embedded in the
binary, so deployment is a single file with no runtime dependencies.

## Features

- Article list with title, summary, source domain, topic pills, and published/fetched dates
- Filter by topic or date from the sidebar — filters are additive and preserved across navigation
- Daily briefing card (markdown) when a summary exists for the selected day
- Daily article histogram over the last 10 days
- `/stats` page — per-topic and per-source article counts for the last 10 days
- Client-side search with match highlighting
- Save favorite articles (stored in `localStorage`)
- Presentation mode for hands-free reading, with autoplay and a progress bar
- Light/dark theme toggle
- Paginated by fetch date, newest first

## Routes

| Route | Purpose |
|---|---|
| `/` | Article listing; accepts `?topic=`, `?date=`, `?page=` |
| `/stats` | Topic and source breakdown for the last 10 days |
| `/api/present` | JSON article feed used by presentation mode |
| `/static/*` | Embedded CSS and JS |

## Setup

```bash
go build -o news_site .
NEWS_DB=/path/to/news_data.db ./news_site
```

The server listens on `http://localhost:8080`.

Common tasks are in the `Makefile`: `make build`, `make run`, `make test`, `make fmt`, `make vet`.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `NEWS_DB` | `../langchain_simple_demo/news_data.db` | Path to the SQLite database |

## Database

The database is **written by the separate news-fetching agent** and opened read-only here
(`mode=ro` plus `PRAGMA query_only = ON`). Schema changes belong upstream, not in this repo.

```sql
CREATE TABLE news_results (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    topic        TEXT NOT NULL,
    title        TEXT,
    url          TEXT,
    summary      TEXT,
    fetched_at   TEXT NOT NULL,
    published_on TEXT,
    model_name   TEXT
);

CREATE UNIQUE INDEX idx_news_url ON news_results (url);

CREATE TABLE news_summaries (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    date       TEXT NOT NULL UNIQUE,
    summary    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    model_name TEXT
);
```

`topic` holds a comma-separated list of topics, split into individual labels at render time.

## Deployment

Pushes to `main` are automatically built, tested, and deployed to EC2 via GitHub Actions. See
[DEPLOYMENT.md](DEPLOYMENT.md) for infrastructure details and the secrets required by the pipeline.

`scripts/sync-db.sh` uploads the local database to the server and restarts the service.

## License

MIT
