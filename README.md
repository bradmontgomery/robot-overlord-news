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
- RSS feed at `/rss`, ordered by when each article was captured

## Routes

| Route | Purpose |
|---|---|
| `/` | Article listing; accepts `?topic=`, `?date=`, `?page=` |
| `/stats` | Topic and source breakdown for the last 10 days |
| `/rss` | RSS 2.0 feed, newest first by `fetched_at`; accepts `?topic=`, `?date=`, `?limit=` (default 50, max 500) |
| `/api/present` | JSON article feed used by presentation mode |
| `/static/*` | Embedded CSS and JS |

## Setup

```bash
cd web && go build -o news_site .
NEWS_DB=/path/to/news_data.db ./news_site
```

The server listens on `http://localhost:8080`.

Common tasks are in the `Makefile`: `make build`, `make run`, `make test`, `make fmt`, `make vet`.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `NEWS_DB` | `../agent/news_data.db` | Path to the SQLite database |

## Database

The database is **written by the news agent in `agent/`** and opened read-only here
(`file:` DSN with `mode=ro` plus `PRAGMA query_only`). The schema contract is
[`schema.sql`](schema.sql); schema changes go through `init_db()` in `agent/` together with
`schema.sql` and the queries in `web/main.go`.

`topic` holds a comma-separated list of topics, split into individual labels at render time.

## Deployment

Pushes to `main` are automatically built, tested, and deployed to EC2 via GitHub Actions. See
[DEPLOYMENT.md](DEPLOYMENT.md) for infrastructure details and the secrets required by the pipeline.

`scripts/sync-db.sh` uploads the local database to the server and restarts the service.

## License

MIT
