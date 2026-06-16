# 🤖 Robot Overlord News

A fast, minimal web app for browsing news articles fetched by a LangChain agent and stored in SQLite.

## Features

- Browse articles with title, summary, and source link
- Filter by topic or date using the sidebar
- Paginated by fetch date (newest first)
- Single binary — no runtime dependencies

## Deployment

Pushes to `main` are automatically built, tested, and deployed to EC2 via GitHub Actions. See [DEPLOYMENT.md](DEPLOYMENT.md) for infrastructure details and the secrets required by the pipeline.

## Setup

```bash
go build -o news_site .
NEWS_DB=/path/to/news_data.db ./news_site
```

The server listens on `http://localhost:8080`.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `NEWS_DB` | `../langchain_simple_demo/news_data.db` | Path to the SQLite database |

## Database Schema

```sql
CREATE TABLE news_results (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    topic      TEXT NOT NULL,
    title      TEXT,
    url        TEXT,
    summary    TEXT,
    fetched_at TEXT NOT NULL
);
```

## License

MIT
