# Robot Overlord News

Go web app that renders news articles collected by a separate LangChain agent. Live at
https://robot-overlord.news.

## Design Goals

These are the review criteria. Changes that violate them get pushed back on, even if they work.

- **Simple UI** — server-rendered HTML, vanilla JS, hand-written CSS. No frontend framework, no
  build step, no npm.
- **Single-file deploy** — one static Go binary with everything embedded. New runtime dependencies
  need a reason.
- **Skimmable** — dense, scannable article lists. Favor fewer words and tighter layout over more
  chrome.

## Layout

| Path | Contents |
|---|---|
| `main.go` | Entire app: handlers, SQL, template funcs. ~600 lines, intentionally one file. |
| `templates/` | `base.html` (shell), `index.html`, `stats.html`, `histogram.html` (shared partial). |
| `static/` | `style.css` + three vanilla JS files (favorites, theme, presentation). |
| `infrastructure/` | CloudFormation, systemd unit, nginx config. |
| `scripts/sync-db.sh` | Rsyncs the local DB to prod and restarts the service. |

## Database — Read Only

The SQLite DB is **owned by the upstream LangChain agent repo**, not this one.

- Opened `mode=ro` with `PRAGMA query_only = ON` and `SetMaxOpenConns(1)` (main.go:76–86).
- Never write migrations, `CREATE TABLE`, or any INSERT/UPDATE here. Schema changes happen upstream.
- Local path comes from `NEWS_DB` (see `.envrc`).

Actual schema:

```sql
CREATE TABLE news_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL, title TEXT, url TEXT, summary TEXT,
    fetched_at TEXT NOT NULL, published_on TEXT, model_name TEXT);
CREATE UNIQUE INDEX idx_news_url ON news_results (url);

CREATE TABLE news_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL UNIQUE, summary TEXT NOT NULL,
    created_at TEXT NOT NULL, model_name TEXT);
```

`topic` holds a comma-separated list; `splitTopics()` fans it out, and the topic filter matches with
a `',' || topic || ','` LIKE trick rather than a join.

## Embedded Assets

`//go:embed templates/* static/*` (main.go:23) is what makes the single-file deploy work.

- **Editing a template or CSS file requires `make build`** — nothing hot-reloads. Restart the server
  after every frontend change.
- No CDN links, no external fonts, no remote scripts. Everything ships inside the binary.
- New assets must live under `templates/` or `static/` to be embedded at all.

## CSS: The `min-width: 0` Rule

Two recent commits (`56f5c9b`, `5394cc1`) fixed the *same* bug in different places: a flex or grid
child holding text refuses to shrink below its content width, so long labels force the whole page
wider than the viewport on mobile.

- Any new flex/grid child that contains text needs `min-width: 0`.
- `overflow: hidden` + `text-overflow: ellipsis` alone does **not** fix it.
- Verify new layout at ~375px width before considering it done.

Theming is CSS custom properties on `:root`, overridden by `:root[data-theme="light"]`. Add new
colors as variables in both blocks — never hardcode a hex outside `:root`.

## Workflow

- Branch, then merge to `main`. **Merging to `main` deploys to production** via
  `.github/workflows/deploy.yml` (rsync binary + `systemctl restart news`).
- `make build` / `make run` / `make fmt` / `make vet`.
- Deploy tasks have dedicated skills: `robot-overlord-news-deploy` (app) and
  `robot-overlord-news-deploy-db` (database). Prefer those over ad-hoc ssh/rsync.
- Infrastructure details live in `DEPLOYMENT.md`; host, IP, and key names live in `.envrc`. Don't
  duplicate those values into other files — the elastic IP is already in more places than it should
  be.

## Known Issues

- **No tests exist.** CI runs `go test ./...`, which passes vacuously and gates nothing. Don't
  assume coverage.
- No package-level mutable state. `topicColor` is deliberately uncached — it's a pure hash, and a
  memo map would need locking to stay safe under concurrent requests. Don't reintroduce one.
