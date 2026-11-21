# Robot Overlord News

Full stack for https://robot-overlord.news, as two decoupled components in different languages:

- **`web/`**: Go web app that renders the articles. Deployed to EC2.
- **`agent/`**: Python/LangGraph agent that collects them. Runs on a GPU box. Has its own
  `agent/CLAUDE.md`; read that before touching agent code.

They share no code or build tooling. Their only interface is the SQLite file, and its shape is
defined in `schema.sql`.

The design goals below apply to `web/`.

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
| `web/main.go` | Entire web app: handlers, SQL, template funcs. ~600 lines, intentionally one file. |
| `web/templates/` | `base.html` (shell), `index.html`, `stats.html`, `histogram.html` (shared partial). |
| `web/static/` | `style.css` + vanilla JS files (favorites, theme, theme-init, presentation). |
| `web/static/meta/` | robots/humans/ai/llms/security.txt, served from the site root by `serveMeta` (main.go). |
| `agent/` | News agent: own `pyproject.toml`/`uv.lock`, `src/news/`, `tests/` (100% coverage), `docs/`. |
| `schema.sql` | DB contract between agent (writer) and web (reader). |
| `Makefile` | Thin delegator to `web/` and `agent/` Makefiles, plus `infra-deploy`. |
| `infrastructure/` | CloudFormation, systemd unit, nginx config. |
| `scripts/sync-db.sh` | Rsyncs the local DB to prod and restarts the service. |

## Database — Agent Writes, Web Reads

- **Schema owner: `agent/`.** `init_db()` in `agent/src/news/agent.py` creates and migrates it.
  `schema.sql` is the contract, and `agent/tests/test_schema_contract.py` fails if the two drift.
- **Web is read-only.** It opens the DB with `mode=ro`, `PRAGMA query_only = ON`, and
  `SetMaxOpenConns(1)` (`main()` in web/main.go). The DSN needs the `file:` prefix, or
  modernc.org/sqlite silently ignores `mode=ro`. Never put migrations, `CREATE TABLE`, or
  INSERT/UPDATE in `web/`.
- **Schema changes go in one PR** that updates `init_db()`, `schema.sql`, and the queries in
  `web/main.go` together. Deploy order matters: additive columns can ship agent-first. A removal
  or rename has to ship web-first.
- Env var names differ: web reads `NEWS_DB`, agent writes `NEWS_DB_PATH`.

`topic` holds a comma-separated list; `splitTopics()` fans it out, and the topic filter matches with
a `',' || topic || ','` LIKE trick rather than a join.

## Embedded Assets

`//go:embed templates/* static/*` (top of web/main.go) is what makes the single-file deploy work.

- **Editing a template or CSS file requires `make build`** — nothing hot-reloads. Restart the server
  after every frontend change.
- No CDN links, no external fonts, no remote scripts. Everything ships inside the binary.
- New assets must live under `web/templates/` or `web/static/` to be embedded at all.

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

- Branch, then merge to `main`. **Merging changes under `web/` to `main` deploys to production**
  via `.github/workflows/web.yml` (rsync binary + `systemctl restart news`). PRs touching `web/`
  get a build check.
- `.github/workflows/agent.yml` runs the agent tests on changes to `agent/` or `schema.sql`.
  It never deploys.
- From the root: `make build` / `make run` / `make fmt` / `make vet` (web); `make news` /
  `make report` / `make agent-test` (agent).
- Deploy tasks have dedicated skills: `robot-overlord-news-deploy` (app) and
  `robot-overlord-news-deploy-db` (database). Prefer those over ad-hoc ssh/rsync.
- Infrastructure details live in `DEPLOYMENT.md`; host, IP, and key names live in `.envrc`. Don't
  duplicate those values into other files — the elastic IP is already in more places than it should
  be.

## Known Issues

- **`web/` has no tests.** CI runs `go test ./...`, which passes vacuously. `agent/` does have a
  real suite at 100% coverage. Don't assume web has any coverage.
- No package-level mutable state. `topicColor` is deliberately uncached — it's a pure hash, and a
  memo map would need locking to stay safe under concurrent requests. Don't reintroduce one.
