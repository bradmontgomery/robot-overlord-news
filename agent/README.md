# News Agent

LangGraph agent that researches AI news, judges relevance, and writes articles plus a daily
summary into the SQLite database served by the web app in `../web/`.

```bash
uv sync
cp .envrc.example .envrc   # then edit
make test                  # 100% coverage required
make news                  # full run against local Ollama
```

See [`docs/news-agent.md`](docs/news-agent.md) for design and configuration, and
[`CLAUDE.md`](CLAUDE.md) for the env var reference and the schema contract.
