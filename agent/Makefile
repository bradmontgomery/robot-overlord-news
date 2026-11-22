.PHONY: news report test

news:
	PYTHONPATH=src uv run python src/news/agent.py

report:
	PYTHONPATH=src uv run python src/news/metrics_report.py

test:
	uv run pytest --cov=src --cov-report=term-missing --cov-fail-under=100 tests/
