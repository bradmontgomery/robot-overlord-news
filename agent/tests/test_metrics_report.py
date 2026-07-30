import json

import pytest

from news import metrics_report as mr


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _llm(step="agent_step", tokens=2000, done="stop", total=3000, **extra):
    return {
        "type": "llm_call",
        "run_id": "2026-07-28T03:29:53+00:00",
        "step": step,
        "model": "qwen3.5:latest",
        "elapsed_s": 4.0,
        "input_tokens": tokens,
        "output_tokens": total - tokens,
        "total_tokens": total,
        "done_reason": done,
        **extra,
    }


def _step(step, elapsed=10.0, **extra):
    return {
        "type": "step",
        "run_id": "2026-07-28T03:29:53+00:00",
        "step": step,
        "elapsed_s": elapsed,
        **extra,
    }


@pytest.fixture
def records():
    """A miniature but structurally faithful slice of news_metrics.jsonl."""
    return [
        # Truncated at the ceiling, and a normal call, in the agent loop.
        _llm(tokens=3800, done="length", total=4096),
        _llm(tokens=1200),
        _llm(tokens=4100, done="length", total=4096),
        # judge_batch also truncates (no num_ctx); summarize_primary never does.
        _llm(step="judge_batch", tokens=3900, done="length", total=4096),
        _llm(step="summarize_primary", tokens=1800, total=2200, num_ctx=49152),
        # An oversized prompt that must land in the overflow bin.
        _llm(tokens=16000, total=16500),
        # Searches: one healthy, one failed, one with no `topics` key at all.
        _step("search_batch", 30.0, topics="anthropic, openai", success=True),
        _step("search_batch", 20.0, topics="anthropic, openai", success=False,
              error="XML syntax error on line 17 (status code: -1)"),
        _step("search_batch", 12.0, topics="rag", success=True),
        _step("search_batch", 9.0, success=True),
        _step("judge_articles", 18.0, articles_judged=10, articles_deleted=3),
        _step("summarize_day", 12.0, used_model="ministral-3:8b", success=True),
        _step("run", 700.0, topics=65, new_articles=22),
        {"type": "step", "run_id": "2026-07-28T05:00:00+00:00", "step": "run",
         "elapsed_s": 800.0, "topics": 65, "new_articles": 30},
    ]


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def test_load_records_skips_blank_and_malformed_lines(tmp_path):
    path = tmp_path / "m.jsonl"
    path.write_text('{"type": "step"}\n\n  \nnot json at all\n{"type": "llm_call"}\n')

    assert mr.load_records(str(path)) == [{"type": "step"}, {"type": "llm_call"}]


def test_load_records_on_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")

    assert mr.load_records(str(path)) == []


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def test_summarize_counts_truncation_over_all_llm_calls(records):
    k = mr.summarize(records)["kpis"]

    assert k["llm_calls"] == 6
    assert k["truncated"] == 3
    assert k["truncated_pct"] == pytest.approx(50.0)
    # Headline is scoped to the current config, not the all-time pool.
    assert k["current_trunc_pct"] == pytest.approx(100 * 2 / 4)
    assert k["current_label"] == "num_ctx 4,096 (default)"


def test_truncation_by_step_shows_num_ctx_only_where_set(records):
    by_step = {r["step"]: r for r in mr.summarize(records)["truncation_by_step"]}

    assert by_step["agent_step"]["truncated"] == 2
    assert by_step["agent_step"]["num_ctx"] is None
    assert by_step["judge_batch"]["truncated"] == 1
    # The one step that sets num_ctx is the one that never truncates.
    assert by_step["summarize_primary"]["truncated"] == 0
    assert by_step["summarize_primary"]["num_ctx"] == 49152


def test_truncation_by_step_omits_steps_with_no_calls():
    rows = mr._truncation_by_step([_llm()])

    assert [r["step"] for r in rows] == ["agent_step"]


def test_all_at_ceiling_false_when_a_truncation_misses_the_ceiling():
    records = [_llm(done="length", total=2048), _llm()]

    assert mr.summarize(records)["cohorts"][0]["all_at_ceiling"] is False


def test_token_bins_split_and_fold_the_overflow(records):
    bins = {b["lo"]: b for b in mr.summarize(records)["token_bins"]}

    assert bins[3500]["truncated"] == 1 and bins[3500]["completed"] == 0
    assert bins[1000]["completed"] == 1 and bins[1000]["pct_truncated"] == 0.0
    assert bins[1000]["label"] == "1.0k"
    # The 16k outlier folds into the capped bin instead of stretching the axis.
    assert bins[mr.BIN_CAP]["overflow"] is True
    assert bins[mr.BIN_CAP]["label"] == "5k+"


def test_step_share_is_measured_against_total_run_wall_time(records):
    share = {r["step"]: r for r in mr.summarize(records)["step_share"]}

    # search_batch totals 71s against 1500s of run wall time.
    assert share["search_batch"]["total_s"] == pytest.approx(71.0)
    assert share["search_batch"]["pct"] == pytest.approx(100 * 71 / 1500)
    assert share["summarize_day"]["total_s"] == pytest.approx(12.0)


def test_topic_rows_are_sorted_slowest_first_and_skip_untagged(records):
    topics = mr.summarize(records)["topics"]

    assert [r["topic"] for r in topics] == ["anthropic, openai", "rag"]
    assert topics[0]["median_s"] == pytest.approx(25.0)  # median of 30 and 20
    assert topics[0]["runs"] == 2
    assert topics[0]["failures"] == 1
    assert topics[1]["failures"] == 0


def test_failures_count_only_explicit_false_not_missing_success():
    """`success` is absent on most records; `not r.get(...)` would over-count."""
    records = [
        _step("search_batch", topics="a", success=False),
        _step("search_batch", topics="a", success=True),
        _step("search_batch", topics="a"),  # no `success` key — not a failure
        _llm(),  # llm_calls have no `success` either
    ]

    summary = mr.summarize(records)

    assert summary["kpis"]["failures"] == 1
    assert summary["topics"][0]["failures"] == 1


def test_summarize_rolls_up_judging_and_yield(records):
    k = mr.summarize(records)["kpis"]

    assert k["judged"] == 10
    assert k["deleted"] == 3
    assert k["runs"] == 2
    assert k["new_articles"] == 52
    assert k["median_wall_s"] == pytest.approx(750.0)


def test_run_rows_sort_oldest_first_and_label_by_time(records):
    runs = mr.summarize(records)["runs"]

    assert [r["label"] for r in runs] == ["03:29", "05:00"]
    assert runs[0]["wall_s"] == 700.0


def test_errors_are_grouped_with_the_status_suffix_stripped(records):
    errors = mr.summarize(records)["errors"]

    assert errors == [("XML syntax error on line 17", 1)]


def test_summarize_handles_no_records():
    summary = mr.summarize([])

    assert summary["kpis"]["runs"] == 0
    assert summary["kpis"]["truncated_pct"] == 0.0
    assert summary["kpis"]["current_trunc_pct"] == 0.0
    assert summary["cohorts"] == []
    assert summary["topics"] == [] and summary["runs"] == []


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def test_nice_max_rounds_up_to_a_human_number():
    assert mr._nice_max(594) == 600
    assert mr._nice_max(33) == 40
    assert mr._nice_max(0) == 1.0


def test_median_falls_back_on_empty_input():
    assert mr._median([]) == 0.0
    assert mr._median([1, 3]) == 2.0


def test_render_html_is_self_contained_and_carries_the_headline(records):
    page = mr.render_html(mr.summarize(records))

    assert page.startswith("<!doctype html>")
    assert "https://" not in page and "<script" not in page
    # The 4096 finding, scoped to the cohort it describes, and both theme scopes.
    assert "4,096 tokens" in page
    assert "Under num_ctx 4,096 (default)" in page
    assert 'prefers-color-scheme: dark' in page
    assert ':root[data-theme="dark"]' in page
    # Three charts, each with a table view for the contrast-relief rule.
    assert page.count("<svg") == 3
    assert page.count("<details>") == 3


def test_render_html_escapes_topic_text():
    records = [_step("search_batch", topics="<script>x</script>", success=True),
               _step("run", 10.0, new_articles=1)]

    page = mr.render_html(mr.summarize(records))

    assert "<script>x</script>" not in page
    assert "&lt;script&gt;" in page


def test_render_html_omits_the_ceiling_note_when_nothing_truncates():
    records = [_llm(), _step("run", 10.0, new_articles=1)]

    page = mr.render_html(mr.summarize(records))

    assert "stopped at exactly" not in page


def test_render_html_survives_a_run_with_no_topic_batches():
    records = [_llm(), _step("run", 10.0, new_articles=1)]

    assert "<svg" in mr.render_html(mr.summarize(records))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_main_writes_a_report(tmp_path, records, capsys):
    metrics = tmp_path / "m.jsonl"
    metrics.write_text("".join(json.dumps(r) + "\n" for r in records))
    out = tmp_path / "report.html"

    assert mr.main(["--metrics", str(metrics), "--out", str(out)]) == 0
    assert "<!doctype html>" in out.read_text()
    assert f"Wrote {out}" in capsys.readouterr().out


def test_main_errors_on_an_empty_metrics_file(tmp_path, capsys):
    metrics = tmp_path / "empty.jsonl"
    metrics.write_text("\n")

    assert mr.main(["--metrics", str(metrics), "--out", str(tmp_path / "o.html")]) == 1
    assert "No metrics found" in capsys.readouterr().err


def test_module_runs_as_a_script(tmp_path, records, monkeypatch):
    """Cover the __main__ guard the way `make report` exercises it."""
    import runpy
    import sys

    # runpy warns if the module is already imported; drop it so the run is clean.
    monkeypatch.delitem(sys.modules, "news.metrics_report", raising=False)
    metrics = tmp_path / "m.jsonl"
    metrics.write_text("".join(json.dumps(r) + "\n" for r in records))
    monkeypatch.setattr(
        "sys.argv",
        ["metrics_report.py", "--metrics", str(metrics), "--out", str(tmp_path / "o.html")],
    )

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("news.metrics_report", run_name="__main__")

    assert exc.value.code == 0


# ---------------------------------------------------------------------------
# cohorts — before/after a config change
# ---------------------------------------------------------------------------


def _judge(run_id, output, done, **extra):
    return _llm(step="judge_batch", tokens=1050, total=1050 + output, done=done,
                run_id=run_id, success=True, **extra)


@pytest.fixture
def two_cohorts():
    """One run per era, shaped like the real before/after.

    Pre-change: prompts capped below 4,096, responses cut off there, an unbounded
    judge runaway. Post-change: prompts above the old ceiling, nothing cut off, a
    judge runaway bounded at exactly the cap.
    """
    pre, post = "2026-07-28T03:00:00+00:00", "2026-07-30T01:00:00+00:00"
    return [
        # --- num_ctx unset: Ollama's 4096 default ---
        _llm(tokens=2000, total=2400, run_id=pre, tool_call_count=1),
        _llm(tokens=3800, total=4096, done="length", run_id=pre, tool_call_count=0),
        _judge(pre, 2900, "length"),
        {"type": "step", "run_id": pre, "step": "run", "elapsed_s": 730.0,
         "topics": 65, "new_articles": 22},
        # --- num_ctx raised to 16384 ---
        _llm(tokens=5200, total=5600, num_ctx=16384, run_id=post, tool_call_count=2),
        _llm(tokens=2000, total=2400, num_ctx=16384, run_id=post, tool_call_count=2),
        _judge(post, 256, "length", num_predict=256),
        {"type": "step", "run_id": post, "step": "run", "elapsed_s": 660.0,
         "topics": 65, "new_articles": 24},
    ]


def test_cohorts_split_on_whether_num_ctx_was_logged(two_cohorts):
    cohorts = mr.summarize(two_cohorts)["cohorts"]

    assert [c["label"] for c in cohorts] == ["num_ctx 4,096 (default)", "num_ctx 16,384"]
    assert cohorts[0]["num_ctx"] is None  # absence dates the run to the default era
    assert cohorts[1]["num_ctx"] == 16384


def test_cohorts_are_ordered_oldest_first(two_cohorts):
    cohorts = mr.summarize(two_cohorts)["cohorts"]

    assert cohorts[0]["first_run"] < cohorts[1]["first_run"]


def test_each_cohort_is_measured_only_on_its_own_runs(two_cohorts):
    pre, post = mr.summarize(two_cohorts)["cohorts"]

    # Pre-change: 1 of 2 agent calls truncated. Post-change: none of 2.
    assert (pre["agent_calls"], pre["truncated"]) == (2, 1)
    assert (post["agent_calls"], post["truncated"]) == (2, 0)
    assert pre["truncated_pct"] == 50.0
    assert post["truncated_pct"] == 0.0
    # The truncated pre-change step emitted no tool call — a wasted iteration.
    assert pre["truncated_no_tool"] == 1
    assert post["truncated_no_tool"] == 0


def test_headline_kpi_tracks_the_current_config_not_the_pool(two_cohorts):
    """Pooling would drag a working fix into a slow drift instead of a step change."""
    k = mr.summarize(two_cohorts)["kpis"]

    assert k["current_label"] == "num_ctx 16,384"
    assert k["current_trunc_pct"] == 0.0
    # The all-time pooled figure is still non-zero — that's the number we avoid.
    assert k["truncated_pct"] > 0


def test_prompts_above_the_old_ceiling_are_counted(two_cohorts):
    """The one signal that says inputs were clipped, not just outputs."""
    pre, post = mr.summarize(two_cohorts)["cohorts"]

    assert pre["prompts_over_ceiling"] == 0  # impossible under a 4096 window
    assert post["prompts_over_ceiling"] == 1


def test_capped_judge_runaways_still_count_as_runaways(two_cohorts):
    """A cap bounds a runaway's cost; it does not prevent it.

    The capped call stops at exactly num_predict, so a `>` test would report zero
    and make the cap look like a cure.
    """
    pre, post = mr.summarize(two_cohorts)["cohorts"]

    assert post["judge_runaways"] == 1
    assert pre["judge_runaways"] == 1


def test_cohort_stats_survive_a_cohort_with_no_runs():
    """agent_step records can exist for a run whose `run` record never landed."""
    stats = mr._cohort_stats([_llm()], None, {"2026-07-28T03:29:53+00:00"})

    assert stats["runs"] == 0
    assert stats["median_wall_s"] == 0.0
    assert stats["mean_tool_calls"] == 0.0


def test_render_shows_the_comparison_table_only_when_configs_differ(records, two_cohorts):
    one = mr.render_html(mr.summarize(records))
    both = mr.render_html(mr.summarize(two_cohorts))

    assert 'table class="cohort"' not in one
    assert 'table class="cohort"' in both
    assert "Before and after the context change" in both


def test_comparison_marks_direction_not_just_magnitude(two_cohorts):
    page = mr.render_html(mr.summarize(two_cohorts))

    # Falling truncation and rising prompt sizes are both improvements, and the
    # arrow carries that so the meaning survives without color.
    assert 'class="delta good">↓' in page
    assert 'class="delta good">↑' in page


def test_histograms_are_small_multiples_one_per_cohort(records, two_cohorts):
    one = mr.render_html(mr.summarize(records))
    both = mr.render_html(mr.summarize(two_cohorts))

    assert one.count("<svg") == 3
    assert both.count("<svg") == 4  # the extra panel is the second cohort
    assert both.count('class="panel-title"') == 2


def test_ceiling_note_stays_scoped_to_the_cohort_it_describes(two_cohorts):
    """A later config change must not silently delete the earlier explanation."""
    page = mr.render_html(mr.summarize(two_cohorts))

    assert "Under num_ctx 4,096 (default)" in page


def test_render_falls_back_when_there_are_no_agent_steps():
    """No agent_step records means no cohorts, but the page must still render."""
    records = [_llm(step="judge_batch"), _step("run", 10.0, new_articles=1)]

    page = mr.render_html(mr.summarize(records))

    assert page.count("<svg") == 3
    assert 'table class="cohort"' not in page
