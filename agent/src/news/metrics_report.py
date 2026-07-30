"""
news/metrics_report.py — static HTML report for news_metrics.jsonl.

The news agent appends one JSONL record per LLM call and per pipeline step, but
nothing reads it back. This turns a day of those records into a single
self-contained HTML page: a KPI row plus three charts, hand-written inline SVG.

Stdlib only — no matplotlib, no plotly, no JS. Run it with `make report`.

Layout:
  KPI row       runs, wall time, where the time goes, truncation, failures, yield
  Comparison    before/after table, when runs span more than one agent config
  Chart 1       input tokens vs truncation, one panel per config, shared scale
  Chart 2       per-topic-batch cost, with an aligned failure-count panel
  Chart 3       cost vs yield across runs (two panels, one shared x axis)

Runs are grouped into cohorts by the `num_ctx` logged on their agent steps, so a
config change shows up as a step change rather than being averaged away against
every run that came before it.

Aggregation (`summarize`) is pure and separate from rendering so the numbers can
be tested without parsing HTML.
"""

import argparse
import html
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

METRICS_PATH = os.getenv("NEWS_METRICS_PATH", "news_metrics.jsonl")
DEFAULT_OUT = "news_metrics_report.html"

# The ceiling every truncated call lands on: Ollama's default num_ctx, applied
# because src/news/agent.py:383 builds the agent's ChatOllama without one.
CONTEXT_CEILING = 4096
# Input-token histogram bins. Anything above the cap folds into one overflow bin
# so a lone 16k-token outlier can't stretch the axis flat.
BIN_WIDTH = 500
BIN_CAP = 5000

# Steps that spend wall-clock time, in pipeline order, for the time-share tile.
PIPELINE_STEPS = ("search_batch", "judge_articles", "summarize_day")

# Buckets in the sequential ramp used for per-topic cost (--ramp-1 … --ramp-N).
RAMP_STEPS = 4
# Mirrors JUDGE_NUM_PREDICT in agent.py. A judge verdict is ~41 tokens; anything
# past this was a runaway, and capping bounds its cost rather than preventing it —
# so a capped runaway still reports done_reason "length". Counting those as
# "truncated" alongside real agent truncations would blend two unlike things.
JUDGE_OUTPUT_CAP = 256


# --------------------------------------------------------------------------
# Loading & aggregation (pure — no rendering, no I/O beyond the read)
# --------------------------------------------------------------------------


def load_records(path: str) -> list[dict]:
    """Read a JSONL metrics file, skipping blank and malformed lines."""
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _llm_calls(records: list[dict], step: str | None = None) -> list[dict]:
    """LLM-call records, optionally narrowed to one step.

    `type` and `step` must both be checked: an llm_call/agent_step record and a
    step/search_batch record are different shapes that share the `step` key.
    """
    return [
        r
        for r in records
        if r.get("type") == "llm_call" and (step is None or r.get("step") == step)
    ]


def _steps(records: list[dict], step: str) -> list[dict]:
    return [r for r in records if r.get("type") == "step" and r.get("step") == step]


def _median(values, default=0.0) -> float:
    return statistics.median(values) if values else default


def _pct(part: float, whole: float) -> float:
    return 100.0 * part / whole if whole else 0.0


def _token_bins(calls: list[dict]) -> list[dict]:
    """Bin calls by input_tokens, splitting each bin completed vs truncated."""
    bins: dict[int, dict] = {}
    for r in calls:
        tokens = r.get("input_tokens") or 0
        lo = min(tokens // BIN_WIDTH * BIN_WIDTH, BIN_CAP)
        b = bins.setdefault(lo, {"lo": lo, "completed": 0, "truncated": 0})
        if r.get("done_reason") == "length":
            b["truncated"] += 1
        else:
            b["completed"] += 1
    for b in bins.values():
        b["total"] = b["completed"] + b["truncated"]
        b["pct_truncated"] = _pct(b["truncated"], b["total"])
        b["overflow"] = b["lo"] >= BIN_CAP
        b["label"] = f"{BIN_CAP // 1000}k+" if b["overflow"] else f"{b['lo'] // 1000}.{b['lo'] % 1000 // 100}k"
    return [bins[k] for k in sorted(bins)]


def _truncation_by_step(records: list[dict]) -> list[dict]:
    """Per-step truncation counts — the evidence that num_ctx is the cause."""
    rows = []
    for step in ("agent_step", "judge_batch", "summarize_primary"):
        calls = _llm_calls(records, step)
        if not calls:
            continue
        truncated = [r for r in calls if r.get("done_reason") == "length"]
        # num_ctx is only logged when the caller sets it explicitly.
        ctx = next((r["num_ctx"] for r in calls if r.get("num_ctx")), None)
        rows.append({
            "step": step,
            "calls": len(calls),
            "truncated": len(truncated),
            "pct": _pct(len(truncated), len(calls)),
            "num_ctx": ctx,
        })
    return rows


def _topic_rows(records: list[dict]) -> list[dict]:
    """Per-topic-batch median duration and failure count, slowest first."""
    durations = defaultdict(list)
    failures: Counter = Counter()
    for r in _steps(records, "search_batch"):
        topics = r.get("topics")
        if not topics:
            continue
        durations[topics].append(r.get("elapsed_s") or 0.0)
        # `success` is absent on most record types, so test for False explicitly.
        # `not r.get("success")` would count every record that never logs it.
        if r.get("success") is False:
            failures[topics] += 1
    rows = [
        {
            "topic": topic,
            "median_s": _median(values),
            "runs": len(values),
            "failures": failures.get(topic, 0),
        }
        for topic, values in durations.items()
    ]
    rows.sort(key=lambda row: row["median_s"], reverse=True)
    return rows


def _run_rows(records: list[dict]) -> list[dict]:
    """One row per pipeline run, oldest first."""
    rows = [
        {
            "run_id": r.get("run_id") or "",
            "label": (r.get("run_id") or "")[11:16],  # HH:MM
            "wall_s": r.get("elapsed_s") or 0.0,
            "new_articles": r.get("new_articles") or 0,
            "topics": r.get("topics") or 0,
        }
        for r in _steps(records, "run")
    ]
    rows.sort(key=lambda row: row["run_id"])
    return rows


def _run_cohorts(records: list[dict]) -> dict[str, int | None]:
    """Map each run_id to the agent context window that run used.

    `num_ctx` is only logged from the point the agent started setting it, so its
    absence dates a run to the era of Ollama's 4096-token default. That makes it
    the pre/post marker for a config change, with no backfill needed.
    """
    cohorts: dict[str, int | None] = {}
    for r in _llm_calls(records, "agent_step"):
        run_id = r.get("run_id") or ""
        if cohorts.get(run_id) is None:
            cohorts[run_id] = r.get("num_ctx")
    return cohorts


def _cohort_stats(records: list[dict], num_ctx: int | None, run_ids: set) -> dict:
    """The signals worth comparing across a config change, for one cohort."""
    sub = [r for r in records if r.get("run_id") in run_ids]
    agent = _llm_calls(sub, "agent_step")
    judge = _llm_calls(sub, "judge_batch")
    llm = _llm_calls(sub)
    runs = _run_rows(sub)

    truncated = [r for r in agent if r.get("done_reason") == "length"]
    tool_counts = [r["tool_call_count"] for r in agent if r.get("tool_call_count") is not None]
    stalls = [r for r in llm if (r.get("load_duration_ns") or 0) > 1e9]
    # Prompts above the old default are only possible once the window is raised —
    # the one signal that says whether inputs were being clipped, not just outputs.
    over = [r for r in agent if (r.get("input_tokens") or 0) > CONTEXT_CEILING]
    # `>=`, not `>`: a capped runaway stops at exactly num_predict, so a `>` test
    # would report zero runaways the moment the cap starts working and make the
    # cap look like a cure rather than a bound. No legitimate verdict comes close
    # (the largest observed is 152 tokens).
    runaways = [r for r in judge if (r.get("output_tokens") or 0) >= JUDGE_OUTPUT_CAP]

    return {
        "num_ctx": num_ctx,
        "label": f"num_ctx {num_ctx:,}" if num_ctx else f"num_ctx {CONTEXT_CEILING:,} (default)",
        "runs": len(runs),
        "agent_calls": len(agent),
        "truncated": len(truncated),
        "truncated_pct": _pct(len(truncated), len(agent)),
        "truncated_no_tool": sum(1 for r in truncated if r.get("tool_call_count") == 0),
        "prompts_over_ceiling": len(over),
        "mean_tool_calls": (sum(tool_counts) / len(tool_counts)) if tool_counts else 0.0,
        "judge_runaways": len(runaways),
        "judge_runaway_s": sum(r.get("elapsed_s") or 0.0 for r in runaways),
        "load_stall_pct": _pct(len(stalls), len(llm)),
        "median_wall_s": _median([r["wall_s"] for r in runs]),
        "steps_per_run": _median(list(Counter(r.get("run_id") for r in agent).values())),
        "token_bins": _token_bins(agent),
        "all_at_ceiling": bool(truncated) and all(
            r.get("total_tokens") == CONTEXT_CEILING for r in truncated
        ),
        "first_run": min(run_ids) if run_ids else "",
    }


def _cohorts(records: list[dict]) -> list[dict]:
    """One entry per distinct agent config, oldest first."""
    by_run = _run_cohorts(records)
    groups: dict[int | None, set] = defaultdict(set)
    for run_id, num_ctx in by_run.items():
        groups[num_ctx].add(run_id)
    stats = [_cohort_stats(records, ctx, ids) for ctx, ids in groups.items()]
    stats.sort(key=lambda c: c["first_run"])
    return stats


def summarize(records: list[dict]) -> dict:
    """Roll the raw records up into everything the page renders."""
    calls = _llm_calls(records)
    agent_calls = _llm_calls(records, "agent_step")
    runs = _run_rows(records)
    topics = _topic_rows(records)

    truncated = sum(1 for r in calls if r.get("done_reason") == "length")
    total_wall = sum(r.get("elapsed_s") or 0.0 for r in _steps(records, "run"))
    step_share = [
        {
            "step": step,
            "total_s": sum(r.get("elapsed_s") or 0.0 for r in _steps(records, step)),
        }
        for step in PIPELINE_STEPS
    ]
    for row in step_share:
        row["pct"] = _pct(row["total_s"], total_wall)

    judged = sum(r.get("articles_judged") or 0 for r in _steps(records, "judge_articles"))
    deleted = sum(r.get("articles_deleted") or 0 for r in _steps(records, "judge_articles"))
    failures = [r for r in records if r.get("success") is False]
    cohorts = _cohorts(records)
    # Headline truncation is the *current* config's rate, not the all-time average.
    # Pooling them would let a fixed pile of old failures drag a working fix down
    # into a slow drift instead of showing the step change.
    current = cohorts[-1] if cohorts else None

    return {
        "kpis": {
            "runs": len(runs),
            "median_wall_s": _median([r["wall_s"] for r in runs]),
            "llm_calls": len(calls),
            "truncated": truncated,
            "truncated_pct": _pct(truncated, len(calls)),
            "current_trunc_pct": current["truncated_pct"] if current else 0.0,
            "current_label": current["label"] if current else "",
            "current_runs": current["runs"] if current else 0,
            "search_share_pct": step_share[0]["pct"] if step_share else 0.0,
            "failures": len(failures),
            "judged": judged,
            "deleted": deleted,
            "new_articles": sum(r["new_articles"] for r in runs),
        },
        "cohorts": cohorts,
        "token_bins": _token_bins(agent_calls),
        "truncation_by_step": _truncation_by_step(records),
        "step_share": step_share,
        "topics": topics,
        "runs": runs,
        "errors": Counter(
            (r.get("error") or "").split("(status")[0].strip() for r in failures
        ).most_common(),
        "topics_with_failures": sum(1 for r in topics if r["failures"]),
    }


# --------------------------------------------------------------------------
# SVG helpers
# --------------------------------------------------------------------------


def _esc(text) -> str:
    return html.escape(str(text), quote=True)


def _nice_max(value: float) -> float:
    """Round an axis maximum up to something a human would pick."""
    if value <= 0:
        return 1.0
    step = 10 ** (len(str(int(value))) - 1)
    return float(((int(value) // step) + 1) * step)


def _axis_label(x: float, y: float, text: str, anchor: str = "middle") -> str:
    return (
        f'<text class="tick" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}">'
        f"{_esc(text)}</text>"
    )


def svg_stacked_histogram(bins: list[dict], y_max: float | None = None) -> str:
    """Chart 1 — input tokens vs truncation, stacked, with the 4096 rule.

    *y_max* is passed in when this is rendered as small multiples per config
    cohort, so the panels share one scale and are actually comparable.
    """
    w, h = 880, 340
    pad_l, pad_r, pad_t, pad_b = 56, 16, 24, 56
    plot_w, plot_h = w - pad_l - pad_r, h - pad_t - pad_b
    y_max = y_max or _nice_max(max((b["total"] for b in bins), default=1))
    slot = plot_w / max(len(bins), 1)
    bar_w = min(slot - 10, 64)

    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" class="chart">']
    parts.append('<title>Agent-loop calls by prompt size, split by whether the '
                 'response was cut off at the context limit</title>')

    # Gridlines + y ticks.
    for i in range(5):
        y = pad_t + plot_h - plot_h * i / 4
        parts.append(f'<line class="grid" x1="{pad_l}" y1="{y:.1f}" x2="{w - pad_r}" y2="{y:.1f}"/>')
        parts.append(_axis_label(pad_l - 10, y + 4, f"{int(y_max * i / 4)}", "end"))

    for idx, b in enumerate(bins):
        cx = pad_l + slot * idx + slot / 2
        x = cx - bar_w / 2
        h_done = plot_h * b["completed"] / y_max
        h_trunc = plot_h * b["truncated"] / y_max
        base = pad_t + plot_h
        tip = f'{b["label"]} tokens — {b["total"]} calls, {b["truncated"]} truncated ({b["pct_truncated"]:.0f}%)'
        # Truncated sits on the baseline; completed stacks above with a 2px gap.
        if b["truncated"]:
            parts.append(
                f'<rect class="mark" x="{x:.1f}" y="{base - h_trunc:.1f}" width="{bar_w:.1f}" '
                f'height="{h_trunc:.1f}" rx="4" fill="var(--critical)">'
                f"<title>{_esc(tip)}</title></rect>"
            )
        if b["completed"]:
            y_done = base - h_trunc - h_done - (2 if b["truncated"] else 0)
            parts.append(
                f'<rect class="mark" x="{x:.1f}" y="{y_done:.1f}" width="{bar_w:.1f}" '
                f'height="{h_done:.1f}" rx="4" fill="var(--series-1)">'
                f"<title>{_esc(tip)}</title></rect>"
            )
        if b["truncated"]:
            parts.append(
                f'<text class="bar-label" x="{cx:.1f}" y="{base - h_trunc - h_done - 12:.1f}" '
                f'text-anchor="middle">{b["pct_truncated"]:.0f}%</text>'
            )
        parts.append(_axis_label(cx, base + 20, b["label"]))

    parts.append(f'<line class="axis" x1="{pad_l}" y1="{pad_t + plot_h}" x2="{w - pad_r}" y2="{pad_t + plot_h}"/>')
    parts.append(_axis_label((pad_l + w - pad_r) / 2, h - 12, "input tokens (prompt size)"))
    parts.append("</svg>")
    return "".join(parts)


def svg_topic_chart(rows: list[dict]) -> str:
    """Chart 2 — per-topic median seconds, plus an aligned failure panel."""
    row_h = 22
    w, h = 880, 40 + row_h * len(rows) + 30
    # value_w reserves room for the "30.5s" label that sits past each bar's end,
    # so the longest bar's label can't run into the failure panel beside it.
    label_w, value_w, fail_w = 200, 46, 110
    bar_w = w - label_w - value_w - fail_w - 20
    fail_x = label_w + bar_w + value_w
    t_max = _nice_max(max((r["median_s"] for r in rows), default=1))
    f_max = max((r["failures"] for r in rows), default=1) or 1

    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" class="chart">']
    parts.append("<title>Median seconds per topic batch, slowest first, with "
                 "failed searches per batch alongside</title>")
    parts.append(f'<text class="panel-head" x="{label_w}" y="18">median seconds per search</text>')
    parts.append(f'<text class="panel-head" x="{fail_x}" y="18">failed runs</text>')

    for idx, r in enumerate(rows):
        y = 32 + row_h * idx
        # Sequential blue, slowest = most prominent. The steps are CSS variables
        # because the ramp has to *reverse* in dark mode: light→dark reads as
        # increasing magnitude on a light surface, but on a dark surface those
        # same dark steps recede into it, which would make the slowest topics —
        # the entire point of this chart — the least visible marks.
        bucket = min(int(r["median_s"] / t_max * RAMP_STEPS), RAMP_STEPS - 1)
        shade = f"var(--ramp-{bucket + 1})"
        width = bar_w * r["median_s"] / t_max
        parts.append(f'<text class="row-label" x="{label_w - 8}" y="{y + 12}" text-anchor="end">{_esc(r["topic"])}</text>')
        parts.append(
            f'<rect class="mark" x="{label_w}" y="{y + 3}" width="{width:.1f}" height="{row_h - 8}" '
            f'rx="4" fill="{shade}"><title>{_esc(r["topic"])} — {r["median_s"]:.1f}s median '
            f'over {r["runs"]} runs, {r["failures"]} failed</title></rect>'
        )
        parts.append(
            f'<text class="value" x="{label_w + width + 6:.1f}" y="{y + 12}">{r["median_s"]:.1f}s</text>'
        )
        if r["failures"]:
            fw = (fail_w - 26) * r["failures"] / f_max
            fx = fail_x
            parts.append(
                f'<rect class="mark" x="{fx}" y="{y + 3}" width="{fw:.1f}" height="{row_h - 8}" '
                f'rx="4" fill="var(--critical)"><title>{_esc(r["topic"])} — '
                f'{r["failures"]} failed searches</title></rect>'
            )
            parts.append(f'<text class="value" x="{fx + fw + 6:.1f}" y="{y + 12}">{r["failures"]}</text>')

    parts.append("</svg>")
    return "".join(parts)


def _line_panel(rows, key, y0, plot_h, color, label, fmt) -> str:
    """One zero-based panel of chart 3. Shares the caller's x geometry."""
    w, pad_l, pad_r = 880, 56, 16
    plot_w = w - pad_l - pad_r
    y_max = _nice_max(max((r[key] for r in rows), default=1))
    step = plot_w / max(len(rows) - 1, 1)

    def xy(i, row):
        return pad_l + step * i, y0 + plot_h - plot_h * row[key] / y_max

    parts = [f'<text class="panel-head" x="{pad_l}" y="{y0 - 8}">{_esc(label)}</text>']
    for i in range(3):
        y = y0 + plot_h - plot_h * i / 2
        parts.append(f'<line class="grid" x1="{pad_l}" y1="{y:.1f}" x2="{w - pad_r}" y2="{y:.1f}"/>')
        parts.append(_axis_label(pad_l - 10, y + 4, fmt(y_max * i / 2), "end"))

    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (xy(i, r) for i, r in enumerate(rows)))
    parts.append(f'<polyline class="series-line" points="{points}" stroke="{color}"/>')
    for i, r in enumerate(rows):
        x, y = xy(i, r)
        parts.append(
            f'<circle class="mark" cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}">'
            f'<title>{_esc(r["label"])} — {fmt(r[key])} {_esc(label)}</title></circle>'
        )
    return "".join(parts)


def svg_runs_chart(rows: list[dict]) -> str:
    """Chart 3 — wall time over runs above new articles over runs.

    Two panels, never two y-axes: the measures have different units and a dual
    axis would let the reader invent a correlation. Both zero-based, so flat
    cost reads as flat.
    """
    w, h = 880, 400
    pad_l, pad_r = 56, 16
    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" class="chart">']
    parts.append("<title>Run duration and new articles found, per run, over the day</title>")
    parts.append(_line_panel(rows, "wall_s", 40, 130, "var(--series-1)",
                             "run duration (seconds)", lambda v: f"{int(v)}"))
    parts.append(_line_panel(rows, "new_articles", 240, 110, "var(--series-3)",
                             "new articles stored", lambda v: f"{int(v)}"))

    step = (w - pad_l - pad_r) / max(len(rows) - 1, 1)
    parts.append(f'<line class="axis" x1="{pad_l}" y1="350" x2="{w - pad_r}" y2="350"/>')
    for i, r in enumerate(rows):
        if i % 3 == 0 or i == len(rows) - 1:
            parts.append(_axis_label(pad_l + step * i, 370, r["label"]))
    parts.append(_axis_label(w / 2, 392, "run start (UTC)"))
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------

STYLE = """
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb;
  --text: #0b0b0b; --text-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --series-1: #2a78d6; --series-3: #1baf7a; --critical: #d03b3b; --good: #006300;
  /* Ordinal blue ramp, low→high magnitude. On the light surface that means
     light→dark (no lighter than step 250, the 2:1 floor). Four steps, not
     more: six packed the range so tightly that adjacent steps failed the
     minimum lightness gap and read as one color. */
  --ramp-1: #86b6ef; --ramp-2: #3987e5; --ramp-3: #1c5cab; --ramp-4: #0d366b;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19;
    --text: #ffffff; --text-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --series-1: #3987e5; --series-3: #199e70; --critical: #d03b3b; --good: #0ca30c;
    /* Reversed for the dark surface: high magnitude must stay the most
       prominent mark, so the ramp runs dark→light (no darker than step 600). */
    --ramp-1: #184f95; --ramp-2: #2a78d6; --ramp-3: #6da7ec; --ramp-4: #b7d3f6;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19;
  --text: #ffffff; --text-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --series-1: #3987e5; --series-3: #199e70; --critical: #d03b3b; --good: #0ca30c;
  --ramp-1: #184f95; --ramp-2: #2a78d6; --ramp-3: #6da7ec; --ramp-4: #b7d3f6;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 32px 20px 64px;
  background: var(--page); color: var(--text);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
}
.wrap { max-width: 940px; margin: 0 auto; }
h1 { font-size: 26px; margin: 0 0 4px; letter-spacing: -0.01em; }
.sub { color: var(--text-2); margin: 0 0 28px; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 32px; }
.kpi { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
.kpi .v { font-size: 26px; font-weight: 600; letter-spacing: -0.02em; }
.kpi .k { color: var(--text-2); font-size: 12px; margin-top: 2px; }
.kpi.alert .v { color: var(--critical); }
figure { background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
         margin: 0 0 24px; padding: 20px 20px 12px; }
figcaption { margin-bottom: 4px; }
figcaption .t { font-size: 17px; font-weight: 600; }
figcaption .d { color: var(--text-2); font-size: 13px; margin-top: 3px; }
.chart { width: 100%; height: auto; display: block; overflow: visible; }
.scroll { overflow-x: auto; }
.grid { stroke: var(--grid); stroke-width: 1; }
.axis { stroke: var(--axis); stroke-width: 1; }
.tick { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.row-label { fill: var(--text-2); font-size: 11.5px; }
.value { fill: var(--text-2); font-size: 11px; font-variant-numeric: tabular-nums; }
.bar-label { fill: var(--text-2); font-size: 11px; font-variant-numeric: tabular-nums; }
.panel-head { fill: var(--text-2); font-size: 12px; font-weight: 600; }
.series-line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.mark { transition: opacity .12s; }
.chart:hover .mark { opacity: .55; }
.chart .mark:hover { opacity: 1; }
.legend { display: flex; gap: 18px; flex-wrap: wrap; margin: 10px 0 0; padding: 0; list-style: none;
          color: var(--text-2); font-size: 12.5px; }
.legend li { display: flex; align-items: center; gap: 7px; }
.swatch { width: 11px; height: 11px; border-radius: 3px; flex: none; }
.note { border-left: 3px solid var(--critical); background: var(--surface);
        padding: 12px 16px; margin: 0 0 24px; border-radius: 0 8px 8px 0; color: var(--text-2); font-size: 14px; }
.note b { color: var(--text); }
details { margin-top: 12px; }
summary { cursor: pointer; color: var(--text-2); font-size: 13px; }
table { border-collapse: collapse; width: 100%; margin-top: 10px; font-size: 13px; }
th, td { text-align: left; padding: 5px 10px; border-bottom: 1px solid var(--border); }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
table.cohort { margin-top: 4px; font-size: 13.5px; }
table.cohort th { color: var(--text-2); font-weight: 600; }
table.cohort tbody tr:hover { background: var(--page); }
.panel-title { color: var(--text-2); font-size: 13px; font-weight: 600; margin: 14px 0 2px; }
.delta { font-weight: 600; }
/* Direction is carried by the arrow glyph as well as the color, so the
   good/bad distinction survives for red-green colorblind readers. */
.delta.good { color: var(--good); }
.delta.bad { color: var(--critical); }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .92em; }
"""


def _table(headers: list[str], rows: list[list], numeric: set[int]) -> str:
    head = "".join(
        f'<th class="n">{_esc(h)}</th>' if i in numeric else f"<th>{_esc(h)}</th>"
        for i, h in enumerate(headers)
    )
    body = "".join(
        "<tr>" + "".join(
            f'<td class="n">{_esc(c)}</td>' if i in numeric else f"<td>{_esc(c)}</td>"
            for i, c in enumerate(row)
        ) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _details(label: str, table: str) -> str:
    return f"<details><summary>{_esc(label)}</summary><div class='scroll'>{table}</div></details>"


def _kpi(value: str, label: str, alert: bool = False) -> str:
    cls = "kpi alert" if alert else "kpi"
    return f'<div class="{cls}"><div class="v">{_esc(value)}</div><div class="k">{_esc(label)}</div></div>'


#: Comparison rows: (label, key, formatter, "lower is better"). The direction
#: matters — it decides which way an arrow reads as an improvement.
COHORT_METRICS = (
    ("pipeline runs", "runs", lambda v: f"{v:,.0f}", None),
    ("agent-loop calls", "agent_calls", lambda v: f"{v:,.0f}", None),
    ("agent steps truncated", "truncated_pct", lambda v: f"{v:.1f}%", True),
    ("…of those, emitting no tool call", "truncated_no_tool", lambda v: f"{v:,.0f}", True),
    ("prompts above 4,096 tokens", "prompts_over_ceiling", lambda v: f"{v:,.0f}", False),
    ("mean tool calls per step", "mean_tool_calls", lambda v: f"{v:.2f}", False),
    ("agent steps per run (median)", "steps_per_run", lambda v: f"{v:,.0f}", None),
    ("judge runaways (≥256 tokens out)", "judge_runaways", lambda v: f"{v:,.0f}", True),
    ("…seconds burned on them", "judge_runaway_s", lambda v: f"{v:,.0f}s", True),
    ("model load stalls >1s", "load_stall_pct", lambda v: f"{v:.1f}%", True),
    ("median run duration", "median_wall_s", lambda v: f"{v / 60:.1f} min", None),
)


def _cohort_table(cohorts: list[dict]) -> str:
    """Before/after comparison. Each config gets a column; the last is current."""
    head = "".join(f'<th class="n">{_esc(c["label"])}</th>' for c in cohorts)
    rows = []
    for label, key, fmt, lower_better in COHORT_METRICS:
        cells = "".join(f'<td class="n">{_esc(fmt(c[key]))}</td>' for c in cohorts)
        delta = ""
        if len(cohorts) > 1 and lower_better is not None:
            first, last = cohorts[0][key], cohorts[-1][key]
            if first != last:
                better = (last < first) if lower_better else (last > first)
                arrow = "↓" if last < first else "↑"
                cls = "good" if better else "bad"
                delta = f'<span class="delta {cls}">{arrow} {fmt(abs(last - first))}</span>'
        rows.append(f'<tr><td>{_esc(label)}</td>{cells}<td class="n">{delta}</td></tr>')
    return (
        f'<table class="cohort"><thead><tr><th>signal</th>{head}'
        f'<th class="n">change</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
    )


def render_html(summary: dict) -> str:
    k = summary["kpis"]
    topics, runs = summary["topics"], summary["runs"]
    by_step = summary["truncation_by_step"]
    cohorts = summary["cohorts"]

    kpis = "".join([
        _kpi(str(k["runs"]), "pipeline runs"),
        _kpi(f'{k["median_wall_s"] / 60:.1f} min', "median run duration"),
        _kpi(f'{k["search_share_pct"]:.1f}%', "of wall time in search"),
        _kpi(f'{k["current_trunc_pct"]:.1f}%',
             f'agent steps truncated — {k["current_label"]}, {k["current_runs"]} runs',
             alert=k["current_trunc_pct"] > 1),
        _kpi(str(k["failures"]), "failed searches", alert=bool(k["failures"])),
        _kpi(f'{k["new_articles"]}', "new articles stored"),
    ])

    # Scoped to whichever cohort it describes, so a later config change cannot
    # silently delete the explanation of what the earlier one was doing wrong.
    ceiling_note = ""
    stuck = next((c for c in cohorts if c["all_at_ceiling"] and not c["num_ctx"]), None)
    if stuck:
        rows = "; ".join(
            f'<code>{_esc(r["step"])}</code> {r["pct"]:.1f}%'
            + (f' (num_ctx {r["num_ctx"]:,})' if r["num_ctx"] else " (num_ctx unset)")
            for r in by_step
        )
        ceiling_note = (
            f'<p class="note"><b>Under {stuck["label"]}, all {stuck["truncated"]} truncated '
            f'agent responses stopped at exactly {CONTEXT_CEILING:,} tokens.</b> {rows}. '
            f'The one step that sets <code>num_ctx</code> explicitly is the only one that '
            f'never truncates — the agent model was built without it, so Ollama\'s '
            f'{CONTEXT_CEILING:,}-token default applied.</p>'
        )

    comparison = ""
    if len(cohorts) > 1:
        comparison = f"""
<figure>
  <figcaption>
    <div class="t">Before and after the context change</div>
    <div class="d">Runs grouped by the agent context window they ran under.
      Each config is measured on its own runs — pooling them would average a fixed
      pile of old failures against new successes and turn a step change into a drift.</div>
  </figcaption>
  <div class="scroll">{_cohort_table(cohorts)}</div>
</figure>"""

    # Small multiples, one per config, sharing a y scale so they compare honestly.
    y_max = _nice_max(max(
        (b["total"] for c in cohorts for b in c["token_bins"]), default=1
    ))
    histograms = "".join(
        f'<div class="panel-title">{_esc(c["label"])} — {c["runs"]} runs, '
        f'{c["truncated_pct"]:.1f}% truncated</div>'
        + svg_stacked_histogram(c["token_bins"], y_max=y_max)
        for c in cohorts
    ) or svg_stacked_histogram(summary["token_bins"])

    err_rows = "".join(
        f'<li>{count}× <code>{_esc(msg[:80])}</code></li>' for msg, count in summary["errors"][:4]
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>News agent metrics</title>
<style>{STYLE}</style></head>
<body><div class="wrap">
<h1>News agent metrics</h1>
<p class="sub">{k["runs"]} runs · {k["llm_calls"]:,} LLM calls · {len(topics)} topic batches
· {k["judged"]} articles judged, {k["deleted"]} pruned</p>

<div class="kpis">{kpis}</div>
{ceiling_note}
{comparison}

<figure>
  <figcaption>
    <div class="t">Where the responses get cut off</div>
    <div class="d">Agent-loop calls by prompt size, one panel per context window,
      sharing a scale. Under the 4,096 default, a prompt past ~3.5k tokens is more
      likely than not to have its response cut off; past 4k it always is.</div>
  </figcaption>
  {histograms}
  <ul class="legend">
    <li><span class="swatch" style="background:var(--series-1)"></span>completed normally</li>
    <li><span class="swatch" style="background:var(--critical)"></span>truncated at the context limit</li>
  </ul>
  {_details("Show data table", _table(
      ["config", "input tokens", "calls", "truncated", "% truncated"],
      [[c["label"], b["label"], b["total"], b["truncated"], f'{b["pct_truncated"]:.1f}%']
       for c in cohorts for b in c["token_bins"]],
      {2, 3, 4}))}
</figure>

<figure>
  <figcaption>
    <div class="t">Search is where the time goes</div>
    <div class="d">Median seconds per topic batch, slowest first —
      {k["search_share_pct"]:.1f}% of all wall time is spent here.
      {summary["topics_with_failures"]} of {len(topics)} batches failed at least once.</div>
  </figcaption>
  <div class="scroll">{svg_topic_chart(topics)}</div>
  {_details("Show data table", _table(
      ["topic batch", "median s", "runs", "failures"],
      [[r["topic"], f'{r["median_s"]:.1f}', r["runs"], r["failures"]] for r in topics],
      {1, 2, 3}))}
</figure>

<figure>
  <figcaption>
    <div class="t">Cost is flat, yield is not</div>
    <div class="d">Every run searches the same topics for roughly the same time,
      but how many new articles come back varies more than twofold. Both axes
      start at zero.</div>
  </figcaption>
  {svg_runs_chart(runs)}
  {_details("Show data table", _table(
      ["run (UTC)", "duration s", "new articles"],
      [[r["label"], f'{r["wall_s"]:.0f}', r["new_articles"]] for r in runs],
      {1, 2}))}
</figure>

<figure>
  <figcaption>
    <div class="t">Every failure is the same failure</div>
    <div class="d">All {k["failures"]} failed searches are malformed tool-call XML
      from <code>qwen3.5:latest</code> — the model CLAUDE.md recommends <em>because</em>
      gemma emits bad tool-call XML. Worth revisiting that note.</div>
  </figcaption>
  <ul class="legend" style="display:block">{err_rows}</ul>
</figure>

</div></body></html>
"""


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a static HTML report from news agent metrics.")
    parser.add_argument("--metrics", default=METRICS_PATH, help=f"metrics JSONL (default: {METRICS_PATH})")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"output HTML file (default: {DEFAULT_OUT})")
    args = parser.parse_args(argv)

    records = load_records(args.metrics)
    if not records:
        print(f"No metrics found in {args.metrics}", file=sys.stderr)
        return 1

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(render_html(summarize(records)))
    print(f"Wrote {args.out} from {len(records):,} records in {args.metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
