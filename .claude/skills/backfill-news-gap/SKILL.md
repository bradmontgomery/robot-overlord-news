---
name: backfill-news-gap
description: Backfill missing days in the news agent's SQLite database (news_data.db) after the news agent failed to run on one or more dates. Use whenever the user says they missed running/collecting news on specific dates, mentions gap days in the news database, or asks to "backfill" news_results or news_summaries. Re-dates articles that were actually fetched late (on a subsequent run) but were published on the missing day, then regenerates that day's summary — so the missed day stops looking empty.
---

# Backfill a missing news day

## Why this is needed

`agent/src/news/agent.py` stores each article's `fetched_at` (when the agent ran) separately
from `published_on` (when the article was actually published). If the agent doesn't run
on a given day — cron didn't fire, the box was down, whatever — no rows get `fetched_at`
on that date, and it looks like a gap.

But the *next* successful run usually still pulls articles whose `published_on` falls on
the missed day (news search results aren't perfectly same-day). This skill moves those
specific rows' `fetched_at` to match their `published_on`, so the missed day gets
populated with the news that was actually about that day, and generates its summary.

This is a data-repair operation on real history, not a simulation — treat every step as
load-bearing. Don't skip the backup or the dry-run preview to save time.

## Inputs you need from the user

The date(s) that are missing. If the user gives you a range or says "the news I collected
on day X and Y" as the source of backfill candidates, distinguish clearly between:

- **missing dates** — days with no `news_results` rows at all (what you're filling in)
- **source dates** — the day(s) whose `fetched_at` rows you'll scan for candidates (usually
  the run immediately after the gap, sometimes the next several runs if the gap is wider)

If it's ambiguous which dates are missing, or which source runs to scan, **ask** — moving
the wrong rows corrupts history and there's no clean way to detect that after the fact.

## Procedure

1. **Confirm the DB path.** Use `$NEWS_DB_PATH` if set, otherwise `news_data.db` in
   `agent/`. Until the GPU box's cron is pointed at this repo, the live file may still be in
   the old `langchain_simple_demo` checkout; if both exist, ask which one. (Note: `agent/CLAUDE.md` documents the env var's default as `news_results.db`,
   but the actual file used in this repo has always been `news_data.db` — trust what's on
   disk over the doc.)

2. **Verify the gap yourself — don't trust the user's date list blindly.** Run:

   ```sql
   SELECT date(fetched_at), COUNT(*) FROM news_results GROUP BY 1 ORDER BY 1 DESC LIMIT 20;
   ```

   Confirm the dates the user named are genuinely absent. If a "missing" date already has
   rows, stop and ask what they actually want.

3. **Back up the database before any write:**

   ```bash
   cp news_data.db news_data.db.bak-$(date +%Y-%m-%d)-preupdate
   ```

4. **Preview candidates before touching anything.** For each source date, count how many
   rows would move to each missing date:

   ```sql
   SELECT date(fetched_at), date(published_on), COUNT(*)
   FROM news_results
   WHERE date(fetched_at) IN (<source dates>)
     AND date(published_on) IN (<missing dates>)
   GROUP BY 1, 2;
   ```

   Show this breakdown to the user (or at least sanity-check it yourself) before running
   the update — it's your only chance to catch a wrong date range before it's applied.

5. **Run the update.** `fetched_at` is a full ISO-8601 timestamp
   (`2026-09-22T04:01:04.396128+00:00`); `published_on` is date-only. Swap in the
   published date but keep the original time-of-day, so ordering within the day stays
   meaningful:

   ```sql
   UPDATE news_results
   SET fetched_at = date(published_on) || 'T' || substr(fetched_at, 12)
   WHERE date(fetched_at) IN (<source dates>)
     AND date(published_on) IN (<missing dates>);
   ```

6. **Verify.** Re-run the `GROUP BY date(fetched_at)` query and confirm each missing date
   now has rows, and total row count across the affected dates is unchanged (rows moved,
   none created or lost).

7. **Generate the summary for each newly-filled date.** Reuse the project's own
   summarization path rather than reimplementing it — it already reads `news_results` for
   a date and upserts into `news_summaries`:

   ```bash
   cd agent && PYTHONPATH=src uv run python src/news/agent.py --summarize --day YYYY-MM-DD
   ```

   Run this once per missing date.

8. **Report a short before/after table**: for each missing date, rows added and the
   `news_summaries` row now present (date, model, summary length is enough — no need to
   paste the full summary unless asked).

## Guardrails

- Never skip the backup (step 3) or the preview (step 4), even if the user says "just do
  it" — this is the one place a wrong guess about date ranges becomes permanent.
- If a date range is ambiguous (e.g. the user says "those dates" after mentioning two
  different sets of dates in the same message), ask which set they mean rather than
  guessing — it changes which rows move.
- Don't touch `url` or other columns — only `fetched_at` moves.
