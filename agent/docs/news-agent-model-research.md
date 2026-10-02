# News Agent — Model Research Notes

Evaluations of candidate models for the news agent's three LLM workloads. Each entry
records the verdict and the evidence behind it, so a model that was ruled out stays
ruled out for a stated reason.

## The three workloads

> **Note:** the relevance judge described below (chat model, `reasoning=False`, `num_predict=256`,
> `N: YES/NO` parsing) has been retired. `judge_articles` now asks a Noul decision model
> (`nimble:latest`, `TYPESAFE_DEFAULT_MODEL`) one yes/no question per article and deletes articles
> scoring below `JUDGE_THRESHOLD`. The judge-specific sections here are historical.

`src/news/agent.py` runs three distinct LLM jobs, two of which share one env var:

| Workload | Code | Model | Config | Skill required |
|---|---|---|---|---|
| ReAct search loop | `agent_node` (`agent.py:404`) | `OLLAMA_MODEL` = `qwen3.5:latest` | `num_ctx=16384` | Tool-call hygiene: short query, then `store_result` once per article, many turns |
| Relevance judge | `judge_articles` (`agent.py:438`) | **same** `OLLAMA_MODEL` | `reasoning=False`, `num_predict=256`, no `num_ctx` | Binary classification with editorial judgment ("is this AI news or just Nvidia stock news?") |
| Day summary | `summarize_day` (`agent.py:556`) | `SUMMARY_MODEL` = `ministral-3:8b` | `num_ctx=49152` | Long-context abstractive summarization, ~48k of digest in one pass |

**Key coupling:** `MODEL_NAME` (`agent.py:32`) drives both the ReAct loop *and* the judge.
You cannot swap the loop model without also swapping the judge. Any evaluation of a new
`OLLAMA_MODEL` is therefore an evaluation of two workloads at once.

**Hardware ceiling:** RTX 5070 Ti, 16303 MiB. Every verdict below is VRAM-bound first
and quality-bound second.

---

## 2026-08-10 — ornith:9b

**Verdict: maybe, on the ReAct loop only — contingent on a `reasoning=False` check that
could disqualify it outright. Not evaluated as a summary model.**

Dense 8.95B, Q4_K_M, 5.6 GB, 262,144-token native context, MIT, **post-trained on
Qwen3.5** — the same base as the current `OLLAMA_MODEL`. Ollama tags it `tools`; it is
RL-trained to open the assistant turn with a `<think>` block. Every published benchmark
is coding: SWE-Bench Verified 69.4 (vs Qwen3.5-9B's 53.2), Terminal-Bench 2.1 43.1,
NL2Repo 27.2, OpenClaw.

### Blocker to check first: does `reasoning=False` work?

The Ollama page shows a `tools` badge and **no `thinking` badge**, yet the HF card says
the assistant turn opens with `<think>…</think>` by default. `judge_articles` depends on
`reasoning=False` working. Two ways this breaks, both already described in the code's own
comments:

- **Ollama rejects `think=false`** for a model not tagged thinking-capable → exception →
  caught at `agent.py:485` → warning printed, `success=False` logged, **batch kept
  unfiltered**. Every batch. The judge silently becomes a no-op and the false positives it
  exists to catch all survive.
- **Ollama ignores it** and the `<think>` trace lands in `.content` → runaway hits the
  256-token `num_predict` cap with empty content (the bimodal failure the
  `JUDGE_NUM_PREDICT` comment at `agent.py:60` documents), or the trace restates article
  numbers and `_parse_judge_verdicts`' last-occurrence-wins rule scrambles verdicts.

One direct call reproducing the judge's model shape settles it:
`ChatOllama(model="ornith:9b", reasoning=False, num_predict=256)` on a 10-article batch,
then inspect `.content` and `response_metadata["done_reason"]`.

### If it passes, the expected gain is still small

- The ReAct loop is the only task ornith's training plausibly touches, and it is
  post-trained **on the model already in use**. The delta on "issue a short search query,
  then call `store_result` once per article" is not obviously positive.
- The judge and the summary need editorial judgment and prose, not agentic coding. The HF
  card's silence on general knowledge, summarization, and instruction-following outside
  coding is itself the evidence.
- **The 256K context is not a selling point here.** The windows are VRAM-bound, not
  native-context-bound. The real upside is smaller and concrete: 5.6 GB vs qwen3.5's
  6.6 GB frees ~1 GB for KV cache, so `NEWS_NUM_CTX` could go above 16384 on the same card.

Not a candidate for `SUMMARY_MODEL`: `ministral-3:8b` (6 GB, 256k native) is already
right-sized, and a reasoning model would fight the "no preamble" instruction in
`_SUMMARIZE_SYSTEM_PROMPT`.

---

## 2026-08-10 — Muse-Glimmer-30B-GGUF

**Verdict: no, on this hardware. Revisit only on a 24 GB+ card.**

Dense ~29.6B multimodal causal transformer, 131k+ context, Apache 2.0, sliding-window +
GQA. Strong on paper for two of the three workloads — the card explicitly lists
"LLM-as-a-judge evaluation" and "Tool use and function calling," with MCP Atlas 75.5 and
SWE-Bench Pro 51.2. Ruled out for two independent reasons:

1. **It does not fit.** Smallest quant is `muse-glimmer-30B-kquant-17gb.gguf` at
   **16.8 GB**, before any KV cache, against **16303 MiB** total. The card itself
   recommends 24 GB VRAM. This is the wall already hit by `gemma4:26b` (17 GB), which
   spilled to CPU and returned empty summaries — the reason `_invoke_summary_model` treats
   blank output as a hard failure (`agent.py:546`).
2. **It is not a drop-in even with the VRAM.** The perception encoder is a separate
   `mmproj-kquant.gguf` (1.4 GB), and Ollama's `ollama run hf.co/<repo>` direct-pull path
   does not wire up multi-file multimodal projectors the way a native library model does.
   The `dflash-kquant.gguf` speculative drafter is llama.cpp/vLLM territory, not Ollama.

---

## How to A/B a candidate model

The metrics machinery already supports this — `news_metrics.jsonl` records `model` per
call, `news_results.model_name` records the storing model, and `make report` groups by
agent config.

1. Run the `reasoning=False` check above (any reasoning-capable candidate).
2. `ollama pull <model>`, then one run with `OLLAMA_MODEL=<model>` on a fixed small topic
   list against a baseline run on the same list.
3. Compare on the metrics that discriminate:
   - fraction of `agent_step` records with `tool_call_count > 0`
   - the `done_reason` distribution (`length` vs `stop`)
   - stored-articles-per-topic grouped by `model_name`

## Open item

A `NEWS_JUDGE_MODEL` env var would let the judge and the ReAct loop be chosen
independently. Today one variable decides both, which makes any loop-model experiment a
judge experiment too.

## Sources

- [ollama.com/library/ornith](https://ollama.com/library/ornith)
- [deepreinforce-ai/Ornith-1.0-9B](https://huggingface.co/deepreinforce-ai/Ornith-1.0-9B)
- [meta-models/Muse-Glimmer-30B-GGUF](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF)
- [Run Ornith 1.0 Locally](https://codersera.com/blog/how-to-run-ornith-1-0-locally-2026/)
