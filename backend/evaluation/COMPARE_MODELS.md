# Comparing hallucination rates between models

This shows how to produce a concrete, presentable before/after comparison when
swapping `GENERATION_MODEL` (e.g. `gemma3:1b` → `llama3.1:8b`), using the
existing evaluation suite in `evaluation/questions.json`.

## What actually gets measured

The eval suite isn't just pass/fail on text — it checks the specific symptoms
of hallucination:

- **Answer term pass rate** — did the answer contain the facts actually present
  in the source documents (vs. inventing something else)?
- **Refusal success rate** — for the out-of-scope question (capital of France),
  did the model correctly say "not in the documents" instead of answering from
  its own general knowledge? Answering anyway is a hallucination by definition
  here, since the system is supposed to be grounded only in retrieved context.
- **Citation validity rate** — every citation `[Source N]` the model emits is
  checked against the actual chunk text it points to. If a sentence cites a
  source but the claim isn't backed by that source's content, it's logged as
  an *unsupported citation* — this is the most direct hallucination signal,
  since it catches fabricated claims dressed up with a fake-looking citation.
- **Retrieval hit rate / precision@k** — these are control variables. They
  depend on embeddings and retrieval, not generation, so they should stay
  roughly flat between runs. If they move too, something other than the model
  swap changed and the comparison isn't clean.

## Steps

1. Set the baseline model in `app/config.py`:
   ```python
   GENERATION_MODEL: str = "gemma3:1b"
   ```
   Start the backend:
   ```powershell
   uvicorn app.main:app --reload
   ```
   In another terminal, from `backend/`:
   ```powershell
   python evaluation/compare_models.py run --label gemma3-1b
   ```

2. Stop the backend. Pull the candidate model if you haven't:
   ```powershell
   ollama pull llama3.1:8b
   ```
   Change `GENERATION_MODEL` in `app/config.py` to `"llama3.1:8b"`, restart the
   backend, then run:
   ```powershell
   python evaluation/compare_models.py run --label llama3.1-8b
   ```

3. Generate the comparison report:
   ```powershell
   python evaluation/compare_models.py report --baseline gemma3-1b --candidate llama3.1-8b
   ```

This prints a markdown table to the terminal and saves it to
`evaluation/results/report_gemma3-1b_vs_llama3.1-8b.md`. That file is what you
hand to your professor/supervisor — it has:

- A metrics table with deltas (baseline vs candidate vs change)
- Which specific questions flipped from fail → pass (or regressed)
- Concrete examples of unsupported citations in each run, with the actual
  answer text, so you can show *specific hallucinated claims* side by side,
  not just a percentage

## Why this is defensible as evidence

- Same question set, same retrieval pipeline, same temperature (0.0) for both
  runs — the only variable that changes is the generation model.
- The citation-validity check isn't keyword matching on the answer; it checks
  whether the sentence making a claim actually overlaps with the content of
  the source it cites. That is a real faithfulness check, not a vibe check.
- Run each model 2-3 times if you want to report variance, since local LLM
  inference can have minor run-to-run differences even at temperature 0 on
  some backends. Average the `citation_valid_count` / `citation_checked_count`
  ratio across runs for a more stable number.
