"""
Before/after model comparison for RAGLab hallucination evaluation.

Usage:
  1. Start the backend with GENERATION_MODEL = "gemma3:1b" (or whichever baseline model).
     uvicorn app.main:app --reload
  2. Run:
     python evaluation/compare_models.py run --label gemma3-1b
  3. Stop the backend, change GENERATION_MODEL in app/config.py (e.g. "llama3.1:8b"),
     pull the model with `ollama pull llama3.1:8b`, restart the backend.
  4. Run:
     python evaluation/compare_models.py run --label llama3.1-8b
  5. Generate the comparison report:
     python evaluation/compare_models.py report --baseline gemma3-1b --candidate llama3.1-8b

Each `run` call hits POST /api/evaluation/run on the live server and saves the
full JSON response to evaluation/results/<label>.json, tagged with a timestamp.
The `report` command loads two saved runs and prints a side-by-side table of the
metrics that matter for judging hallucination, plus a markdown file you can hand
to a reviewer.
"""
import argparse
import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DEFAULT_URL = "http://127.0.0.1:8000/api/evaluation/run"


def run_eval(label: str, url: str):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Calling {url} ...")
    req = urllib.request.Request(url, data=b"{}", headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        print(f"Failed to reach backend at {url}: {e}")
        print("Make sure `uvicorn app.main:app` is running with the model you want to test.")
        sys.exit(1)

    # Tag with the model actually used, read from the first result's reproducibility info
    model_used = None
    for r in data.get("results", []):
        repro = r.get("reproducibility")
        if repro and repro.get("model"):
            model_used = repro["model"]
            break

    payload = {
        "label": label,
        "model_reported": model_used,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "data": data,
    }

    out_path = os.path.join(RESULTS_DIR, f"{label}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    summary = data["summary"]
    print(f"Saved to {out_path}")
    print(f"Model reported by server: {model_used}")
    print(f"Answer term pass: {summary['answer_term_pass_count']}/{summary['answerable_questions']}")
    print(f"Refusal success:  {summary['refusal_success_count']}/{summary['unanswerable_questions']}")
    print(f"Citation valid:   {summary.get('citation_valid_count', 0)}/{summary.get('citation_checked_count', 0)}")
    print(f"Unsupported cites:{summary.get('unsupported_citation_count', 0)}")


def pct(numerator, denominator):
    if denominator == 0:
        return 0.0
    return round(100.0 * numerator / denominator, 1)


def load_run(label: str):
    path = os.path.join(RESULTS_DIR, f"{label}.json")
    if not os.path.exists(path):
        print(f"No saved run found for label '{label}' at {path}")
        print("Run `python evaluation/compare_models.py run --label <label>` first.")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_report(baseline_label: str, candidate_label: str):
    baseline = load_run(baseline_label)
    candidate = load_run(candidate_label)

    b = baseline["data"]["summary"]
    c = candidate["data"]["summary"]

    rows = [
        ("Model", baseline.get("model_reported") or baseline_label, candidate.get("model_reported") or candidate_label, None),
        ("Answer term pass rate",
         f"{pct(b['answer_term_pass_count'], b['answerable_questions'])}%",
         f"{pct(c['answer_term_pass_count'], c['answerable_questions'])}%",
         pct(c['answer_term_pass_count'], c['answerable_questions']) - pct(b['answer_term_pass_count'], b['answerable_questions'])),
        ("Refusal success rate (correctly says 'I don't know')",
         f"{pct(b['refusal_success_count'], b['unanswerable_questions'])}%",
         f"{pct(c['refusal_success_count'], c['unanswerable_questions'])}%",
         pct(c['refusal_success_count'], c['unanswerable_questions']) - pct(b['refusal_success_count'], b['unanswerable_questions'])),
        ("Citation validity rate (claims actually backed by sources)",
         f"{pct(b.get('citation_valid_count', 0), b.get('citation_checked_count', 0))}%",
         f"{pct(c.get('citation_valid_count', 0), c.get('citation_checked_count', 0))}%",
         pct(c.get('citation_valid_count', 0), c.get('citation_checked_count', 0)) - pct(b.get('citation_valid_count', 0), b.get('citation_checked_count', 0))),
        ("Unsupported citations (raw count, lower = fewer fabricated claims)",
         str(b.get('unsupported_citation_count', 0)),
         str(c.get('unsupported_citation_count', 0)),
         c.get('unsupported_citation_count', 0) - b.get('unsupported_citation_count', 0)),
        ("Retrieval hit rate (control variable, should be ~stable)",
         f"{round(b.get('retrieval_hit_rate', 0) * 100, 1)}%",
         f"{round(c.get('retrieval_hit_rate', 0) * 100, 1)}%",
         None),
        ("Avg precision@k (control variable, should be ~stable)",
         round(b.get('avg_precision_at_k', 0), 3),
         round(c.get('avg_precision_at_k', 0), 3),
         None),
        ("Average latency (s)",
         round(b['average_latency'], 2),
         round(c['average_latency'], 2),
         round(c['average_latency'] - b['average_latency'], 2)),
    ]

    lines = []
    lines.append(f"# Hallucination comparison: {baseline_label} vs {candidate_label}\n")
    lines.append(f"- Baseline run captured: {baseline['captured_at']}")
    lines.append(f"- Candidate run captured: {candidate['captured_at']}")
    lines.append(f"- Same question set, same retrieval pipeline, same temperature (0.0) for both runs.\n")

    lines.append("| Metric | Baseline | Candidate | Delta |")
    lines.append("|---|---|---|---|")
    for name, bv, cv, delta in rows:
        delta_str = "" if delta is None else (f"+{delta}" if isinstance(delta, (int, float)) and delta > 0 else str(delta))
        lines.append(f"| {name} | {bv} | {cv} | {delta_str} |")

    # Per-question diff: cases where the candidate newly passes or regresses
    lines.append("\n## Per-question changes\n")
    b_results = {r["id"]: r for r in baseline["data"]["results"]}
    c_results = {r["id"]: r for r in candidate["data"]["results"]}

    improved, regressed, unchanged = [], [], []
    for qid in b_results:
        if qid not in c_results:
            continue
        b_pass = b_results[qid]["answer_passed"]
        c_pass = c_results[qid]["answer_passed"]
        if not b_pass and c_pass:
            improved.append(qid)
        elif b_pass and not c_pass:
            regressed.append(qid)
        else:
            unchanged.append(qid)

    lines.append(f"- Newly passing with candidate model: {improved or 'none'}")
    lines.append(f"- Regressed with candidate model: {regressed or 'none'}")
    lines.append(f"- Unchanged: {unchanged}")

    lines.append("\n## Unsupported citation examples (candidate run)\n")
    found_example = False
    for qid, r in c_results.items():
        cv = r.get("citation_validation")
        if cv and cv.get("unsupported"):
            found_example = True
            lines.append(f"- **{qid}** ({r['question']}): unsupported citations {cv['unsupported']}")
            lines.append(f"  - Answer: {r['answer']}")
    if not found_example:
        lines.append("- None found in candidate run.")

    report_text = "\n".join(lines)

    report_path = os.path.join(RESULTS_DIR, f"report_{baseline_label}_vs_{candidate_label}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nSaved markdown report to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare RAGLab generation models for hallucination rate")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run the evaluation suite against the currently running backend and save results")
    run_parser.add_argument("--label", required=True, help="Name for this run, e.g. 'gemma3-1b' or 'llama3.1-8b'")
    run_parser.add_argument("--url", default=DEFAULT_URL, help="Evaluation endpoint URL")

    report_parser = sub.add_parser("report", help="Build a comparison report from two saved runs")
    report_parser.add_argument("--baseline", required=True, help="Label of the baseline run")
    report_parser.add_argument("--candidate", required=True, help="Label of the candidate run")

    args = parser.parse_args()

    if args.command == "run":
        run_eval(args.label, args.url)
    elif args.command == "report":
        build_report(args.baseline, args.candidate)


if __name__ == "__main__":
    main()
