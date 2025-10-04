#!/usr/bin/env python3
"""
RAG Abstention Evaluation (path-safe)
Reports: Unanswered%, Hallucination Risk, Coverage, Abstention Accuracy
Calls your entrypoint via: python -m Src.cli.main
"""

import sys
import json
import re
import time
import subprocess
from pathlib import Path

# =========================
# Path-safe configuration
# =========================
BASE = Path(__file__).resolve().parent        # .../Src/eval
SRC  = BASE.parent                            # .../Src
REPO = SRC.parent                             # repo root
GROUND_TRUTH_FILE = str((BASE / "ground_truth.json").resolve())
MAIN_MODULE = "Src.cli.main"

K = 3
MODES = ["bm25", "dense", "hybrid"]
HYBRID_ALPHAS = [0.3, 0.6]
DEBUG = True

# =========================
# Lightweight parsing
# =========================
def extract_answer(output: str) -> str:
    if "Product Citations" in output:
        answer_part = output.split("Product Citations")[0]
    else:
        answer_part = output
    lines = []
    for line in answer_part.splitlines():
        s = line.strip()
        if s and not s.startswith("[") and "Query intent" not in s and "LangChainDeprecation" not in s:
            lines.append(s)
    return re.sub(r"\s+", " ", " ".join(lines)).strip()

def extract_retrieved_docs(output: str):
    docs = []
    if "Product Citations" in output:
        section = output.split("Product Citations")[1]
        docs.extend(re.findall(r"row=(\d+)", section, re.IGNORECASE))
    # dedupe keep order
    seen, uniq = set(), []
    for d in docs:
        if d not in seen:
            seen.add(d); uniq.append(d)
    return uniq[:K]

# =========================
# Call your app
# =========================
def run_query(mode: str, query: str, alpha=None):
    cmd = [sys.executable, "-m", MAIN_MODULE, "--mode", mode, "--query", query]
    if mode == "hybrid" and alpha is not None:
        cmd += ["--alpha", str(alpha)]

    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO),
            capture_output=True,
            text=True,
            timeout=240
        )
        elapsed = time.time() - started

        if proc.returncode != 0:
            return {"answer":"", "retrieved_docs":[], "time":elapsed, "ok":False,
                    "error": f"app exit {proc.returncode}: {proc.stderr[:300]}"}

        out = proc.stdout
        ans = extract_answer(out)
        ret = extract_retrieved_docs(out)
        return {"answer": ans, "retrieved_docs": ret, "time": elapsed, "ok": True}
    except subprocess.TimeoutExpired:
        return {"answer":"", "retrieved_docs":[], "time":240.0, "ok":False, "error":"TIMEOUT"}
    except Exception as e:
        return {"answer":"", "retrieved_docs":[], "time":0.0, "ok":False, "error":str(e)}

# =========================
# Abstention metrics
# =========================
NO_ANSWER_PATTERNS = [
    "i don't know", "cannot answer", "no information", "not enough information",
    "unavailable", "no relevant", "no results", "no matching",
]

def is_unanswered(ans: str) -> bool:
    if not ans or not ans.strip():
        return True
    a = ans.lower().strip()
    return any(p in a for p in NO_ANSWER_PATTERNS)

def calculate_abstention_metrics(results, ground_truth):
    """
    results: list of dicts with fields: answer, retrieved_docs, ok
    ground_truth: test_cases with 'relevant_rows' ([] means unanswerable)
    """
    n = len(results)
    if n == 0:
        return {}

    unanswered = sum(1 for r in results if is_unanswered(r["answer"]))
    with_context = sum(1 for r in results if r["retrieved_docs"])

    # If ground truth marks some queries as unanswerable (relevant_rows == []),
    # abstention accuracy = proportion of correctly abstained vs incorrectly answered.
    gt_unans = [len(tc.get("relevant_rows", [])) == 0 for tc in ground_truth]
    abstain_correct = 0
    abstain_total = 0
    for r, is_gt_unans in zip(results, gt_unans):
        abstained = is_unanswered(r["answer"])
        if is_gt_unans or abstained:
            abstain_total += 1
            if (is_gt_unans and abstained) or ((not is_gt_unans) and (not abstained)):
                abstain_correct += 1

    # Hallucination risk: answered but zero retrieved docs
    answered = [r for r in results if not is_unanswered(r["answer"])]
    hallucinations = sum(1 for r in answered if not r["retrieved_docs"])
    halluc_risk = (hallucinations / len(answered)) if answered else 0.0

    return {
        "total_queries": n,
        "unanswered_pct": unanswered / n,
        "coverage": with_context / n,
        "abstention_accuracy": (abstain_correct / abstain_total) if abstain_total else 0.0,
        "hallucination_risk": halluc_risk,
    }

# =========================
# Main
# =========================
def main():
    print("=" * 80)
    print("RAG ABSTENTION EVALUATION")
    print("=" * 80)
    print("Metrics: Unanswered%, Hallucination Risk, Coverage, Abstention Accuracy")
    print("=" * 80)
    try:
        with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
            gt = json.load(f)
        test_cases = gt["test_cases"]
        print(f"\nLoaded {len(test_cases)} test cases\n")
    except FileNotFoundError:
        print(f"ERROR: {GROUND_TRUTH_FILE} not found!")
        return 1

    for mode in MODES:
        alphas = HYBRID_ALPHAS if mode == "hybrid" else [None]
        for alpha in alphas:
            tag = f"{mode}_α{alpha}" if alpha is not None else mode
            print("\n" + "=" * 80)
            print(f"MODE: {tag.upper()}")
            print("=" * 80 + "\n")

            results = []
            for tc in test_cases:
                q = tc["query"]
                print(f"Q: {q}")
                r = run_query(mode, q, alpha)
                if DEBUG and not r.get("ok", False):
                    print("  [warn]", r.get("error", "no output"))
                results.append(r)

            metrics = calculate_abstention_metrics(results, test_cases)
            if not metrics:
                print("No results produced; check main path/DB/ollama.")
                continue

            print(f"Total Queries: {metrics['total_queries']}")
            print(f"Unanswered% : {metrics['unanswered_pct']*100:.1f}%")
            print(f"Coverage    : {metrics['coverage']*100:.1f}%")
            print(f"Abst.Acc.   : {metrics['abstention_accuracy']*100:.1f}%")
            print(f"Halluc.Risk : {metrics['hallucination_risk']*100:.1f}%")

    return 0

if __name__ == "__main__":
    sys.exit(main())
