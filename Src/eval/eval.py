#!/usr/bin/env python3
"""
RAG Evaluation (path-safe)
Measures: EM, F1, Precision@k, Recall@k, MRR@k, nDCG@k, Faithfulness, Attribution
Calls your entrypoint via: python -m Src.cli.main
"""

import sys
import json
import re
import math
import time
import subprocess
from pathlib import Path
from collections import defaultdict

# =========================
# Path-safe configuration
# =========================
BASE = Path(__file__).resolve().parent        # .../Src/eval
SRC  = BASE.parent                            # .../Src
REPO = SRC.parent                             # repo root
GROUND_TRUTH_FILE = str((BASE / "ground_truth.json").resolve())
MAIN_MODULE = "Src.cli.main"                  # your CLI entrypoint module

# Retrieval/eval config
K = 3
MODES = ["bm25", "dense", "hybrid"]
HYBRID_ALPHAS = [0.3, 0.6]
DEBUG = True

# =========================
# Metric helpers
# =========================
def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().strip().split())

def tokenize(text: str):
    return normalize_text(text).split()

def exact_match(prediction: str, gold_answers) -> float:
    pred_norm = normalize_text(prediction)
    for gold in gold_answers:
        if pred_norm == normalize_text(gold):
            return 1.0
    return 0.0

def f1_score(prediction: str, gold_answers) -> float:
    pred_tokens = set(tokenize(prediction))
    best = 0.0
    for gold in gold_answers:
        gold_tokens = set(tokenize(gold))
        if not pred_tokens and not gold_tokens:
            return 1.0
        if not pred_tokens or not gold_tokens:
            continue
        common = pred_tokens & gold_tokens
        if not common:
            continue
        precision = len(common) / len(pred_tokens)
        recall    = len(common) / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best

def precision_at_k(retrieved_ids, relevant_ids, k: int) -> float:
    if k == 0 or not retrieved_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    rel = set(map(str, relevant_ids))
    hits = sum(1 for d in top_k if str(d) in rel)
    return hits / len(top_k)

def recall_at_k(retrieved_ids, relevant_ids, k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = set(map(str, retrieved_ids[:k]))
    rel   = set(map(str, relevant_ids))
    hits  = len(top_k & rel)
    return hits / len(rel) if rel else 0.0

def mrr_at_k(retrieved_ids, relevant_ids, k: int) -> float:
    rel = set(map(str, relevant_ids))
    for rank, d in enumerate(retrieved_ids[:k], start=1):
        if str(d) in rel:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(retrieved_ids, relevant_ids, k: int) -> float:
    def dcg(rels):
        return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))
    rel = set(map(str, relevant_ids))
    gains = [1 if str(d) in rel else 0 for d in retrieved_ids[:k]]
    ideal = sorted(gains, reverse=True)
    dcg_s = dcg(gains)
    idcg  = dcg(ideal)
    return dcg_s / idcg if idcg > 0 else 0.0

def faithfulness_score(answer: str, contexts) -> float:
    if not contexts:
        return 0.0
    a = set(tokenize(answer))
    if not a:
        return 1.0
    c = set()
    for ctx in contexts:
        c.update(tokenize(ctx))
    return len(a & c) / len(a)

def attribution_score(answer: str, retrieved_ids) -> float:
    # Look for row=123 style citations in the answer and check if they’re in retrieved_ids
    cites = re.findall(r'(?:row|ROW)[=:\s]+(\d+)', answer)
    if not cites:
        return 1.0
    retrieved = set(map(str, retrieved_ids))
    good = sum(1 for cid in cites if cid in retrieved)
    return good / len(cites)

# =========================
# Output parsing
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
    return re.sub(r"\s+", " ", " ".join(lines)).strip()[:1000]

def extract_retrieved_docs(output: str):
    docs = []
    if "Product Citations" in output:
        section = output.split("Product Citations")[1]
        docs.extend(re.findall(r"row=(\d+)", section, re.IGNORECASE))
    seen, uniq = set(), []
    for d in docs:
        if d not in seen:
            seen.add(d); uniq.append(d)
    return uniq[:K]

def extract_contexts(output: str):
    ctxs = []
    if "Product Citations" in output:
        section = output.split("Product Citations")[1]
        ctxs.extend(re.findall(r'name="([^"]+)"', section, re.IGNORECASE))
        if not ctxs:
            for line in section.splitlines()[:5]:
                if "row=" in line.lower() and len(line.strip()) > 20:
                    clean = re.sub(r"row=\d+,?\s*", "", line, flags=re.IGNORECASE).strip()
                    if clean:
                        ctxs.append(clean)
    return ctxs[:K]

# =========================
# Running your app
# =========================
def run_query(mode: str, query: str, alpha=None):
    """
    Call your app via: python -m Src.cli.main --mode ... --query ...
    We run with cwd=REPO so any relative paths inside your app resolve.
    """
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
            return {
                "answer": "", "retrieved_docs": [], "contexts": [],
                "time": elapsed, "success": False,
                "error": f"app exit {proc.returncode}: {proc.stderr[:300]}",
            }

        out = proc.stdout
        answer = extract_answer(out)
        retrieved = extract_retrieved_docs(out)
        contexts = extract_contexts(out)

        if DEBUG:
            print(f"\n[DEBUG] Answer length: {len(answer)} chars")
            print(f"[DEBUG] Answer preview: {answer[:150]}...")
            print(f"[DEBUG] Retrieved docs: {retrieved}")
            print(f"[DEBUG] Contexts found: {len(contexts)}")

        return {
            "answer": answer,
            "retrieved_docs": retrieved,
            "contexts": contexts,
            "time": elapsed,
            "success": bool(answer or retrieved),
        }

    except subprocess.TimeoutExpired:
        return {"answer": "", "retrieved_docs": [], "contexts": [], "time": 240.0, "success": False, "error": "TIMEOUT"}
    except Exception as e:
        return {"answer": "", "retrieved_docs": [], "contexts": [], "time": 0.0, "success": False, "error": str(e)}

# =========================
# Metric wrapper
# =========================
def compute_metrics(result, test_case):
    ans = result["answer"]
    ret = result["retrieved_docs"]
    ctx = result["contexts"]
    gold_answer = test_case.get("expected_answer", "")
    relevant = test_case.get("relevant_rows", [])
    return {
        "EM": exact_match(ans, [gold_answer]),
        "F1": f1_score(ans, [gold_answer]),
        "Precision@k": precision_at_k(ret, relevant, K),
        "Recall@k": recall_at_k(ret, relevant, K),
        "MRR@k": mrr_at_k(ret, relevant, K),
        "nDCG@k": ndcg_at_k(ret, relevant, K),
        "Faithfulness": faithfulness_score(ans, ctx),
        "Attribution": attribution_score(ans, ret),
        "Time": result["time"],
    }

# =========================
# Main
# =========================
def main():
    print("=" * 80)
    print("RAG SYSTEM EVALUATION - PATH-SAFE VERSION")
    print("=" * 80)
    print(f"Metrics: EM, F1, P@{K}, R@{K}, MRR@{K}, nDCG@{K}, Faithfulness, Attribution")
    print("=" * 80)

    try:
        with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
            gt = json.load(f)
        tests = gt["test_cases"]
        print(f"\nLoaded {len(tests)} test cases from {GROUND_TRUTH_FILE}\n")
    except FileNotFoundError:
        print(f"\nERROR: {GROUND_TRUTH_FILE} not found!")
        return 1

    all_results = defaultdict(list)

    for mode in MODES:
        alphas = HYBRID_ALPHAS if mode == "hybrid" else [None]
        for alpha in alphas:
            mode_name = f"{mode}_α{alpha}" if alpha is not None else mode
            print("=" * 80)
            print(f"MODE: {mode_name.upper()}")
            print("=" * 80 + "\n")

            for tc in tests:
                query = tc["query"]
                category = tc.get("category", "n/a")
                print(f"Query: '{query}' ({category})")
                print("-" * 80)

                result = run_query(mode, query, alpha)

                if not result["success"]:
                    print(f"  ❌ FAILED: {result.get('error', 'No results')}\n")
                    continue

                metrics = compute_metrics(result, tc)
                metrics["query"] = query
                metrics["category"] = category
                all_results[mode_name].append(metrics)

                print(
                    f"  EM: {metrics['EM']:.3f} | F1: {metrics['F1']:.3f} | "
                    f"P@{K}: {metrics['Precision@k']:.3f} | R@{K}: {metrics['Recall@k']:.3f}"
                )
                print(
                    f"  MRR@{K}: {metrics['MRR@k']:.3f} | nDCG@{K}: {metrics['nDCG@k']:.3f} | "
                    f"Faith: {metrics['Faithfulness']:.3f} | Attr: {metrics['Attribution']:.3f}"
                )
                print(f"  Time: {metrics['Time']:.2f}s\n")

    print("=" * 80)
    print("SUMMARY (Averaged Across All Queries)")
    print("=" * 80 + "\n")

    for mode_name, results in all_results.items():
        if not results:
            continue
        avg = {
            m: sum(r[m] for r in results) / len(results)
            for m in [
                "EM", "F1", "Precision@k", "Recall@k",
                "MRR@k", "nDCG@k", "Faithfulness", "Attribution", "Time",
            ]
        }
        print(f"[{mode_name}]")
        print(
            f"  EM {avg['EM']:.3f} | F1 {avg['F1']:.3f} | "
            f"P@{K} {avg['Precision@k']:.3f} | R@{K} {avg['Recall@k']:.3f}"
        )
        print(
            f"  MRR@{K} {avg['MRR@k']:.3f} | nDCG@{K} {avg['nDCG@k']:.3f} | "
            f"Faith {avg['Faithfulness']:.3f} | Attr {avg['Attribution']:.3f}"
        )
        print(f"  Avg Time: {avg['Time']:.2f}s\n")

    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(dict(all_results), f, indent=2)
    print("Detailed results saved to: evaluation_results.json")
    return 0

if __name__ == "__main__":
    sys.exit(main())
