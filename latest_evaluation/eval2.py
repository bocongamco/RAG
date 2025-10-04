#!/usr/bin/env python3
"""
RAG Abstention Evaluation
Measures: Unanswered %, Hallucination Rate, Coverage, Abstention Accuracy
"""

import sys
import json
import re
import subprocess
from collections import defaultdict

GROUND_TRUTH_FILE = "ground_truth.json"
MODES = ["bm25", "dense", "hybrid"]
HYBRID_ALPHA = 0.6

# Abstention detection patterns
ABSTENTION_PATTERNS = [
    r"i don'?t know",
    r"not sure",
    r"cannot (determine|find|answer)",
    r"no (information|data|products?) (available|found)",
    r"unable to (find|determine|answer)",
    r"insufficient (information|data|evidence)",
    r"not enough information",
    r"i'?m sorry,? (but )?i (can'?t|cannot)",
]

def is_abstention(answer):
    """Check if answer is an abstention."""
    if not answer or len(answer.strip()) < 10:
        return True
    
    answer_lower = answer.lower()
    for pattern in ABSTENTION_PATTERNS:
        if re.search(pattern, answer_lower):
            return True
    
    return False

def has_valid_citations(answer, retrieved_docs):
    """Check if answer has valid citations matching retrieved docs."""
    citations = re.findall(r'(?:row|ROW)[=:\s]+(\d+)', answer)
    
    if not citations:
        return False
    
    retrieved_set = set(str(doc_id) for doc_id in retrieved_docs)
    valid_citations = sum(1 for cite in citations if cite in retrieved_set)
    
    # At least 50% of citations must be valid
    return valid_citations / len(citations) >= 0.5

def calculate_coverage(retrieved_docs, relevant_docs):
    """Calculate retrieval coverage score."""
    if not relevant_docs:
        return 0.0
    
    retrieved_set = set(str(d) for d in retrieved_docs)
    relevant_set = set(str(d) for d in relevant_docs)
    
    overlap = len(retrieved_set & relevant_set)
    return overlap / len(relevant_set)

def extract_answer(output):
    """Extract answer from main.py output."""
    if "Product Citations" in output:
        answer_part = output.split("Product Citations")[0]
    else:
        answer_part = output
    
    lines = []
    for line in answer_part.split('\n'):
        line_stripped = line.strip()
        if (line_stripped and 
            not line_stripped.startswith('[') and 
            'Query intent' not in line and
            'LangChainDeprecation' not in line):
            lines.append(line_stripped)
    
    return ' '.join(lines).strip()[:1000]

def extract_retrieved_docs(output):
    """Extract retrieved document IDs."""
    doc_ids = []
    if "Product Citations" in output:
        citation_section = output.split("Product Citations")[1]
        matches = re.findall(r'row=(\d+)', citation_section, re.IGNORECASE)
        doc_ids.extend(matches)
    
    seen = set()
    unique_ids = []
    for doc_id in doc_ids:
        if doc_id not in seen:
            seen.add(doc_id)
            unique_ids.append(doc_id)
    
    return unique_ids[:3]

def run_query(mode, query, alpha=None):
    """Execute query and return results."""
    cmd = [sys.executable, "main.py", "--mode", mode, "--query", query]
    if mode == "hybrid" and alpha:
        cmd.extend(["--alpha", str(alpha)])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = result.stdout
        
        answer = extract_answer(output)
        retrieved_docs = extract_retrieved_docs(output)
        
        return {
            "answer": answer,
            "retrieved_docs": retrieved_docs,
            "success": bool(output)
        }
    except Exception as e:
        return {
            "answer": "",
            "retrieved_docs": [],
            "success": False,
            "error": str(e)
        }

def evaluate_abstention(test_cases, mode, alpha=None):
    """Evaluate abstention behavior for a mode."""
    results = []
    
    for test_case in test_cases:
        query = test_case["query"]
        relevant_docs = test_case["relevant_rows"]
        should_abstain = test_case.get("unanswerable", False)
        
        result = run_query(mode, query, alpha)
        
        if not result["success"]:
            continue
        
        answer = result["answer"]
        retrieved_docs = result["retrieved_docs"]
        
        # Abstention detection
        did_abstain = is_abstention(answer)
        has_citations = has_valid_citations(answer, retrieved_docs)
        coverage = calculate_coverage(retrieved_docs, relevant_docs)
        
        # Classification
        if did_abstain:
            answer_type = "abstained"
        elif not has_citations and len(answer) > 50:
            answer_type = "no_evidence"  # Potential hallucination
        elif coverage < 0.3 and not did_abstain:
            answer_type = "low_coverage"  # Answered despite poor retrieval
        else:
            answer_type = "answered"
        
        # Correctness (if labeled as unanswerable)
        if should_abstain:
            correct = did_abstain  # Should abstain
        else:
            correct = not did_abstain  # Should answer
        
        results.append({
            "query": query,
            "answer_type": answer_type,
            "did_abstain": did_abstain,
            "has_citations": has_citations,
            "coverage": coverage,
            "should_abstain": should_abstain,
            "correct": correct
        })
    
    return results

def calculate_metrics(results):
    """Calculate abstention metrics."""
    total = len(results)
    if total == 0:
        return {}
    
    # Count answer types
    abstained = sum(1 for r in results if r["did_abstain"])
    no_evidence = sum(1 for r in results if r["answer_type"] == "no_evidence")
    low_coverage = sum(1 for r in results if r["answer_type"] == "low_coverage")
    answered = sum(1 for r in results if r["answer_type"] == "answered")
    
    # Correctness
    correct_abstentions = sum(1 for r in results if r["should_abstain"] and r["did_abstain"])
    incorrect_abstentions = sum(1 for r in results if not r["should_abstain"] and r["did_abstain"])
    
    # Hallucination proxy (answered without evidence)
    potential_hallucinations = sum(1 for r in results 
                                   if r["answer_type"] in ["no_evidence", "low_coverage"])
    
    return {
        "total_queries": total,
        "unanswered_%": (abstained / total) * 100,
        "answered_%": (answered / total) * 100,
        "no_evidence_%": (no_evidence / total) * 100,
        "low_coverage_%": (low_coverage / total) * 100,
        "hallucination_risk_%": (potential_hallucinations / total) * 100,
        "avg_coverage": sum(r["coverage"] for r in results) / total,
        "abstention_accuracy": sum(r["correct"] for r in results) / total,
    }

def main():
    print("="*80)
    print("RAG ABSTENTION EVALUATION")
    print("="*80)
    print("Metrics: Unanswered%, Hallucination Risk, Coverage, Abstention Accuracy")
    print("="*80 + "\n")
    
    # Load ground truth
    with open(GROUND_TRUTH_FILE, 'r') as f:
        ground_truth = json.load(f)
    
    test_cases = ground_truth["test_cases"]
    
    # Add unanswerable labels (mark queries that should abstain)
    # For your dataset, assume all are answerable unless specified
    for tc in test_cases:
        tc.setdefault("unanswerable", False)
    
    print(f"Loaded {len(test_cases)} test cases\n")
    
    all_results = {}
    
    # Evaluate each mode
    for mode in MODES:
        alpha = HYBRID_ALPHA if mode == "hybrid" else None
        mode_name = f"{mode}_α{alpha}" if alpha else mode
        
        print(f"{'='*80}")
        print(f"MODE: {mode_name.upper()}")
        print(f"{'='*80}\n")
        
        results = evaluate_abstention(test_cases, mode, alpha)
        all_results[mode_name] = results
        
        metrics = calculate_metrics(results)
        
        print(f"Total Queries: {metrics['total_queries']}")
        print(f"Answered: {metrics['answered_%']:.1f}%")
        print(f"Unanswered (Abstained): {metrics['unanswered_%']:.1f}%")
        print(f"No Evidence Answers: {metrics['no_evidence_%']:.1f}%")
        print(f"Low Coverage Answers: {metrics['low_coverage_%']:.1f}%")
        print(f"Hallucination Risk: {metrics['hallucination_risk_%']:.1f}%")
        print(f"Avg Coverage: {metrics['avg_coverage']:.3f}")
        print(f"Abstention Accuracy: {metrics['abstention_accuracy']:.3f}\n")
        
        # Detailed breakdown
        print("Query-by-Query Breakdown:")
        print("-"*80)
        for r in results:
            status = "✓" if r["correct"] else "✗"
            print(f"{status} {r['query']:40s} | {r['answer_type']:15s} | "
                  f"Cov={r['coverage']:.2f} | Cite={'Yes' if r['has_citations'] else 'No'}")
        print()
    
    # Summary comparison
    print(f"{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}\n")
    
    for mode_name, results in all_results.items():
        metrics = calculate_metrics(results)
        print(f"[{mode_name}]")
        print(f"  Unanswered: {metrics['unanswered_%']:.1f}% | "
              f"Hallucination Risk: {metrics['hallucination_risk_%']:.1f}% | "
              f"Coverage: {metrics['avg_coverage']:.3f}")
    
    # Save results
    output = {
        mode: {"results": res, "metrics": calculate_metrics(res)}
        for mode, res in all_results.items()
    }
    
    with open("abstention_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nDetailed results saved to: abstention_results.json")

if __name__ == "__main__":
    sys.exit(main())
