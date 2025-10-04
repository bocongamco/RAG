#!/usr/bin/env python3
"""
Complete RAG Evaluation System - FIXED VERSION
Measures: EM, F1, Precision@k, Recall@k, MRR@k, nDCG@k, Faithfulness, Attribution
"""

import sys
import json
import re
import math
import time
import subprocess
from pathlib import Path
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

GROUND_TRUTH_FILE = "ground_truth.json"
MODES = ["bm25", "dense", "hybrid"]
HYBRID_ALPHAS = [0.3, 0.6]
K = 3
DEBUG = True  # Set to False to hide debug output

# ============================================================================
# METRIC FUNCTIONS
# ============================================================================

def normalize_text(text):
    """Lowercase and remove extra whitespace."""
    return " ".join(str(text).lower().strip().split())

def tokenize(text):
    """Simple word tokenization."""
    return normalize_text(text).split()

def exact_match(prediction, gold_answers):
    """Exact Match (EM): 1.0 if prediction matches any gold answer exactly."""
    pred_norm = normalize_text(prediction)
    for gold in gold_answers:
        if pred_norm == normalize_text(gold):
            return 1.0
    return 0.0

def f1_score(prediction, gold_answers):
    """F1 Score: Token overlap between prediction and best matching gold answer."""
    pred_tokens = set(tokenize(prediction))
    
    best_f1 = 0.0
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
        recall = len(common) / len(gold_tokens)
        f1 = 2 * (precision * recall) / (precision + recall)
        best_f1 = max(best_f1, f1)
    
    return best_f1

def precision_at_k(retrieved_ids, relevant_ids, k):
    """Precision@k: Fraction of top-k results that are relevant."""
    if k == 0 or not retrieved_ids:
        return 0.0
    
    top_k = retrieved_ids[:k]
    relevant_set = set(str(doc_id) for doc_id in relevant_ids)
    hits = sum(1 for doc_id in top_k if str(doc_id) in relevant_set)
    return hits / len(top_k)

def recall_at_k(retrieved_ids, relevant_ids, k):
    """Recall@k: Fraction of relevant docs found in top-k results."""
    if not relevant_ids:
        return 0.0
    
    top_k = set(str(doc_id) for doc_id in retrieved_ids[:k])
    relevant_set = set(str(doc_id) for doc_id in relevant_ids)
    
    hits = len(top_k & relevant_set)
    return hits / len(relevant_set)

def mrr_at_k(retrieved_ids, relevant_ids, k):
    """MRR@k (Mean Reciprocal Rank): 1/rank of first relevant document."""
    relevant_set = set(str(doc_id) for doc_id in relevant_ids)
    
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if str(doc_id) in relevant_set:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(retrieved_ids, relevant_ids, k):
    """nDCG@k (Normalized Discounted Cumulative Gain): Ranking quality metric."""
    def dcg(relevances):
        return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))
    
    relevant_set = set(str(doc_id) for doc_id in relevant_ids)
    
    relevances = [1 if str(doc_id) in relevant_set else 0 
                  for doc_id in retrieved_ids[:k]]
    ideal_relevances = sorted(relevances, reverse=True)
    
    dcg_score = dcg(relevances)
    idcg_score = dcg(ideal_relevances)
    
    return dcg_score / idcg_score if idcg_score > 0 else 0.0

def faithfulness_score(answer, contexts):
    """Faithfulness: Fraction of answer tokens that appear in retrieved contexts."""
    if not contexts:
        return 0.0
    
    answer_tokens = set(tokenize(answer))
    if not answer_tokens:
        return 1.0
    
    context_tokens = set()
    for ctx in contexts:
        context_tokens.update(tokenize(ctx))
    
    overlap = len(answer_tokens & context_tokens)
    return overlap / len(answer_tokens)

def attribution_score(answer, retrieved_ids):
    """Attribution: Check if citations in answer match retrieved documents."""
    citations = re.findall(r'(?:row|ROW)[=:\s]+(\d+)', answer)
    
    if not citations:
        return 1.0
    
    retrieved_set = set(str(doc_id) for doc_id in retrieved_ids)
    valid_citations = sum(1 for cite in citations if cite in retrieved_set)
    
    return valid_citations / len(citations)

# ============================================================================
# OUTPUT PARSING - FIXED
# ============================================================================

def extract_answer(output):
    """Extract generated answer from main.py output - IMPROVED VERSION."""
    
    # Split by Product Citations to get just the answer part
    if "Product Citations" in output:
        answer_part = output.split("Product Citations")[0]
    else:
        answer_part = output
    
    # Remove system messages
    lines = []
    for line in answer_part.split('\n'):
        line_stripped = line.strip()
        # Skip empty lines, system messages, warnings
        if (line_stripped and 
            not line_stripped.startswith('[') and 
            'Query intent' not in line and
            'LangChainDeprecation' not in line and
            'D:\\Data science' not in line):
            lines.append(line_stripped)
    
    answer = ' '.join(lines)
    
    # Clean up common artifacts
    answer = re.sub(r'\s+', ' ', answer)  # Multiple spaces
    answer = answer.strip()
    
    return answer[:1000]  # Limit to 1000 chars

def extract_retrieved_docs(output):
    """Extract retrieved document IDs from output - IMPROVED VERSION."""
    doc_ids = []
    
    # Look in Product Citations section
    if "Product Citations" in output:
        citation_section = output.split("Product Citations")[1]
        # Extract row numbers
        matches = re.findall(r'row=(\d+)', citation_section, re.IGNORECASE)
        doc_ids.extend(matches)
    
    # Deduplicate while preserving order
    seen = set()
    unique_ids = []
    for doc_id in doc_ids:
        if doc_id not in seen:
            seen.add(doc_id)
            unique_ids.append(doc_id)
    
    return unique_ids[:K]

def extract_contexts(output):
    """Extract product names/descriptions as context - IMPROVED VERSION."""
    contexts = []
    
    # Look in Product Citations section
    if "Product Citations" in output:
        citation_section = output.split("Product Citations")[1]
        
        # Pattern 1: name="..."
        names = re.findall(r'name="([^"]+)"', citation_section, re.IGNORECASE)
        contexts.extend(names)
        
        # Pattern 2: Product names without quotes (fallback)
        if not contexts:
            # Each line in citations might be a product
            for line in citation_section.split('\n')[:5]:
                if 'row=' in line.lower() and len(line.strip()) > 20:
                    # Extract text between row= and end of line
                    clean_line = re.sub(r'row=\d+,?\s*', '', line, flags=re.IGNORECASE)
                    clean_line = clean_line.strip()
                    if clean_line:
                        contexts.append(clean_line)
    
    return contexts[:K]

# ============================================================================
# QUERY EXECUTION
# ============================================================================

def run_query(mode, query, alpha=None):
    """Execute query using main.py and return metrics."""
    cmd = [sys.executable, "main.py", "--mode", mode, "--query", query]
    if mode == "hybrid" and alpha:
        cmd.extend(["--alpha", str(alpha)])
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=240  # Increased timeout
        )
        elapsed = time.time() - start_time
        
        output = result.stdout
        
        # Parse output
        answer = extract_answer(output)
        retrieved_docs = extract_retrieved_docs(output)
        contexts = extract_contexts(output)
        
        # Debug output
        if DEBUG:
            print(f"\n[DEBUG] Answer length: {len(answer)} chars")
            print(f"[DEBUG] Answer preview: {answer[:150]}...")
            print(f"[DEBUG] Retrieved docs: {retrieved_docs}")
            print(f"[DEBUG] Contexts found: {len(contexts)}")
        
        return {
            "answer": answer,
            "retrieved_docs": retrieved_docs,
            "contexts": contexts,
            "time": elapsed,
            "success": bool(answer and retrieved_docs)
        }
    
    except subprocess.TimeoutExpired:
        return {
            "answer": "",
            "retrieved_docs": [],
            "contexts": [],
            "time": 120.0,
            "success": False,
            "error": "TIMEOUT"
        }
    except Exception as e:
        return {
            "answer": "",
            "retrieved_docs": [],
            "contexts": [],
            "time": 0.0,
            "success": False,
            "error": str(e)
        }

def compute_metrics(result, test_case):
    """Compute all metrics for a query result."""
    answer = result["answer"]
    retrieved_docs = result["retrieved_docs"]
    contexts = result["contexts"]
    
    expected_answer = test_case["expected_answer"]
    relevant_docs = test_case["relevant_rows"]
    
    return {
        "EM": exact_match(answer, [expected_answer]),
        "F1": f1_score(answer, [expected_answer]),
        "Precision@k": precision_at_k(retrieved_docs, relevant_docs, K),
        "Recall@k": recall_at_k(retrieved_docs, relevant_docs, K),
        "MRR@k": mrr_at_k(retrieved_docs, relevant_docs, K),
        "nDCG@k": ndcg_at_k(retrieved_docs, relevant_docs, K),
        "Faithfulness": faithfulness_score(answer, contexts),
        "Attribution": attribution_score(answer, retrieved_docs),
        "Time": result["time"]
    }

# ============================================================================
# MAIN EVALUATION
# ============================================================================

def main():
    print("="*80)
    print("RAG SYSTEM EVALUATION - FIXED VERSION")
    print("="*80)
    print(f"Metrics: EM, F1, P@{K}, R@{K}, MRR@{K}, nDCG@{K}, Faithfulness, Attribution")
    print("="*80)
    
    # Load ground truth
    try:
        with open(GROUND_TRUTH_FILE, 'r') as f:
            ground_truth = json.load(f)
        test_cases = ground_truth["test_cases"]
        print(f"\nLoaded {len(test_cases)} test cases from {GROUND_TRUTH_FILE}\n")
    except FileNotFoundError:
        print(f"\nERROR: {GROUND_TRUTH_FILE} not found!")
        return 1
    
    # Results storage
    all_results = defaultdict(list)
    
    # Run evaluation
    for mode in MODES:
        alphas = HYBRID_ALPHAS if mode == "hybrid" else [None]
        
        for alpha in alphas:
            mode_name = f"{mode}_α{alpha}" if alpha else mode
            
            print(f"{'='*80}")
            print(f"MODE: {mode_name.upper()}")
            print(f"{'='*80}\n")
            
            for test_case in test_cases:
                query = test_case["query"]
                category = test_case["category"]
                
                print(f"Query: '{query}' ({category})")
                print("-"*80)
                
                # Execute query
                result = run_query(mode, query, alpha)
                
                if not result["success"]:
                    print(f"  ❌ FAILED: {result.get('error', 'No results')}\n")
                    continue
                
                # Compute metrics
                metrics = compute_metrics(result, test_case)
                metrics["query"] = query
                metrics["category"] = category
                all_results[mode_name].append(metrics)
                
                # Display metrics
                print(f"  EM: {metrics['EM']:.3f} | F1: {metrics['F1']:.3f} | "
                      f"P@{K}: {metrics['Precision@k']:.3f} | R@{K}: {metrics['Recall@k']:.3f}")
                print(f"  MRR@{K}: {metrics['MRR@k']:.3f} | nDCG@{K}: {metrics['nDCG@k']:.3f} | "
                      f"Faith: {metrics['Faithfulness']:.3f} | Attr: {metrics['Attribution']:.3f}")
                print(f"  Time: {metrics['Time']:.2f}s\n")
    
    # Summary
    print(f"{'='*80}")
    print("SUMMARY (Averaged Across All Queries)")
    print(f"{'='*80}\n")
    
    for mode_name, results in all_results.items():
        if not results:
            continue
        
        avg_metrics = {
            metric: sum(r[metric] for r in results) / len(results)
            for metric in ["EM", "F1", "Precision@k", "Recall@k", "MRR@k", 
                          "nDCG@k", "Faithfulness", "Attribution", "Time"]
        }
        
        print(f"[{mode_name}]")
        print(f"  EM {avg_metrics['EM']:.3f} | F1 {avg_metrics['F1']:.3f} | "
              f"P@{K} {avg_metrics['Precision@k']:.3f} | R@{K} {avg_metrics['Recall@k']:.3f}")
        print(f"  MRR@{K} {avg_metrics['MRR@k']:.3f} | nDCG@{K} {avg_metrics['nDCG@k']:.3f} | "
              f"Faith {avg_metrics['Faithfulness']:.3f} | Attr {avg_metrics['Attribution']:.3f}")
        print(f"  Avg Time: {avg_metrics['Time']:.2f}s\n")
    
    # Save detailed results
    output_file = "evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(dict(all_results), f, indent=2)
    
    print(f"Detailed results saved to: {output_file}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
