#!/usr/bin/env python3
"""
Fixed Multi-Mode RAG Evaluation: Dense vs BM25 vs Hybrid
Corrected relevance scoring and evaluation logic
"""

import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vector import dense_search, bm25_retriever, hybrid_search


def calculate_relevance(docs, query, category):
    """
    Improved relevance scoring that considers all query terms and context
    """
    query_lower = query.lower()
    query_terms = [term.lower() for term in query.split() if len(term) > 2]
    
    total_relevance = 0
    
    for doc in docs:
        content_lower = doc.page_content.lower()
        doc_relevance = 0
        
        # Category-specific relevance checks
        if category == "Exact Product":
            if "acer" in query_lower and "acer" in content_lower:
                doc_relevance += 2
            if "extensa" in query_lower and "extensa" in content_lower:
                doc_relevance += 2
            if any(term in content_lower for term in ["price", "$", "₹", "cost"]):
                doc_relevance += 1
                
        elif category == "Brand Query":
            brand_terms = ["hp", "dell", "acer", "lenovo", "asus"]
            for brand in brand_terms:
                if brand in query_lower and brand in content_lower:
                    doc_relevance += 3
            if "laptop" in content_lower:
                doc_relevance += 1
                
        elif category in ["Semantic Query", "Conceptual Query", "Abstract Query"]:
            if "student" in query_lower and any(term in content_lower for term in ["student", "study", "education", "portable", "lightweight"]):
                doc_relevance += 2
            if "gaming" in query_lower and "gaming" in content_lower:
                doc_relevance += 2
            if "professional" in query_lower and any(term in content_lower for term in ["professional", "business", "office", "work"]):
                doc_relevance += 2
            if "affordable" in query_lower and any(term in content_lower for term in ["budget", "affordable", "cheap", "value"]):
                doc_relevance += 2
            if "laptop" in content_lower:
                doc_relevance += 1
                
        elif category == "Keyword Heavy":
            tech_terms = ["intel", "core", "i3", "i5", "i7", "8gb", "16gb", "ssd", "hdd", "ram", "amd", "ryzen"]
            for term in tech_terms:
                if term in query_lower and term in content_lower:
                    doc_relevance += 1
            if "laptop" in content_lower:
                doc_relevance += 1
                
        elif category == "Out-of-scope":
            if "laptop" in content_lower:
                doc_relevance = 0
            else:
                doc_relevance = 3
        
        # General term matching as fallback
        for term in query_terms:
            if term in content_lower:
                doc_relevance += 0.5
                
        total_relevance += min(doc_relevance, 3)
    
    return int(total_relevance)

def multi_mode_evaluation():
    print("FIXED MULTI-MODE RAG EVALUATION")
    print("=" * 50)
    print("Testing Dense vs BM25 vs Hybrid retrieval modes")
    
    # Define retrieval modes
    modes = {
        "Dense": lambda q, k: dense_search(q, k=k),
        "BM25": lambda q, k: bm25_retriever.search(q, k=k), 
        "Hybrid": lambda q, k: hybrid_search(q, k=k, alpha=0.6)
    }
    
    # Test queries designed to show mode differences (using actual models in your dataset)
    test_cases = [
        ("Exact Product", "Acer Extensa 15 price", "BM25 should excel"),
        ("Brand Query", "HP laptop specifications", "BM25 should excel"), 
        ("Semantic Query", "best laptop for students", "Dense should excel"),
        ("Conceptual Query", "affordable gaming laptop", "Dense/Hybrid should excel"),
        ("Keyword Heavy", "Intel Core i3 8GB RAM", "BM25/Hybrid should excel"),
        ("Abstract Query", "professional work laptop", "Dense should excel"),
        ("Out-of-scope", "smartphone prices", "All should handle poorly")
    ]
    
    print(f"\n1. MODE COMPARISON BY QUERY TYPE")
    print("-" * 50)
    
    all_results = {}
    
    for category, query, expectation in test_cases:
        print(f"\nQuery: '{query}' ({category})")
        print(f"Expected: {expectation}")
        print("-" * 40)
        
        mode_results = {}
        
        for mode_name, search_func in modes.items():
            start_time = time.time()
            
            try:
                docs = search_func(query, 3) or []
                elapsed = time.time() - start_time
                
                # Use improved relevance calculation
                relevance_score = calculate_relevance(docs, query, category)
                max_score = 3 * 3  # 3 docs × max 3 points each
                
                mode_results[mode_name] = {
                    "docs_count": len(docs),
                    "time": elapsed,
                    "relevance_score": relevance_score,
                    "max_score": max_score
                }
                
                # Show sample results for debugging
                if mode_name == "BM25" and category == "Exact Product":
                    print(f"  {mode_name:6s}: {len(docs)} docs | {elapsed:.3f}s | relevance: {relevance_score}/{max_score}")
                    for i, doc in enumerate(docs):
                        content_preview = doc.page_content[:80].replace('\n', ' ')
                        acer_count = content_preview.lower().count('acer')
                        print(f"    Doc {i+1}: Acer mentions: {acer_count} | {content_preview}...")
                else:
                    print(f"  {mode_name:6s}: {len(docs)} docs | {elapsed:.3f}s | relevance: {relevance_score}/{max_score}")
                
            except Exception as e:
                mode_results[mode_name] = {
                    "docs_count": 0,
                    "time": 0,
                    "relevance_score": 0,
                    "max_score": 9,
                    "error": str(e)
                }
                print(f"  {mode_name:6s}: ERROR - {e}")
        
        all_results[category] = {
            "query": query,
            "expectation": expectation,
            "results": mode_results
        }
    
    # Parameter sensitivity analysis
    print(f"\n2. HYBRID MODE PARAMETER ANALYSIS")
    print("-" * 40)
    
    alpha_values = [0.0, 0.3, 0.5, 0.7, 1.0]
    test_query = "gaming laptop with good price"
    
    print(f"Testing query: '{test_query}'")
    print("α=0.0 (pure BM25) → α=1.0 (pure Dense)")
    
    for alpha in alpha_values:
        start_time = time.time()
        docs = hybrid_search(test_query, k=3, alpha=alpha) or []
        elapsed = time.time() - start_time
        
        gaming_mentions = sum(1 for doc in docs if "gaming" in doc.page_content.lower())
        price_mentions = sum(1 for doc in docs if any(term in doc.page_content.lower() 
                           for term in ["price", "$", "₹"]))
        
        # Show actual document differences
        first_doc_preview = docs[0].page_content[:50].replace('\n', ' ') if docs else "No docs"
        
        mode_desc = "Pure BM25" if alpha == 0.0 else "Pure Dense" if alpha == 1.0 else "Hybrid"
        print(f"  α={alpha:.1f} ({mode_desc:10s}): {len(docs)} docs | gaming:{gaming_mentions} price:{price_mentions} | {elapsed:.3f}s")
        print(f"       First doc: {first_doc_preview}...")
    
    # Performance comparison with proper timing
    print(f"\n3. PERFORMANCE BENCHMARKING")
    print("-" * 40)
    
    benchmark_queries = [
        "laptop", "gaming", "business", "HP", "Dell"
    ]
    
    mode_performance = {}
    
    for mode_name, search_func in modes.items():
        total_time = 0
        total_docs = 0
        
        for query in benchmark_queries:
            start_time = time.time()
            docs = search_func(query, 3) or []
            query_time = time.time() - start_time
            total_time += max(query_time, 0.0001)  # Prevent zero division
            total_docs += len(docs)
        
        avg_time = total_time / len(benchmark_queries)
        throughput = len(benchmark_queries) / total_time
        
        mode_performance[mode_name] = {
            "avg_time": avg_time,
            "throughput": throughput,
            "total_docs": total_docs
        }
        
        print(f"  {mode_name:6s}: {avg_time:.4f}s avg | {throughput:.1f} q/s | {total_docs} docs total")
    
    # Improved summary and insights
    print(f"\n4. EVALUATION SUMMARY")
    print("=" * 50)
    
    # Find best performing mode for each category
    category_winners = {}
    
    for category, data in all_results.items():
        if category == "Out-of-scope":
            # For out-of-scope, lower relevance is better
            best_mode = min(data["results"].items(), 
                          key=lambda x: x[1].get("relevance_score", 999))
        else:
            # For in-scope, higher relevance is better
            best_mode = max(data["results"].items(), 
                          key=lambda x: x[1].get("relevance_score", 0))
        
        category_winners[category] = best_mode[0]
        winner_score = best_mode[1].get("relevance_score", 0)
        winner_max = best_mode[1].get("max_score", 9)
        
        print(f"{category:15s}: {best_mode[0]} won ({winner_score}/{winner_max})")
    
    # Mode strengths analysis
    print(f"\nMODE STRENGTHS ANALYSIS:")
    
    dense_wins = sum(1 for winner in category_winners.values() if winner == "Dense")
    bm25_wins = sum(1 for winner in category_winners.values() if winner == "BM25") 
    hybrid_wins = sum(1 for winner in category_winners.values() if winner == "Hybrid")
    
    print(f"- Dense excels at: {dense_wins}/{len(category_winners)} categories")
    print(f"- BM25 excels at: {bm25_wins}/{len(category_winners)} categories")
    print(f"- Hybrid excels at: {hybrid_wins}/{len(category_winners)} categories")
    
    # Detailed performance analysis
    print(f"\nDETAILED PERFORMANCE BREAKDOWN:")
    for category, data in all_results.items():
        print(f"\n{category}: '{data['query']}'")
        for mode_name, stats in data["results"].items():
            score = stats.get("relevance_score", 0)
            max_score = stats.get("max_score", 9)
            time_ms = stats.get("time", 0) * 1000
            print(f"  {mode_name:6s}: {score:2d}/{max_score} relevance ({time_ms:5.1f}ms)")
    
    # Business recommendations based on actual results
    print(f"\nBUSINESS RECOMMENDATIONS:")
    
    if hybrid_wins >= max(dense_wins, bm25_wins):
        print("- Deploy HYBRID mode for production (best overall performance)")
        print("- Use α=0.6 for balanced semantic + keyword matching")
    elif bm25_wins > dense_wins:
        print("- Deploy BM25 mode for production (excels at exact product queries)")
        print("- Consider hybrid for broader query coverage")
    else:
        print("- Deploy DENSE mode for production (excels at semantic understanding)")
        print("- Consider BM25 for exact product lookups")
    
    fastest_mode = min(mode_performance.items(), key=lambda x: x[1]["avg_time"])
    print(f"- Fastest mode: {fastest_mode[0]} ({fastest_mode[1]['avg_time']:.4f}s avg)")
    
    # Quality insights
    if bm25_wins > 0:
        print("- BM25 successfully handles exact keyword matching")
    if dense_wins > 0:
        print("- Dense search excels at conceptual understanding")
    if hybrid_wins > 0:
        print("- Hybrid provides balanced coverage across query types")
    
    return all_results, mode_performance

if __name__ == "__main__":
    results, performance = multi_mode_evaluation()