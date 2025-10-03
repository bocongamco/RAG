#!/usr/bin/env python3
"""
Complete End-to-End RAG Evaluation
Includes: Retrieval quality, Abstention rate, Attribution accuracy, and Answer quality
"""

import time
import sys
import os
import re
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vector import dense_search, bm25_retriever, hybrid_search
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


# Initialize LLM
llm = OllamaLLM(model="llama3.2")

# LLM Prompt Template
ANSWER_PROMPT = ChatPromptTemplate.from_template("""
You are a laptop shopping assistant.
Use ONLY the given CONTEXT. Do not invent facts.

Return the answer in EXACTLY this format:
Answer: <two short sentences with the exact MODEL and PRICE from context>
Citations: list ALL rows that appear in CONTEXT (comma separated, e.g., row=13, row=46)

If the answer is not in the context, reply exactly:
Answer: I don't know
Citations: (none)

CONTEXT:
{ctx}

QUESTION: {q}
""")


def calculate_relevance(docs, query, category):
    """Relevance scoring for retrieval quality"""
    query_lower = query.lower()
    query_terms = [term.lower() for term in query.split() if len(term) > 2]
    
    total_relevance = 0
    
    for doc in docs:
        content_lower = doc.page_content.lower()
        doc_relevance = 0
        
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
        
        for term in query_terms:
            if term in content_lower:
                doc_relevance += 0.5
                
        total_relevance += min(doc_relevance, 3)
    
    return int(total_relevance)


def verify_attribution_accuracy(response_text, retrieved_docs):
    """
    Verify if citations in LLM response match retrieved documents
    Returns: attribution accuracy score (0.0 to 1.0)
    """
    # Extract row citations from response
    row_citations = re.findall(r'row[=\s]*(\d+)', response_text.lower())
    
    if not row_citations:
        # No citations provided
        if "don't know" in response_text.lower() or "(none)" in response_text.lower():
            return 1.0  # Correctly abstained with no citations
        else:
            return 0.0  # Should have cited but didn't
    
    # Get rows from retrieved docs
    retrieved_rows = [str(d.metadata.get('row', '')) for d in retrieved_docs]
    
    # Check how many citations are correct
    correct_citations = sum(1 for cited_row in row_citations if cited_row in retrieved_rows)
    
    return correct_citations / len(row_citations) if row_citations else 0.0


def is_abstention(response_text):
    """Check if LLM appropriately abstained from answering"""
    abstention_indicators = [
        "i don't know",
        "don't know",
        "cannot answer",
        "insufficient information",
        "not enough context",
        "citations: (none)"
    ]
    response_lower = response_text.lower()
    return any(indicator in response_lower for indicator in abstention_indicators)


def generate_llm_answer(query, retrieved_docs):
    """Generate answer using LLM with retrieved context"""
    if not retrieved_docs:
        return "Answer: I don't know\nCitations: (none)"
    
    # Format context
    context_lines = []
    for doc in retrieved_docs:
        row = doc.metadata.get("row", "?")
        content = doc.page_content[:300]
        context_lines.append(f"[row={row}] {content}")
    context = "\n\n".join(context_lines)
    
    # Generate answer
    chain = ANSWER_PROMPT | llm
    try:
        response = chain.invoke({"ctx": context, "q": query})
        return response
    except Exception as e:
        return f"Answer: I don't know (LLM error: {str(e)})\nCitations: (none)"


def complete_rag_evaluation():
    """
    Complete end-to-end RAG evaluation including:
    1. Retrieval quality (relevance)
    2. Abstention rate (Walert-style)
    3. Attribution accuracy
    4. Answer quality
    """
    
    print("COMPLETE END-TO-END RAG EVALUATION")
    print("=" * 60)
    print("Evaluating: Retrieval + Generation + Attribution + Abstention")
    
    # Test cases with expected behavior
    test_cases = [
        ("Exact Product", "Acer Extensa 15 price", "should_answer", "Should find Acer and cite it"),
        ("Brand Query", "HP laptop specifications", "should_answer", "Should find HP laptops"),
        ("Semantic Query", "best laptop for students", "should_answer", "Should recommend suitable laptop"),
        ("Conceptual Query", "affordable gaming laptop", "should_answer", "Should find gaming laptops"),
        ("Keyword Heavy", "Intel Core i3 8GB RAM", "should_answer", "Should match technical specs"),
        ("Abstract Query", "professional work laptop", "should_answer", "Should understand context"),
        ("Out-of-scope", "smartphone prices", "should_abstain", "Should refuse - wrong domain"),
        ("Impossible Query", "laptop with 1TB RAM and $10 price", "should_abstain", "Should refuse - unrealistic"),
    ]
    
    # Metrics tracking
    retrieval_results = []
    llm_results = []
    attribution_scores = []
    abstention_results = {
        'should_answer': {'answered': 0, 'abstained': 0},
        'should_abstain': {'answered': 0, 'abstained': 0}
    }
    
    print(f"\n1. TESTING {len(test_cases)} QUERIES")
    print("-" * 60)
    
    for category, query, expected_behavior, description in test_cases:
        print(f"\n{category}: '{query}'")
        print(f"Expected: {description}")
        print("-" * 40)
        
        # Retrieve documents (using hybrid as default)
        start_time = time.time()
        docs = hybrid_search(query, k=3, alpha=0.6) or []
        retrieval_time = time.time() - start_time
        
        # Calculate retrieval relevance
        relevance = calculate_relevance(docs, query, category)
        retrieval_results.append({
            'query': query,
            'category': category,
            'relevance': relevance,
            'time': retrieval_time,
            'docs_count': len(docs)
        })
        
        print(f"Retrieval: {len(docs)} docs | {retrieval_time:.3f}s | relevance: {relevance}/9")
        
        # Generate LLM answer
        start_time = time.time()
        llm_response = generate_llm_answer(query, docs)
        generation_time = time.time() - start_time
        
        # Check abstention
        abstained = is_abstention(llm_response)
        if expected_behavior == "should_answer":
            if abstained:
                abstention_results['should_answer']['abstained'] += 1
                abstention_status = "WRONG - abstained when should answer"
            else:
                abstention_results['should_answer']['answered'] += 1
                abstention_status = "CORRECT - answered"
        else:  # should_abstain
            if abstained:
                abstention_results['should_abstain']['abstained'] += 1
                abstention_status = "CORRECT - abstained"
            else:
                abstention_results['should_abstain']['answered'] += 1
                abstention_status = "WRONG - answered when should abstain"
        
        # Check attribution accuracy
        attribution_accuracy = verify_attribution_accuracy(llm_response, docs)
        attribution_scores.append(attribution_accuracy)
        
        # Store results
        llm_results.append({
            'query': query,
            'category': category,
            'response': llm_response,
            'generation_time': generation_time,
            'abstained': abstained,
            'expected_behavior': expected_behavior,
            'correct_behavior': abstention_status.startswith('CORRECT'),
            'attribution_accuracy': attribution_accuracy
        })
        
        print(f"Generation: {generation_time:.3f}s")
        print(f"Abstention: {abstention_status}")
        print(f"Attribution: {attribution_accuracy:.2%} accurate")
        print(f"Response: {llm_response[:150]}...")
    
    # Calculate overall metrics
    print(f"\n2. OVERALL METRICS (Walert-Style)")
    print("=" * 60)
    
    # Abstention metrics
    should_answer_total = abstention_results['should_answer']['answered'] + abstention_results['should_answer']['abstained']
    should_abstain_total = abstention_results['should_abstain']['answered'] + abstention_results['should_abstain']['abstained']
    
    answered_rate = (abstention_results['should_answer']['answered'] / should_answer_total * 100) if should_answer_total > 0 else 0
    correct_abstention_rate = (abstention_results['should_abstain']['abstained'] / should_abstain_total * 100) if should_abstain_total > 0 else 0
    
    print(f"\nABSTENTION ANALYSIS:")
    print(f"  Answered Rate (should answer): {answered_rate:.1f}% ({abstention_results['should_answer']['answered']}/{should_answer_total})")
    print(f"  Correct Abstention Rate: {correct_abstention_rate:.1f}% ({abstention_results['should_abstain']['abstained']}/{should_abstain_total})")
    print(f"  Inappropriate Answers: {abstention_results['should_abstain']['answered']} (answered out-of-scope)")
    print(f"  Inappropriate Abstentions: {abstention_results['should_answer']['abstained']} (refused valid queries)")
    
    # Attribution accuracy
    avg_attribution = sum(attribution_scores) / len(attribution_scores) if attribution_scores else 0
    perfect_attributions = sum(1 for score in attribution_scores if score == 1.0)
    
    print(f"\nATTRIBUTION ACCURACY:")
    print(f"  Average Attribution Accuracy: {avg_attribution:.1%}")
    print(f"  Perfect Attributions: {perfect_attributions}/{len(attribution_scores)} ({perfect_attributions/len(attribution_scores)*100:.1f}%)")
    
    # Retrieval quality
    avg_relevance = sum(r['relevance'] for r in retrieval_results) / len(retrieval_results)
    avg_retrieval_time = sum(r['time'] for r in retrieval_results) / len(retrieval_results)
    
    print(f"\nRETRIEVAL QUALITY:")
    print(f"  Average Relevance: {avg_relevance:.1f}/9 ({avg_relevance/9*100:.1f}%)")
    print(f"  Average Retrieval Time: {avg_retrieval_time:.3f}s")
    
    # Generation performance
    avg_generation_time = sum(r['generation_time'] for r in llm_results) / len(llm_results)
    
    print(f"\nGENERATION PERFORMANCE:")
    print(f"  Average Generation Time: {avg_generation_time:.3f}s")
    print(f"  Total Pipeline Time: {avg_retrieval_time + avg_generation_time:.3f}s")
    
    # Business value summary
    print(f"\n3. BUSINESS VALUE ASSESSMENT")
    print("=" * 60)
    
    overall_correctness = sum(1 for r in llm_results if r['correct_behavior']) / len(llm_results) * 100
    
    print(f"\nSYSTEM RELIABILITY:")
    print(f"  Overall Correct Behavior: {overall_correctness:.1f}%")
    print(f"  Attribution Trustworthiness: {avg_attribution:.1%}")
    print(f"  Answer Rate: {answered_rate:.1f}%")
    print(f"  Appropriate Refusal Rate: {correct_abstention_rate:.1f}%")
    
    print(f"\nKEY FINDINGS:")
    if answered_rate >= 80:
        print(f"  ✓ System answers most valid queries ({answered_rate:.0f}%)")
    else:
        print(f"  ⚠ System abstains too often ({100-answered_rate:.0f}% abstention rate)")
    
    if correct_abstention_rate >= 80:
        print(f"  ✓ System correctly refuses out-of-scope queries ({correct_abstention_rate:.0f}%)")
    else:
        print(f"  ⚠ System answers out-of-scope queries ({100-correct_abstention_rate:.0f}% error rate)")
    
    if avg_attribution >= 0.8:
        print(f"  ✓ Citations are highly accurate ({avg_attribution:.0%})")
    else:
        print(f"  ⚠ Citation accuracy needs improvement ({avg_attribution:.0%})")
    
    print(f"\nCOMPARISON TO WALERT:")
    print(f"  Walert target: >85% answered rate on valid queries")
    print(f"  Our system: {answered_rate:.1f}% answered rate")
    print(f"  Status: {'MEETS' if answered_rate >= 85 else 'BELOW'} target")
    
    # Detailed results table
    print(f"\n4. DETAILED RESULTS BY QUERY")
    print("=" * 60)
    for result in llm_results:
        status = "✓" if result['correct_behavior'] else "✗"
        print(f"\n{status} {result['category']}: '{result['query']}'")
        print(f"   Abstained: {result['abstained']} (expected: {result['expected_behavior']})")
        print(f"   Attribution: {result['attribution_accuracy']:.0%}")
        print(f"   Time: {result['generation_time']:.3f}s")
    
    return {
        'retrieval_results': retrieval_results,
        'llm_results': llm_results,
        'metrics': {
            'answered_rate': answered_rate,
            'correct_abstention_rate': correct_abstention_rate,
            'attribution_accuracy': avg_attribution,
            'avg_relevance': avg_relevance,
            'overall_correctness': overall_correctness
        }
    }


if __name__ == "__main__":
    results = complete_rag_evaluation()