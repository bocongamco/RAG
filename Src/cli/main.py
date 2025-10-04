# main.py
# Command-line RAG assistant for the Laptop dataset.
# Modes: dense (embeddings), bm25 (lexical), hybrid (fusion).
# Can run interactively or answer a single query via --query.

import argparse
from typing import List
from pathlib import Path
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
#from Src.search.vector import dense_search, bm25_retriever, hybrid_search, knowledge_search
from Src.search.vector import dense_search, hybrid_search, knowledge_search, bm25_get
# -------------------
# CLI args
# -------------------
parser = argparse.ArgumentParser(description="Laptop RAG: CLI assistant")
parser.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="hybrid",
                    help="Retrieval mode")
parser.add_argument("--k", type=int, default=3, help="Top-k documents to retrieve")
parser.add_argument("--alpha", type=float, default=0.6,
                    help="Hybrid: weight for dense (0..1)")
parser.add_argument("--query", type=str, default=None,
                    help="If omitted, runs in interactive loop")
args = parser.parse_args()

mode   = args.mode.lower()
top_k  = max(1, min(20, args.k))
alpha  = max(0.0, min(1.0, float(args.alpha)))

# -------------------
# LLM + Prompt
# -------------------
_llm = OllamaLLM(model="llama3.2")

_PROMPT = ChatPromptTemplate.from_template("""
You are a laptop shopping assistant.

Below you will see products from our catalog and technical knowledge articles.

USER QUERY: {q}

YOUR TASK:
1. Find products that match the user's requirements
2. Only recommend products if they match ALL requirements mentioned
3. Use technical knowledge to explain why your recommendation is good

PRODUCTS IN OUR CATALOG:
{ctx}

INSTRUCTIONS:
- Check each product's FULL_SPECS carefully
- A product must match every requirement the user mentioned
- If no exact match exists, say so and explain why
- Cite the ROW number and product NAME

Answer now:
""")

_chain = _PROMPT | _llm

# -------------------
# Helpers
# -------------------
def _detect_query_intent(q: str) -> str:
    """Determine query type - defaults to hybrid for safety."""
    q_lower = q.lower()
    
    # ANY mention of brands/laptops/products = must retrieve products
    brand_or_product = ["acer", "hp", "dell", "lenovo", "asus", "msi", "apple",
                       "samsung", "laptop", "which", "best", "recommend", 
                       "find", "under", "price", "cost"]
    
    has_product_intent = any(term in q_lower for term in brand_or_product)
    
    # Pure knowledge only if clearly asking for explanation with NO product context
    pure_knowledge_starts = ["why is", "what makes", "explain why", 
                            "how does", "what is the difference"]
    is_pure_knowledge = any(q_lower.startswith(kw) for kw in pure_knowledge_starts)
    
    # If it's pure explanation about concepts (not products), use knowledge only
    if is_pure_knowledge and not has_product_intent:
        return "knowledge"
    
    # If asking about price/cost, definitely need products
    if "price" in q_lower or "cost" in q_lower or "under" in q_lower:
        return "product"
    
    # Default to hybrid (safer - retrieves both sources)
    return "hybrid"

def _retrieve_multi_source(q: str) -> tuple:
    """Retrieve from both product catalog and knowledge base."""
    intent = _detect_query_intent(q)
    
    print(f"[Query intent: {intent}]")
    
    # Adjust retrieval based on intent
    if intent == "knowledge":
        product_k = 0  # Don't retrieve products for pure explanation queries
        knowledge_k = 5  # Get more knowledge chunks
    elif intent == "product":
        product_k = top_k
        knowledge_k = 0  # Don't need knowledge for pure product queries
    else:  # hybrid
        product_k = top_k
        knowledge_k = 3
    
    # Get product docs
    if product_k > 0:
        if mode == "dense":
            product_docs = dense_search(q, k=product_k) or []
        elif mode == "bm25":
            product_docs = bm25_get(q, k=product_k) or []
        else:
            product_docs = hybrid_search(q, k=product_k, alpha=alpha) or []
    else:
        product_docs = []
    
    # Get knowledge docs
    if knowledge_k > 0:
        knowledge_docs = knowledge_search(q, k=knowledge_k) or []
    else:
        knowledge_docs = []
    
    return product_docs, knowledge_docs

def _fmt_combined_ctx(product_docs, knowledge_docs) -> str:
    """Format context with full product specifications visible."""
    sections = []
    
    if product_docs:
        sections.append("=== PRODUCT CATALOG ===")
        sections.append("CRITICAL: Each product below is COMPLETELY SEPARATE. DO NOT MIX THEIR SPECS.")
        sections.append("")
        
        for i, d in enumerate(product_docs, 1):
            row = d.metadata.get("row", "")
            name = d.metadata.get("name", "")
            # Show FULL content instead of truncated snippet
            content = d.page_content or ""
            
            sections.append(f"--- PRODUCT #{i} ---")
            sections.append(f"ROW: {row}")
            sections.append(f"NAME: {name}")
            sections.append(f"FULL_SPECS: {content}")  # Full specs, not truncated
            sections.append("")
    
    if knowledge_docs:
        sections.append("=== TECHNICAL KNOWLEDGE (NOT OUR INVENTORY) ===")
        sections.append("Products mentioned below are examples only - NOT in our catalog.")
        sections.append("")
        for d in knowledge_docs:
            source = Path(d.metadata.get("source", "knowledge")).name
            snippet = (d.page_content or "")[:500].replace("\n", " ")
            sections.append(f"[{source}] {snippet}")
            sections.append("")
    
    return "\n".join(sections)

def _citations_line(docs: List) -> str:
    """Print all retrieved docs."""
    if not docs:
        return ""
    lines = [f"\nProduct Citations (k={len(docs)}):"]
    for d in docs:
        row = d.metadata.get("row", "")
        name = d.metadata.get("name", "")
        lines.append(f"  row={row}, name=\"{name}\"")
    return "\n".join(lines)

def _validate_response(response: str, product_docs: List) -> tuple[str, bool]:
    """
    Validate response for hallucinated specs.
    Returns (response, is_valid)
    """
    if not product_docs:
        return response, True
    
    response_lower = response.lower()
    warnings = []
    
    # Check for GPU hallucination
    if any(gpu in response_lower for gpu in ["gtx", "rtx", "dedicated gpu", "nvidia geforce"]):
        has_dedicated = any(
            "GTX" in d.page_content or "RTX" in d.page_content or "GeForce" in d.page_content
            for d in product_docs
        )
        if not has_dedicated:
            warnings.append("⚠️ Response mentions dedicated GPU but retrieved products only have integrated graphics")
    
    # Check for high RAM claims
    if "16gb" in response_lower or "32gb" in response_lower:
        # Extract actual RAM from products
        product_specs = " ".join(d.page_content for d in product_docs).lower()
        if "4gb" in product_specs and "16gb" not in product_specs:
            warnings.append("⚠️ Response mentions 16GB RAM but product actually has 4GB RAM")
    
    if warnings:
        warning_text = "\n".join(warnings)
        return f"{warning_text}\n\n{response}", False
    
    return response, True

def _answer(q: str) -> None:
    """Answer using multi-source retrieval with hallucination detection."""
    product_docs, knowledge_docs = _retrieve_multi_source(q)
    
    if not product_docs and not knowledge_docs:
        print("Answer: I don't know")
        print("Citations: (none)")
        return
    
    ctx = _fmt_combined_ctx(product_docs, knowledge_docs)
    
    try:
        out = _chain.invoke({"ctx": ctx, "q": q})
        
        # Validate response
        validated_out, is_valid = _validate_response(str(out), product_docs)
        
        print(validated_out)
        
        if not is_valid:
            print("\n[DEBUG: Hallucination detected - check product specs carefully]")
            
    except Exception as e:
        print("Answer: Retrieval succeeded but LLM unavailable")
        print(f"[debug] LLM error: {e}")
    
    # Show citations
    if product_docs:
        print(_citations_line(product_docs))
    if knowledge_docs:
        sources = set(Path(d.metadata.get("source", "knowledge")).name for d in knowledge_docs)
        print(f"\nKnowledge sources: {', '.join(sources)}")

# -------------------
# Entry
# -------------------
if __name__ == "__main__":
    if args.query:
        _answer(args.query.strip())
    else:
        print(f"Retrieval mode = {mode.upper()} | k={top_k}" + (f" | alpha={alpha}" if mode=='hybrid' else ""))
        while True:
            q = input("\nAsk your question (q to quit): ").strip()
            if q.lower() == "q":
                break
            if q:
                _answer(q)
