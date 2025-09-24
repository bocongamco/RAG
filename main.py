# main.py
# Command-line RAG assistant for the Laptop dataset.
# Modes: dense (embeddings), bm25 (lexical), hybrid (fusion).
# Can run interactively or answer a single query via --query.

import argparse
from typing import List
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import dense_search, bm25_retriever, hybrid_search

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

# Note: we still nudge the LLM to cite all docs, but we also print the full k-list ourselves
_PROMPT = ChatPromptTemplate.from_template("""
You are a laptop shopping assistant.
Use ONLY the given CONTEXT. Do not invent facts.

Return the answer in EXACTLY this format:
Answer: <two short sentences with the exact MODEL and PRICE from context>
Citations: list ALL rows and names that appear in CONTEXT (comma separated)

If the answer is not in the context, reply exactly:
Answer: I don't know
Citations: (none)

CONTEXT:
{ctx}

QUESTION: {q}
""")
_chain = _PROMPT | _llm

# -------------------
# Helpers
# -------------------
def _fmt_ctx(docs: List) -> str:
    """Format k docs into one context string."""
    lines = []
    for d in docs:
        row = d.metadata.get("row")
        name = d.metadata.get("name")
        snippet = (d.page_content or "")[:350].replace("\n", " ")
        lines.append(f"[row={row} | name={name}] {snippet}")
    return "\n".join(lines)

    
def _citations_line(docs: List) -> str:
    """print all retrieved docs, one per line."""
    lines = [f"Citations (k={len(docs)}):"]
    for d in docs:
        row = d.metadata.get("row", "")
        name = d.metadata.get("name", "")
        lines.append(f"  row={row}, name=\"{name}\"")
    return "\n".join(lines)

def _retrieve(q: str) -> List:
    """Get docs according to mode + k."""
    if mode == "dense":
        return dense_search(q, k=top_k) or []
    elif mode == "bm25":
        return bm25_retriever.search(q, k=top_k) or []
    else:
        return hybrid_search(q, k=top_k, alpha=alpha) or []

def _answer(q: str) -> None:
    docs = _retrieve(q)
    if not docs:
        print("Answer: I don't know")
        print("Citations: (none)")
        return

    ctx = _fmt_ctx(docs)
    try:
        out = _chain.invoke({"ctx": ctx, "q": q})
        print(str(out))
    except Exception as e:
        print("Answer: retrieval succeeded but the LLM is unavailable")
        print("Citations: (none)")
        print(f"[debug] LLM error: {e}")
    # always show the full retrieved set (exactly k or fewer if corpus is small)
    print(_citations_line(docs))

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
