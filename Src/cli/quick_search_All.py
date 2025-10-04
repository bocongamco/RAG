# quick_search.py
# One-file sanity check for dense / bm25 / hybrid retrieval.

import os, sys
from pathlib import Path
import argparse
import pandas as pd
from langchain_core.documents import Document

# make sure we can import vector.py from the project root
sys.path.append(os.path.abspath("."))

# these names match the student-style vector.py we wrote earlier
# - vector_db: the Chroma store
# - bm25_retriever: lexical retriever
# - hybrid_search: fusion of dense + bm25
#from Src.search.vector import vector_db, bm25_retriever, hybrid_search
#from Src.search.vector import vector_db, dense_search, hybrid_search, knowledge_search, bm25_get
from Src.search.vector import init_stores, dense_search as vector_dense_search, hybrid_search, bm25_get

INDEX_CSV = Path("data_index") / "docs_index.csv"

def load_row_meta():
    """Load price/rating/name by row for nicer printouts (optional)."""
    if not INDEX_CSV.exists():
        return {}
    df = pd.read_csv(INDEX_CSV)
    meta = {}
    for _, r in df.iterrows():
        try:
            row_i = int(r["row"])
        except Exception:
            continue
        meta[row_i] = {
            "name": str(r.get("name", "")),
            "price": r.get("price", None),
            "rating": r.get("rating", None),
        }
    return meta

def print_hits(label, docs, row_meta, max_chars=240):
    print(f"\n[{label}] top-{len(docs)}")
    for i, d in enumerate(docs, 1):
        row = d.metadata.get("row")
        name = d.metadata.get("name")
        try:
            row_i = int(row) if row is not None else None
        except Exception:
            row_i = None
        m = row_meta.get(row_i, {}) if row_i is not None else {}
        price = m.get("price")
        rating = m.get("rating")
        print(f"{i}. row={row}  name={name}")
        if price is not None or rating is not None:
            print(f"   price={price}  rating={rating}")
        print("   ", (d.page_content or "")[:max_chars].replace("\n", " "))
    print("-" * 60)

#def dense_search(query: str, k: int):
#    """Dense search via Chroma (Ollama embeddings). Honors k per call."""
#    return vector_db.similarity_search(query, k=k)

def main():
    init_stores() 
    ap = argparse.ArgumentParser(description="Quick retrieval test (dense/bm25/hybrid)")
    ap.add_argument("query", help="e.g. 'price of Acer Aspire 5'")
    ap.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="dense")
    ap.add_argument("--k", type=int, default=3, help="top-k documents (1..20)")
    ap.add_argument("--alpha", type=float, default=0.6, help="hybrid: weight for dense (0..1)")
    args = ap.parse_args()

    q = args.query.strip()
    k = max(1, min(20, args.k))
    alpha = max(0.0, min(1.0, float(args.alpha)))

    row_meta = load_row_meta()

    print(f"Query: {q}")
    print(f"Mode:  {args.mode}  |  k={k}" + (f"  |  alpha={alpha}" if args.mode == "hybrid" else ""))

    try:
        if args.mode == "dense":
            #docs = dense_search(q, k=k) or []
            docs = vector_dense_search(q, k=k) or []
            print_hits("dense", docs, row_meta)
        elif args.mode == "bm25":
            docs = bm25_get(q, k=k) or []
            print_hits("bm25", docs, row_meta)
        else:
            docs = hybrid_search(q, k=k, alpha=alpha) or []
            print_hits("hybrid", docs, row_meta)
    except Exception as e:
        print("ERROR:", e)
        print("\nHints:")
        print("- Is Ollama running and 'mxbai-embed-large' pulled? (needed for dense/hybrid)")
        print("- Does chrome_langchain_db/chroma.sqlite3 exist? If not, run: python vector.py")

if __name__ == "__main__":
    main()
