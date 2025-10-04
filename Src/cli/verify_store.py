# Src/cli/verify_store.py
# Sanity checks for your Chroma store (count + peek), with optional index comparison.

import argparse
from pathlib import Path
import pandas as pd

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

def main():
    # ---------- CLI ----------
    ap = argparse.ArgumentParser(description="Verify Chroma store contents")
    ap.add_argument("--persist", default="./chroma_db", help="Chroma persist directory")
    ap.add_argument("--collection", default="laptop-specs", help="Chroma collection name")
    ap.add_argument("--limit", type=int, default=5, help="How many items to peek")
    ap.add_argument("--index_csv", default="data_index/docs_index.csv", help="Optional CSV to compare counts")
    args = ap.parse_args()

    persist_dir = Path(args.persist)
    index_csv   = Path(args.index_csv)

    # ---------- Init ----------
    emb = OllamaEmbeddings(model="mxbai-embed-large")
    db = Chroma(
        collection_name=args.collection,
        embedding_function=emb,
        persist_directory=str(persist_dir),
    )

    print(f"Chroma dir: {persist_dir.resolve()}")
    print(f"Collection: {args.collection}")

    # ---------- Count ----------
    count = None
    try:
        # Preferred when available
        count = db._collection.count()  # type: ignore[attr-defined]
    except Exception:
        # Fallback
        try:
            count = len(db.get()["ids"])
        except Exception as e:
            print("Could not count docs via fallback:", e)

    print(f"Docs in store: {count if count is not None else 'unknown'}")

    # ---------- Compare with docs_index.csv (optional) ----------
    if index_csv.exists():
        try:
            df = pd.read_csv(index_csv)
            print(f"docs_index.csv rows: {len(df)}  ({index_csv})")
        except Exception as e:
            print("Could not read docs_index.csv:", e)
    else:
        print(f"(info) No index CSV at {index_csv}; skipping comparison")

    # ---------- Peek a few ----------
    try:
        peek = db.get(limit=args.limit)
        ids   = peek.get("ids", []) or []
        metas = peek.get("metadatas", []) or []
        docs  = peek.get("documents", []) or []

        print(f"\nSample (limit={args.limit}):")
        for i, _id in enumerate(ids):
            md   = metas[i] if i < len(metas) else {}
            text = (docs[i] if i < len(docs) else "")[:160].replace("\n", " ")
            row  = md.get("row")
            name = md.get("name")
            print(f"- id={_id}  row={row}  name={name}")
            print(f"  text: {text}")
    except Exception as e:
        print("Could not peek into store:", e)

    # ---------- Tips ----------
    print("\nTips:")
    print("- If Docs in store < docs_index.csv rows: rebuild the DB: delete the persist dir and re-run your indexer.")
    print("- If names/rows look wrong, check how you set metadata when adding documents in your indexing script.")

if __name__ == "__main__":
    main()
