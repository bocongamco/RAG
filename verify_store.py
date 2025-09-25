# verify_store.py
<<<<<<< HEAD
# Sanity checks for your Chroma store (count + peek), with optional index comparison.

import argparse
from pathlib import Path
import pandas as pd

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# ---------- CLI ----------
ap = argparse.ArgumentParser(description="Verify Chroma store contents")
ap.add_argument("--persist", default="./chrome_langchain_db", help="Chroma persist directory")
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
    # Preferred when available; still private but common in the wild.
    count = db._collection.count()  # type: ignore[attr-defined]
except Exception:
    # Fallback: may be heavy on large stores, but fine for small projects.
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
print("- If Docs in store < docs_index.csv rows: rebuild the DB: delete chrome_langchain_db/ and run `python vector.py`.")
print("- If names/rows look wrong in the sample, check how you set metadata when adding documents in vector.py.")
=======
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

emb = OllamaEmbeddings(model="mxbai-embed-large")
db = Chroma(
    collection_name="laptop-specs",
    embedding_function=emb,
    persist_directory="./chrome_langchain_db",
)

# Count docs
try:
    n = db._collection.count()  # chromadb Collection
except Exception:
    n = len(db.get()["ids"])
print("Docs in store:", n)

# Peek a few items
peek = db.get(limit=3)
print("Sample ids:", peek.get("ids", [])[:3])
>>>>>>> origin/main
