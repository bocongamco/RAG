# Retrieval pipeline for the Laptop RAG project.
# Location: src/search/vector.py

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import os, json, hashlib
from datetime import datetime

import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

def _log(msg: str):
    print(msg)

# =========================
# Globals (lazy init)
# =========================
INIT_DONE = False
vector_db: Chroma | None = None
bm25_retriever = None
laptop_docs: List[Document] = []

# =========================
# Project paths (rooted)
# =========================
ROOT_DIR   = Path(__file__).resolve().parents[2]          # repo root
DATA_DIR   = ROOT_DIR / "data"
ART_DIR    = ROOT_DIR / "Collection"
EVAL_DIR   = ART_DIR / "init_eval"
EDA_DIR    = ART_DIR / "eda"
CHROMA_DIR = ROOT_DIR / "chroma_db"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

INDEX_DIR  = ROOT_DIR / "data_index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# CSV candidates (picked at runtime, not import time)
CSV_PATH_1 = DATA_DIR / "Training-Data" / "Amazon_Laptop_Specs_utf8.csv"
CSV_PATH_2 = DATA_DIR / "Amazon_Laptop_Specs_utf8.csv"

INDEX_CSV  = INDEX_DIR / "docs_index.csv"
DOCS_DIR   = INDEX_DIR / "docs"
DOCID_PATH = INDEX_DIR / "docid"
META_PATH  = INDEX_DIR / "meta.json"

# >>> NEW: where alpha is written by the tuner <<<
ALPHA_SUMMARY_PATH = ROOT_DIR / "Outputs" / "alpha_eval" / "summary.json"  # preferred
ALPHA_JSON_PATH    = EVAL_DIR / "alpha.json"                                # legacy fallback

# =========================
# Embeddings (Ollama)
# =========================
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# =========================
# Utilities
# =========================
def _safe_str(v) -> str:
    return "" if (isinstance(v, float) and pd.isna(v)) else str(v)

def chroma_missing(path: Path | str) -> bool:
    return not os.path.exists(os.path.join(str(path), "chroma.sqlite3"))

def get_csv_path() -> Optional[Path]:
    """Pick the best available CSV **at call time**."""
    if CSV_PATH_1.exists():
        return CSV_PATH_1
    if CSV_PATH_2.exists():
        return CSV_PATH_2
    return None

def _row_to_text(i: int, row: pd.Series) -> tuple[str, str, str, str]:
    s = _safe_str
    name, price, rating = s(row.get("Name")), s(row.get("Price")), s(row.get("Customer Rating"))
    ratings_n, dims = s(row.get("Number of Ratings")), s(row.get("Item Dimensions LxWxH"))
    content = (
        f"ROW={i} | MODEL={name} | PRICE={price} | RATING={rating} | RATINGS={ratings_n} | DIMS={dims}. "
        f"{s(row.get('Best Sellers Rank'))} {s(row.get('Net Quantity'))} {s(row.get('Generic Name'))}"
    )
    return content, name, price, rating

def _bucket_name(i: int, bucket_size: int = 100) -> str:
    return f"P{(i // bucket_size) + 1:02d}"

def _dump_markdown(index_rows: List[dict], bucket_size: int = 100) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for r in index_rows:
        bucket = _bucket_name(int(r["row"]), bucket_size)
        out_dir = DOCS_DIR / bucket
        out_dir.mkdir(parents=True, exist_ok=True)
        md = [
            f"# {r['name']}", "",
            f"doc_id: {r['doc_id']}",
            f"row: {r['row']}",
            f"price: {r['price']}",
            f"rating: {r['rating']}", "",
            "## Content",
            r["content"]
        ]
        (out_dir / f"{r['doc_id']}.md").write_text("\n".join(md), encoding="utf-8")

    with DOCID_PATH.open("w", encoding="utf-8") as f:
        for r in index_rows:
            name = '"' + str(r['name']).replace('"', '""') + '"'
            f.write(f"{int(r['row'])},{r['doc_id']},{name}\n")

    META_PATH.write_text(json.dumps({
        "collection": "laptop-specs",
        "count": len(index_rows),
        "created": datetime.utcnow().isoformat() + "Z",
        "schema": ["doc_id","content","row","name","price","rating"],
        "bucket_policy": {"type": "row_range", "bucket_size": 100, "format": "P%02d"}
    }, indent=2), encoding="utf-8")

# =========================
# Build docs index from CSV / load from index
# =========================
def build_from_csv(csv_path: Optional[Path] = None) -> tuple[List[Document], List[str]]:
    csv = csv_path or get_csv_path()
    if not csv or not csv.exists():
        raise FileNotFoundError(f"Missing CSV at {CSV_PATH_1} or {CSV_PATH_2}")

    df = pd.read_csv(csv)
    docs, ids, idx_rows = [], [], []
    for i, row in df.iterrows():
        content, name, price, rating = _row_to_text(i, row)
        doc_id = hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]
        docs.append(Document(page_content=content,
                             metadata={"row": str(i), "name": name, "doc_id": doc_id}))
        ids.append(doc_id)
        idx_rows.append({
            "doc_id": doc_id,
            "content": content,
            "row": int(i),
            "name": name,
            "price": price,
            "rating": rating
        })

    pd.DataFrame(idx_rows, columns=["doc_id","content","row","name","price","rating"]).to_csv(INDEX_CSV, index=False)
    _dump_markdown(idx_rows)
    _log(f"[vector] Built docs_index.csv with {len(idx_rows)} rows from {csv.name}")
    return docs, ids

def build_from_index() -> tuple[List[Document], List[str]]:
    if not INDEX_CSV.exists():
        raise FileNotFoundError(f"Missing {INDEX_CSV}")

    df = pd.read_csv(INDEX_CSV)
    required = {"doc_id","content","row","name","price","rating"}
    if not required.issubset(df.columns):
        missing = sorted(list(required - set(df.columns)))
        raise ValueError(f"docs_index.csv missing required columns: {missing}")

    docs = [
        Document(page_content=row["content"],
                 metadata={"row": str(int(row["row"])), "name": row["name"], "doc_id": row["doc_id"]})
        for _, row in df.iterrows()
    ]
    ids = df["doc_id"].tolist()
    _dump_markdown(df.to_dict(orient="records"))
    _log(f"[vector] Loaded {len(docs)} docs from docs_index.csv")
    return docs, ids

def rebuild_index_from_chroma(db: Chroma) -> tuple[List[Document], List[str]]:
    """
    Last-resort: rebuild docs_index from existing Chroma collection contents.
    """
    _log("[vector] Attempting to rebuild docs_index from Chroma collection...")
    data = db.get(include=["documents", "metadatas", "ids"])
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    ids = data.get("ids") or []

    if not documents:
        _log("[vector] Chroma returned no documents; cannot rebuild index.")
        return [], []

    idx_rows = []
    docs = []
    for i, (doc_text, meta, did) in enumerate(zip(documents, metadatas, ids)):
        meta = meta or {}
        name = meta.get("name") or ""
        row  = meta.get("row")
        try:
            row_int = int(row) if row is not None else i
        except Exception:
            row_int = i

        idx_rows.append({
            "doc_id": did,
            "content": doc_text,
            "row": row_int,
            "name": name,
            "price": "",
            "rating": ""
        })
        docs.append(Document(page_content=doc_text, metadata={"row": str(row_int), "name": name, "doc_id": did}))

    pd.DataFrame(idx_rows, columns=["doc_id","content","row","name","price","rating"]).to_csv(INDEX_CSV, index=False)
    _dump_markdown(idx_rows)
    _log(f"[vector] Rebuilt docs_index.csv from Chroma with {len(idx_rows)} rows")
    return docs, ids

def ensure_index_built(db: Optional[Chroma]) -> tuple[List[Document], List[str]]:
    """
    Ensure we have laptop_docs + ids in memory AND docs_index.csv on disk.
    Priority:
      1) Use existing docs_index.csv
      2) Build from CSV if available
      3) Rebuild from Chroma (if db is not None)
    """
    if INDEX_CSV.exists():
        return build_from_index()

    csv = get_csv_path()
    if csv and csv.exists():
        return build_from_csv(csv)

    if db is not None:
        return rebuild_index_from_chroma(db)

    _log("[vector] Warning: cannot build index (no CSV and no Chroma).")
    return [], []

# =========================
# Lazy init of Chroma + BM25 (no heavy work on import)
# =========================
def init_stores():
    global INIT_DONE, vector_db, bm25_retriever, laptop_docs
    if INIT_DONE:
        return

    if chroma_missing(CHROMA_DIR):
        _log("[vector] No Chroma DB found → building index from CSV and creating Chroma")
        laptop_docs, doc_ids = ensure_index_built(db=None)
        if not laptop_docs:
            raise FileNotFoundError(
                f"No Chroma DB and no CSV to build index.\nExpected CSV at {CSV_PATH_1} or {CSV_PATH_2}"
            )
        vector_db = Chroma(
            collection_name="laptop-specs",
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )
        vector_db.add_documents(laptop_docs, ids=doc_ids)
        _log("[vector] Chroma DB created and persisted")
    else:
        _log("[vector] Using existing Chroma DB")
        vector_db = Chroma(
            collection_name="laptop-specs",
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )
        # Ensure we have a usable BM25 corpus and docs_index.csv even if CSV isn't present
        laptop_docs, _ = ensure_index_built(db=vector_db)

    # Build BM25 corpus
    bm25_retriever = BM25Retriever(laptop_docs)
    INIT_DONE = True

# =========================
# Dense retrieval helpers
# =========================
def dense_search(user_query: str, k: int = 3) -> List[Document]:
    init_stores()
    return vector_db.similarity_search(user_query, k=k)

def dense_search_with_score(user_query: str, k: int = 3) -> List[Tuple[Document, float]]:
    init_stores()
    try:
        return vector_db.similarity_search_with_score(user_query, k=k) or []
    except Exception:
        docs = vector_db.similarity_search(user_query, k=k) or []
        return [(d, 0.0) for d in docs]

# =========================
# BM25 retriever
# =========================
class BM25Retriever:
    def __init__(self, docs_input: List[Document]):
        if not docs_input:
            self.docs: List[str] = []
            self.tokenized: List[List[str]] = []
            self.bm25 = None
            self.doc_map: Dict[int, Document] = {}
        else:
            self.docs = [d.page_content for d in docs_input]
            self.tokenized = [t.lower().split() for t in self.docs]
            self.bm25 = BM25Okapi(self.tokenized)
            self.doc_map = {i: d for i, d in enumerate(docs_input)}

    def search(self, query: str, k: int = 3) -> List[Document]:
        if not self.docs or not self.bm25:
            return []
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.doc_map[i] for i in top_ids]

# =========================
# Learned alpha loader  >>> UPDATED <<<
# =========================
def _load_learned_alpha(default: float = 0.6) -> float:
    """
    Prefer Outputs/alpha_eval/summary.json, then fall back to Collection/init_eval/alpha.json.
    Both must contain a numeric 'alpha' in [0,1].
    """
    # 1) Preferred: summary.json written by the trainer
    try:
        if ALPHA_SUMMARY_PATH.exists():
            data = json.loads(ALPHA_SUMMARY_PATH.read_text(encoding="utf-8"))
            a = data.get("alpha")
            if isinstance(a, (int, float)) and 0.0 <= a <= 1.0:
                return float(a)
    except Exception:
        pass

    # 2) Legacy fallback
    try:
        if ALPHA_JSON_PATH.exists():
            data = json.loads(ALPHA_JSON_PATH.read_text(encoding="utf-8"))
            a = data.get("alpha")
            if isinstance(a, (int, float)) and 0.0 <= a <= 1.0:
                return float(a)
    except Exception:
        pass

    return default

# =========================
# REQUIRED WRAPPERS (used by src/alpha/data.py)
# =========================
def bm25_get(query: str, k: int) -> List[Document]:
    """Return top-k BM25 Documents for a query."""
    init_stores()
    return bm25_retriever.search(query, k=k) if bm25_retriever else []

def dense_get_with_score(query: str, k: int) -> List[Tuple[Document, float]]:
    """Return top-k (Document, distance/score) from Chroma."""
    return dense_search_with_score(query, k=k)

# =========================
# Hybrid search (rank fusion) – uses learned α by default
# =========================
def hybrid_search(query: str, k: int = 5, alpha: Optional[float] = None) -> List[Document]:
    """
    Rank-fuse dense + BM25 using alpha (dense weight).
    If alpha is None, load learned alpha from summary/alpha.json.
    """
    init_stores()
    if alpha is None:
        alpha = _load_learned_alpha(0.6)

    dense_docs = [d for d, _ in dense_get_with_score(query, k=k)]  # keep dense rank
    bm25_docs  = bm25_get(query, k=k)

    combined: Dict[str, float] = {}
    for rank, d in enumerate(dense_docs):
        combined[d.page_content] = combined.get(d.page_content, 0.0) + alpha * (k - rank)
    for rank, d in enumerate(bm25_docs):
        combined[d.page_content] = combined.get(d.page_content, 0.0) + (1 - alpha) * (k - rank)

    doc_map = {d.page_content: d for d in dense_docs + bm25_docs}
    ordered = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [doc_map[c] for c, _ in ordered]

# =========================
# CLI (unchanged)
# =========================
if __name__ == "__main__":
    import argparse, shutil

    parser = argparse.ArgumentParser(description="Laptop RAG vector store utilities")
    parser.add_argument("--status", action="store_true", help="Print paths and what exists")
    parser.add_argument("--init", action="store_true", help="Ensure Chroma + docs_index exist; build if needed")
    parser.add_argument("--rebuild-from-csv", action="store_true",
                        help="Rebuild docs_index.csv (and markdown) from CSV; do not touch Chroma")
    parser.add_argument("--rebuild-from-chroma", action="store_true",
                        help="Rebuild docs_index.csv (and markdown) by reading existing Chroma")
    parser.add_argument("--wipe-index", action="store_true",
                        help="Delete data_index/* before rebuilding (DANGEROUS)")
    parser.add_argument("--wipe-chroma", action="store_true",
                        help="Delete chroma_db/* (DANGEROUS)")

    args = parser.parse_args()

    def _exists(p: Path) -> str:
        return "exists" if p.exists() else "missing"

    if args.wipe_index:
        if INDEX_DIR.exists():
            _log(f"[cli] Wiping INDEX_DIR: {INDEX_DIR}")
            shutil.rmtree(INDEX_DIR)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if args.wipe_chroma:
        if CHROMA_DIR.exists():
            _log(f"[cli] Wiping CHROMA_DIR: {CHROMA_DIR}")
            shutil.rmtree(CHROMA_DIR)
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    if args.status:
        _log("[cli] Status")
        _log(f"ROOT_DIR   : {ROOT_DIR}")
        _log(f"CHROMA_DIR : {CHROMA_DIR} ({_exists(CHROMA_DIR)})")
        _log(f"INDEX_DIR  : {INDEX_DIR} ({_exists(INDEX_DIR)})")
        _log(f"INDEX_CSV  : {INDEX_CSV} ({_exists(INDEX_CSV)})")
        csv = get_csv_path()
        _log(f"CSV        : {csv if csv else 'None found'}")
        if INDEX_CSV.exists():
            try:
                df = pd.read_csv(INDEX_CSV)
                _log(f"docs_index rows: {len(df)}")
            except Exception as e:
                _log(f"docs_index read error: {e}")

    ran_action = False

    if args.rebuild_from_csv:
        csv = get_csv_path()
        if not csv:
            raise FileNotFoundError(f"No CSV found at {CSV_PATH_1} or {CSV_PATH_2}")
        _log(f"[cli] Rebuilding docs_index from CSV: {csv}")
        build_from_csv(csv)
        ran_action = True

    if args.rebuild_from_chroma:
        _log("[cli] Rebuilding docs_index from existing Chroma")
        db = Chroma(collection_name="laptop-specs",
                    embedding_function=embeddings,
                    persist_directory=str(CHROMA_DIR))
        rebuild_index_from_chroma(db)
        ran_action = True

    if args.init:
        _log("[cli] init_stores()")
        init_stores()
        ran_action = True

    if not (args.status or ran_action):
        _log("Nothing to do. Try one of:")
        _log("  --status | --init | --rebuild-from-csv | --rebuild-from-chroma")
