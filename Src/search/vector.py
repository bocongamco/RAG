"""
retrieval pipeline for Laptop RAG
---------------------------------------
- Works with either OllamaEmbeddings (default) or HuggingFaceEmbeddings.
  Use env: EMBED_BACKEND=ollama|hf  (default: ollama)
  - For HF, set EMBED_MODEL (default: sentence-transformers/all-MiniLM-L6-v2)
  - For Ollama, set OLLAMA_EMBED_MODEL (default: mxbai-embed-large)
- Finds CSV from multiple locations and encodings.
- Builds/uses Chroma at chroma_db/ (prefers existing chrome_langchain_db if present).
- Preserves docs_index.csv + Markdown buckets + meta.json.
- Optional knowledge_base/ to a separate knowledge Chroma store.
- BM25 + Dense + Hybrid (with learned alpha from Outputs/alpha_eval/summary.json
  or Collection/init_eval/alpha.json).

CLI
  python vector_merged.py --status | --init | --rebuild-from-csv | --rebuild-from-chroma \
                         [--wipe-index] [--wipe-chroma]
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import os, json, hashlib, re, shutil
from datetime import datetime

import pandas as pd
import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

# Prefer the modern chroma import; fallback to community if necessary
try:  # langchain-chroma >=0.1
    from langchain_chroma import Chroma  # type: ignore
except Exception:  # fallback
    from langchain_community.vectorstores import Chroma  # type: ignore

# -----------------------------
# Embedding backend selection
# -----------------------------
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "ollama").lower().strip()  # "ollama" | "hf"

_embeddings_obj = None

def _load_embeddings():
    global _embeddings_obj
    if _embeddings_obj is not None:
        return _embeddings_obj
    if EMBED_BACKEND == "hf":
        model = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings_obj = HuggingFaceEmbeddings(model_name=model)
    else:  # default: ollama
        model = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
        from langchain_ollama import OllamaEmbeddings
        _embeddings_obj = OllamaEmbeddings(model=model)
    return _embeddings_obj

# -----------------------------
# Logging helper
# -----------------------------

def _log(msg: str):
    print(msg)

# -----------------------------
# Paths (repo-aware, but robust)
# -----------------------------
THIS = Path(__file__).resolve()
# If this file lives under src/search/, repo root is parents[2]; else fall back to parent
REPO_CANDIDATE = THIS.parents[2] if len(THIS.parents) >= 3 else THIS.parent
ROOT_DIR = REPO_CANDIDATE if (REPO_CANDIDATE / ".git").exists() or (REPO_CANDIDATE / "data").exists() else THIS.parent

DATA_DIR   = ROOT_DIR / "data"
ART_DIR    = ROOT_DIR / "Collection"
EVAL_DIR   = ART_DIR / "init_eval"
EDA_DIR    = ART_DIR / "eda"
INDEX_DIR  = ROOT_DIR / "data_index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Chroma: prefer chroma_db/, but if legacy chrome_langchain_db/ exists, use that.
CHROMA_DIR_NEW = ROOT_DIR / "chroma_db"
CHROMA_DIR_OLD = ROOT_DIR / "chrome_langchain_db"
CHROMA_DIR = CHROMA_DIR_OLD if CHROMA_DIR_OLD.exists() else CHROMA_DIR_NEW
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Knowledge base (optional)
KNOWLEDGE_DIR      = ROOT_DIR / "knowledge_base"
KNOWLEDGE_CHROMA   = ROOT_DIR / "knowledge_chroma_db"

# Index outputs
INDEX_CSV  = INDEX_DIR / "docs_index.csv"
DOCS_DIR   = INDEX_DIR / "docs"
DOCID_PATH = INDEX_DIR / "docid"
META_PATH  = INDEX_DIR / "meta.json"

# Alpha files (learned)
ALPHA_SUMMARY_PATH = ROOT_DIR / "Outputs" / "alpha_eval" / "summary.json"
ALPHA_JSON_PATH    = EVAL_DIR / "alpha.json"

# CSV candidates (checked at runtime)
CSV_CANDIDATES = [
    DATA_DIR / "Training-Data" / "Amazon_Laptop_Specs_utf8.csv",
    DATA_DIR / "Amazon_Laptop_Specs_utf8.csv",
    ROOT_DIR / "training-data" / "Amazon_Laptop_Specs.csv",  # legacy location/encoding
    DATA_DIR / "Training-Data" / "Amazon_Laptop_Specs.csv",
    DATA_DIR / "Amazon_Laptop_Specs.csv",
]

# -----------------------------
# Helpers
# -----------------------------

def _safe_str(val) -> str:
    if isinstance(val, float) and pd.isna(val):
        return ""
    return str(val).strip()

_spec_ram = re.compile(r"(\d+)\s*gb(?:\s+(?:ddr\d?|lpddr\d?|ram))?", re.I)
_spec_screen = re.compile(r"(\d+\.?\d*)\s*[\"']?\s*inch", re.I)

def _extract_specs_from_name(name: str) -> dict:
    specs: Dict[str, str] = {}
    nl = name.lower()
    m = _spec_ram.search(nl)
    if m:
        specs["ram"] = f"{m.group(1)}GB"
    m = _spec_screen.search(nl)
    if m:
        specs["screen"] = f"{m.group(1)} inch"
    if any(k in nl for k in ("i7", "core i7")):
        specs["cpu"] = "Intel Core i7"
    elif any(k in nl for k in ("i5", "core i5")):
        specs["cpu"] = "Intel Core i5"
    elif any(k in nl for k in ("i3", "core i3")):
        specs["cpu"] = "Intel Core i3"
    elif "ryzen 7" in nl:
        specs["cpu"] = "AMD Ryzen 7"
    elif "ryzen 5" in nl:
        specs["cpu"] = "AMD Ryzen 5"
    if any(w in nl for w in ["gaming", "gtx", "rtx", "geforce", "radeon rx"]):
        specs["category"] = "Gaming"
    return specs


def _row_to_text(i: int, row: pd.Series) -> tuple[str, str, str, str]:
    s = _safe_str
    name = s(row.get("Name"))
    price = s(row.get("Price"))  # keep as raw string to avoid locale/parse bugs
    rating = s(row.get("Customer Rating"))
    ratings_count = s(row.get("Number of Ratings"))
    dimensions = s(row.get("Item Dimensions LxWxH"))
    rank = s(row.get("Best Sellers Rank"))
    net_qty = s(row.get("Net Quantity"))
    generic = s(row.get("Generic Name"))

    specs = _extract_specs_from_name(name)

    content = (
        f"ROW={i} | NAME={name} | PRICE={price} | "
        f"RAM={specs.get('ram','Not specified')} | SCREEN_SIZE={specs.get('screen','Not specified')} | "
        f"CPU={specs.get('cpu','Not specified')} | CATEGORY={specs.get('category','Standard')} | "
        f"RATING={rating} | RATINGS_COUNT={ratings_count} | DIMENSIONS={dimensions} | RANK={rank} | "
        f"NET_QTY={net_qty} | GENERIC={generic}"
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
            f"# {r['name']}",
            "",
            f"doc_id: {r['doc_id']}",
            f"row: {r['row']}",
            f"price: {r['price']}",
            f"rating: {r['rating']}",
            "",
            "## Content",
            r["content"],
        ]
        (out_dir / f"{r['doc_id']}.md").write_text("\n".join(md), encoding="utf-8")

    with DOCID_PATH.open("w", encoding="utf-8") as f:
        for r in index_rows:
            name = '"' + str(r['name']).replace('"', '""') + '"'
            f.write(f"{int(r['row'])},{r['doc_id']},{name}\n")

    META_PATH.write_text(
        json.dumps(
            {
                "collection": "laptop-specs",
                "count": len(index_rows),
                "created": datetime.utcnow().isoformat() + "Z",
                "schema": ["doc_id", "content", "row", "name", "price", "rating"],
                "bucket_policy": {"type": "row_range", "bucket_size": 100, "format": "P%02d"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

# -----------------------------
# CSV loaders / index builders
# -----------------------------

def _pick_csv() -> Optional[Path]:
    for p in CSV_CANDIDATES:
        if p.exists():
            return p
    return None


def _read_csv_any(csv_path: Path) -> pd.DataFrame:
    # Try utf-8 first; fall back to cp1252 (legacy file)
    try:
        return pd.read_csv(csv_path)
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, encoding="cp1252")


def build_from_csv(csv_path: Optional[Path] = None) -> tuple[List[Document], List[str]]:
    csv = csv_path or _pick_csv()
    if not csv or not csv.exists():
        raise FileNotFoundError("No CSV found in expected locations.")

    df = _read_csv_any(csv)
    docs: List[Document] = []
    ids: List[str] = []
    idx_rows: List[dict] = []

    for i, row in df.iterrows():
        content, name, price, rating = _row_to_text(i, row)
        doc_id = hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]
        docs.append(Document(page_content=content, metadata={"row": str(i), "name": name, "doc_id": doc_id}))
        ids.append(doc_id)
        idx_rows.append({
            "doc_id": doc_id,
            "content": content,
            "row": int(i),
            "name": name,
            "price": price,
            "rating": rating,
        })

    pd.DataFrame(idx_rows, columns=["doc_id", "content", "row", "name", "price", "rating"]).to_csv(INDEX_CSV, index=False)
    _dump_markdown(idx_rows)
    _log(f"[vector] Built docs_index.csv with {len(idx_rows)} rows from {csv.name}")
    return docs, ids


def build_from_index() -> tuple[List[Document], List[str]]:
    if not INDEX_CSV.exists():
        raise FileNotFoundError(f"Missing {INDEX_CSV}")

    df = pd.read_csv(INDEX_CSV)
    required = {"doc_id", "content", "row", "name", "price", "rating"}
    if not required.issubset(df.columns):
        missing = sorted(list(required - set(df.columns)))
        raise ValueError(f"docs_index.csv missing required columns: {missing}")

    docs = [
        Document(
            page_content=row["content"],
            metadata={"row": str(int(row["row"])), "name": row["name"], "doc_id": row["doc_id"]},
        )
        for _, row in df.iterrows()
    ]
    ids = df["doc_id"].tolist()
    _dump_markdown(df.to_dict(orient="records"))
    _log(f"[vector] Loaded {len(docs)} docs from docs_index.csv")
    return docs, ids


def rebuild_index_from_chroma(db: Chroma) -> tuple[List[Document], List[str]]:
    _log("[vector] Attempting to rebuild docs_index from Chroma collection...")
    data = db.get(include=["documents", "metadatas", "ids"]) or {}
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    ids = data.get("ids") or []

    if not documents:
        _log("[vector] Chroma returned no documents; cannot rebuild index.")
        return [], []

    idx_rows: List[dict] = []
    docs: List[Document] = []
    for i, (doc_text, meta, did) in enumerate(zip(documents, metadatas, ids)):
        meta = meta or {}
        name = meta.get("name") or ""
        row = meta.get("row")
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
            "rating": "",
        })
        docs.append(Document(page_content=doc_text, metadata={"row": str(row_int), "name": name, "doc_id": did}))

    pd.DataFrame(idx_rows, columns=["doc_id", "content", "row", "name", "price", "rating"]).to_csv(INDEX_CSV, index=False)
    _dump_markdown(idx_rows)
    _log(f"[vector] Rebuilt docs_index.csv from Chroma with {len(idx_rows)} rows")
    return docs, ids


# -----------------------------
# Knowledge base (optional)
# -----------------------------

def build_knowledge_base() -> Optional[Chroma]:
    if not KNOWLEDGE_DIR.exists():
        _log(f"[kb] No knowledge_base folder at {KNOWLEDGE_DIR}")
        return None

    txts = list(KNOWLEDGE_DIR.glob("*.txt"))
    if not txts:
        _log("[kb] No .txt files in knowledge_base/")
        return None

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    all_docs: List[Document] = []
    for p in txts:
        text = p.read_text(encoding="utf-8")
        for chunk in splitter.split_text(text):
            all_docs.append(Document(page_content=chunk, metadata={"source": str(p)}))

    emb = _load_embeddings()
    kb = Chroma(documents=all_docs, embedding_function=emb, persist_directory=str(KNOWLEDGE_CHROMA))
    _log(f"[kb] Knowledge base created with {len(all_docs)} chunks at {KNOWLEDGE_CHROMA}")
    return kb


def knowledge_search(q: str, k: int = 3):
    if not KNOWLEDGE_CHROMA.exists():
        return []
    emb = _load_embeddings()
    kb = Chroma(embedding_function=emb, persist_directory=str(KNOWLEDGE_CHROMA))
    return kb.similarity_search(q, k=k)


# -----------------------------
# Lazy init of stores
# -----------------------------
INIT_DONE = False
vector_db: Optional[Chroma] = None
bm25_retriever = None  # type: ignore
laptop_docs: List[Document] = []


def chroma_missing(path: Path | str) -> bool:
    return not os.path.exists(os.path.join(str(path), "chroma.sqlite3"))


def ensure_index_built(db: Optional[Chroma]) -> tuple[List[Document], List[str]]:
    if INDEX_CSV.exists():
        return build_from_index()
    csv = _pick_csv()
    if csv and csv.exists():
        return build_from_csv(csv)
    if db is not None:
        return rebuild_index_from_chroma(db)
    _log("[vector] Warning: cannot build index (no CSV and no Chroma).")
    return [], []


def init_stores():
    global INIT_DONE, vector_db, bm25_retriever, laptop_docs
    if INIT_DONE:
        return

    emb = _load_embeddings()

    if chroma_missing(CHROMA_DIR):
        _log("[vector] No Chroma DB found → building index from CSV and creating Chroma")
        laptop_docs, doc_ids = ensure_index_built(db=None)
        if not laptop_docs:
            raise FileNotFoundError("No Chroma DB and no CSV to build index.")
        vector_db = Chroma(collection_name="laptop-specs", embedding_function=emb, persist_directory=str(CHROMA_DIR))
        vector_db.add_documents(laptop_docs, ids=doc_ids)
        _log("[vector] Chroma DB created and persisted")
    else:
        _log("[vector] Using existing Chroma DB")
        vector_db = Chroma(collection_name="laptop-specs", embedding_function=emb, persist_directory=str(CHROMA_DIR))
        laptop_docs, _ = ensure_index_built(db=vector_db)

    bm25_retriever = BM25Retriever(laptop_docs)
    INIT_DONE = True


# -----------------------------
# Dense / BM25 / Hybrid
# -----------------------------
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


def dense_search(user_query: str, k: int = 3) -> List[Document]:
    init_stores()
    return vector_db.similarity_search(user_query, k=k)  # type: ignore


def dense_search_with_score(user_query: str, k: int = 3) -> List[Tuple[Document, float]]:
    init_stores()
    try:
        return vector_db.similarity_search_with_score(user_query, k=k) or []  # type: ignore
    except Exception:
        docs = vector_db.similarity_search(user_query, k=k) or []  # type: ignore
        return [(d, 0.0) for d in docs]


def bm25_get(query: str, k: int) -> List[Document]:
    init_stores()
    return bm25_retriever.search(query, k=k) if bm25_retriever else []


def _load_learned_alpha(default: float = 0.6) -> float:
    try:
        if ALPHA_SUMMARY_PATH.exists():
            data = json.loads(ALPHA_SUMMARY_PATH.read_text(encoding="utf-8"))
            a = data.get("alpha")
            if isinstance(a, (int, float)) and 0.0 <= a <= 1.0:
                return float(a)
    except Exception:
        pass
    try:
        if ALPHA_JSON_PATH.exists():
            data = json.loads(ALPHA_JSON_PATH.read_text(encoding="utf-8"))
            a = data.get("alpha")
            if isinstance(a, (int, float)) and 0.0 <= a <= 1.0:
                return float(a)
    except Exception:
        pass
    return default


def hybrid_search(query: str, k: int = 5, alpha: Optional[float] = None) -> List[Document]:
    init_stores()
    if alpha is None:
        alpha = _load_learned_alpha(0.6)

    dense_docs = [d for d, _ in dense_search_with_score(query, k=k)]
    bm25_docs  = bm25_get(query, k=k)

    combined: Dict[str, float] = {}
    for rank, d in enumerate(dense_docs):
        combined[d.page_content] = combined.get(d.page_content, 0.0) + float(alpha) * (k - rank)
    for rank, d in enumerate(bm25_docs):
        combined[d.page_content] = combined.get(d.page_content, 0.0) + (1.0 - float(alpha)) * (k - rank)

    doc_map = {d.page_content: d for d in dense_docs + bm25_docs}
    ordered = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [doc_map[c] for c, _ in ordered]


# -----------------------------
# Build-all helpers (legacy main)
# -----------------------------

def build_all():
    """Legacy-style one-shot build from CSV + knowledge base."""
    emb = _load_embeddings()
    if chroma_missing(CHROMA_DIR):
        docs, ids = build_from_csv()
        db = Chroma(collection_name="laptop-specs", embedding_function=emb, persist_directory=str(CHROMA_DIR))
        db.add_documents(docs, ids=ids)
        _log("[build] Chroma DB created and persisted")
    else:
        _log("[build] Chroma already present")
    build_knowledge_base()


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merged Laptop RAG vector utilities")
    parser.add_argument("--status", action="store_true", help="Print paths and what exists")
    parser.add_argument("--init", action="store_true", help="Ensure Chroma + docs_index; build if needed")
    parser.add_argument("--rebuild-from-csv", action="store_true", help="Rebuild docs_index.csv (no Chroma touch)")
    parser.add_argument("--rebuild-from-chroma", action="store_true", help="Rebuild docs_index from existing Chroma")
    parser.add_argument("--wipe-index", action="store_true", help="Delete data_index/* (DANGEROUS)")
    parser.add_argument("--wipe-chroma", action="store_true", help="Delete chroma_db/* or chrome_langchain_db/* (DANGEROUS)")

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
        csv = _pick_csv()
        _log(f"CSV        : {csv if csv else 'None found'}")
        if INDEX_CSV.exists():
            try:
                df = pd.read_csv(INDEX_CSV)
                _log(f"docs_index rows: {len(df)}")
            except Exception as e:
                _log(f"docs_index read error: {e}")

    ran_action = False

    if args.rebuild_from_csv:
        csv = _pick_csv()
        if not csv:
            raise FileNotFoundError("No CSV found in expected locations")
        _log(f"[cli] Rebuilding docs_index from CSV: {csv}")
        build_from_csv(csv)
        ran_action = True

    if args.rebuild_from_chroma:
        _log("[cli] Rebuilding docs_index from existing Chroma")
        emb = _load_embeddings()
        db = Chroma(collection_name="laptop-specs", embedding_function=emb, persist_directory=str(CHROMA_DIR))
        rebuild_index_from_chroma(db)
        ran_action = True

    if args.init:
        _log("[cli] init_stores()")
        init_stores()
        ran_action = True

    if not (args.status or ran_action):
        _log("Nothing to do. Try one of: --status | --init | --rebuild-from-csv | --rebuild-from-chroma")
