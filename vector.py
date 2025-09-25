<<<<<<< HEAD
# vector.py
# Retrieval pipeline for the Laptop RAG project.

=======
>>>>>>> origin/main
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
<<<<<<< HEAD
import hashlib
from pathlib import Path
from typing import List
from rank_bm25 import BM25Okapi
import json
from datetime import datetime

# -------------------
# Paths
# -------------------
PROJECT_DIR   = Path(__file__).resolve().parent
CSV_PATH      = PROJECT_DIR / "training-data" / "Amazon_Laptop_Specs.csv"

CHROMA_DIR    = PROJECT_DIR / "chrome_langchain_db"

INDEX_DIR     = PROJECT_DIR / "data_index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

INDEX_CSV     = INDEX_DIR / "docs_index.csv"
DOCS_DIR      = INDEX_DIR / "docs"
DOCID_PATH    = INDEX_DIR / "docid"
META_PATH     = INDEX_DIR / "meta.json"

# -------------------
# Embeddings (Ollama)
# -------------------
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# -------------------
# Helpers
# -------------------
def chroma_missing(path: Path | str) -> bool:
    return not os.path.exists(os.path.join(str(path), "chroma.sqlite3"))

def _safe_str(v) -> str:
    import pandas as _pd
    return "" if (isinstance(v, float) and _pd.isna(v)) else str(v)

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

    # docid (CSV-like lines)
    with DOCID_PATH.open("w", encoding="utf-8") as f:
        for r in index_rows:
            # quote name to avoid extra commas breaking the format
            name = '"' + str(r['name']).replace('"', '""') + '"'
            f.write(f"{int(r['row'])},{r['doc_id']},{name}\n")

    META_PATH.write_text(json.dumps({
        "collection": "laptop-specs",
        "count": len(index_rows),
        "created": datetime.utcnow().isoformat() + "Z",
        "schema": ["doc_id","content","row","name","price","rating"],
        "bucket_policy": {"type": "row_range", "bucket_size": 100, "format": "P%02d"}
    }, indent=2), encoding="utf-8")

# -------------------
# Build index
# -------------------
def build_from_csv():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing CSV at {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    docs, ids, idx_rows = [], [], []
    for i, row in df.iterrows():
        content, name, price, rating = _row_to_text(i, row)
        doc_id = hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]
        docs.append(Document(page_content=content, metadata={"row": str(i), "name": name}))
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
    print(f"Built docs_index.csv with {len(idx_rows)} rows")
    _dump_markdown(idx_rows)
    return docs, ids

def build_from_index():
    df = pd.read_csv(INDEX_CSV)
    req = {"doc_id","content","row","name","price","rating"}
    if not req.issubset(df.columns):
        raise ValueError(f"docs_index.csv missing required columns: {sorted(list(req - set(df.columns)))}")

    docs = [
        Document(page_content=row["content"], metadata={"row": str(int(row["row"])), "name": row["name"]})
        for _, row in df.iterrows()
    ]
    ids = df["doc_id"].tolist()
    _dump_markdown(df.to_dict(orient="records"))
    print(f"Loaded {len(docs)} docs from docs_index.csv")
    return docs, ids

# -------------------
# Build or load Chroma + ensure data_index exists
# -------------------
if chroma_missing(CHROMA_DIR):
    # No Chroma: (a) prefer index if present else (b) build from CSV, then write Chroma
    if INDEX_CSV.exists():
        print("No Chroma DB found, but docs_index.csv exists → rebuilding from index")
        laptop_docs, doc_ids = build_from_index()
    else:
        print("No Chroma DB or index found → embedding original CSV")
        laptop_docs, doc_ids = build_from_csv()

    vector_db = Chroma(
        collection_name="laptop-specs",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    vector_db.add_documents(laptop_docs, ids=doc_ids)
    print("Chroma DB created and persisted")
else:
    print("Using existing Chroma DB")
    vector_db = Chroma(
        collection_name="laptop-specs",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    # If index is missing but we have the CSV, (re)build data_index for BM25 transparency.
    if INDEX_CSV.exists():
        laptop_docs, _ = build_from_index()
    elif CSV_PATH.exists():
        print("data_index missing → building from CSV (will NOT modify Chroma)")
        laptop_docs, _ = build_from_csv()
    else:
        print("Warning: data_index and CSV are missing → BM25 disabled")
        laptop_docs = []

# -------------------
# Retrieval
# -------------------
# a) Dense retriever (fixed k=3 for backward-compat; not used by main.py)
dense_retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# b) Dynamic dense search (respects k per call)
def dense_search(user_query: str, k: int = 3):
    return vector_db.similarity_search(user_query, k=k)

# c) BM25 retriever
class BM25Retriever:
    def __init__(self, docs_input: List[Document]):
        if not docs_input:
            self.docs, self.tokenized, self.bm25, self.doc_map = [], [], None, {}
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

bm25_retriever = BM25Retriever(laptop_docs)

# d) Hybrid (rank fusion)
def hybrid_search(query: str, k: int = 3, alpha: float = 0.6) -> List[Document]:
    dense_docs = dense_search(query, k=k) or []
    bm25_docs  = bm25_retriever.search(query, k=k) or []

    combined: dict[str, float] = {}
    for rank, d in enumerate(dense_docs):
        combined[d.page_content] = combined.get(d.page_content, 0.0) + alpha * (k - rank)
    for rank, d in enumerate(bm25_docs):
        combined[d.page_content] = combined.get(d.page_content, 0.0) + (1 - alpha) * (k - rank)

    merged = []
    for text, _ in sorted(combined.items(), key=lambda x: x[1], reverse=True):
        doc = next((dd for dd in dense_docs + bm25_docs if dd.page_content == text), None)
        if doc and doc not in merged:
            merged.append(doc)
        if len(merged) >= k:
            break
    return merged
=======

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "training-data" / "Amazon_Laptop_Specs.csv"
DB_DIR = BASE_DIR / "chrome_langchain_db"


# Load csv file
df = pd.read_csv(CSV_PATH)
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = DB_DIR

def needs_build(path: str) -> bool:
    # Chroma writes a chroma.sqlite3 when a collection is persisted
    return not os.path.exists(os.path.join(path, "chroma.sqlite3"))

add_document = needs_build(db_location)

def s(v):
    return "" if (isinstance(v, float) and pd.isna(v)) else str(v)

def price_num(v):
    import re, math
    txt = s(v)
    m = re.search(r"[\d\.,]+", txt)
    return float(m.group(0).replace(",", "")) if m else None

if add_document:
    documents, ids = [], []
    for i, row in df.iterrows():
        name = s(row.get("Name"))
        price = s(row.get("Price"))
        rating = s(row.get("Customer Rating"))
        ratings_n = s(row.get("Number of Ratings"))
        dims = s(row.get("Item Dimensions LxWxH"))

        # Put the KEY facts up front so retrieval gives the model what it needs
        content = (
            f"ROW={i} | MODEL={name} | PRICE={price} | RATING={rating} | RATINGS={ratings_n} | DIMS={dims}. "
            f"{s(row.get('Best Sellers Rank'))} {s(row.get('Net Quantity'))} {s(row.get('Generic Name'))}"
        )

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "row": str(i),                  # force string so it always survives round-trips
                    "name": name,
                    "price_num": price_num(price),  # numeric—useful later if you want rules
                    "rating_num": float(rating) if rating.replace('.','',1).isdigit() else None,
                },
            )
        )
        ids.append(str(i))

# Create / open vector store
vector_store = Chroma(
    collection_name="laptop-specs",
    embedding_function=embeddings,
    persist_directory=db_location,
)

if add_document:
    vector_store.add_documents(documents, ids=ids)
    vector_store.persist()  # <-- actually writes chroma.sqlite3

# Create retriever from vector store
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
>>>>>>> origin/main
