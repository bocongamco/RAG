# vector.py
# Builds vector DB from CSV + knowledge PDFs

import os
import re
import pandas as pd
from pathlib import Path
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import numpy as np

# ============
# CONFIG
# ============
PROJECT_DIR = Path(__file__).resolve().parent
_CSV = PROJECT_DIR / "training-data/Amazon_Laptop_Specs.csv"
_DOCS_INDEX = PROJECT_DIR / "data_index/docs_index.csv"
_CHROMA_DIR = PROJECT_DIR / "chrome_langchain_db"
_KNOWLEDGE_DIR = PROJECT_DIR / "knowledge_base"
_KNOWLEDGE_CHROMA = PROJECT_DIR / "knowledge_chroma_db"

_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ============
# HELPERS
# ============
def _safe_str(val) -> str:
    """Convert value to string, handling NaN."""
    if pd.isna(val):
        return ""
    return str(val).strip()

def _extract_specs_from_name(name: str) -> dict:
    """Extract RAM, screen size, processor from product name."""
    specs = {}
    name_lower = name.lower()
    
    # Extract RAM (e.g., "8GB", "16 GB RAM")
    if ram_match := re.search(r'(\d+)\s*gb(?:\s+(?:ddr\d?|lpddr\d?|ram))?', name_lower):
        specs['ram'] = f"{ram_match.group(1)}GB"
    
    # Extract screen size (e.g., "15.6 inch", '13"')
    if screen_match := re.search(r'(\d+\.?\d*)\s*["\']?\s*inch', name_lower):
        specs['screen'] = f"{screen_match.group(1)} inch"
    
    # Extract processor hints
    if 'i7' in name_lower or 'core i7' in name_lower:
        specs['cpu'] = 'Intel Core i7'
    elif 'i5' in name_lower or 'core i5' in name_lower:
        specs['cpu'] = 'Intel Core i5'
    elif 'i3' in name_lower or 'core i3' in name_lower:
        specs['cpu'] = 'Intel Core i3'
    elif 'ryzen 7' in name_lower:
        specs['cpu'] = 'AMD Ryzen 7'
    elif 'ryzen 5' in name_lower:
        specs['cpu'] = 'AMD Ryzen 5'
    
    # Detect gaming keywords
    if any(word in name_lower for word in ['gaming', 'gtx', 'rtx', 'geforce', 'radeon rx']):
        specs['category'] = 'Gaming'
    
    return specs

def _row_to_text(i: int, row: pd.Series) -> tuple[str, str, str, str]:
    """Convert CSV row to searchable text."""
    s = _safe_str
    
    # Core fields from CSV
    name = s(row.get("Name"))
    price = s(row.get("Price"))
    rating = s(row.get("Customer Rating"))
    ratings_count = s(row.get("Number of Ratings"))
    dimensions = s(row.get("Item Dimensions LxWxH"))
    rank = s(row.get("Best Sellers Rank"))
    
    # Extract specs from product name
    specs = _extract_specs_from_name(name)
    
    # Build searchable content
    content = (
        f"ROW={i} | NAME={name} | PRICE={price} | "
        f"RAM={specs.get('ram', 'Not specified')} | "
        f"SCREEN_SIZE={specs.get('screen', 'Not specified')} | "
        f"CPU={specs.get('cpu', 'Not specified')} | "
        f"CATEGORY={specs.get('category', 'Standard')} | "
        f"RATING={rating} | RATINGS_COUNT={ratings_count} | "
        f"DIMENSIONS={dimensions} | RANK={rank}"
    )
    
    return content, name, price, rating

def build_docs_from_csv():
    """Read CSV and create Documents."""
    print("No Chroma DB or index found → embedding original CSV")
    
    # Load CSV
    df = pd.read_csv(_CSV, encoding='cp1252')
    
    docs = []
    rows_data = []
    
    for i, row in df.iterrows():
        content, name, price, rating = _row_to_text(i, row)
        
        doc = Document(
            page_content=content,
            metadata={
                "row": str(i),
                "name": name,
                "price": price,
                "rating": rating
            }
        )
        docs.append(doc)
        
        rows_data.append({
            "row": i,
            "name": name,
            "price": price,
            "rating": rating,
            "created": datetime.now().isoformat() + "Z",
        })
    
    # Save index
    os.makedirs(_DOCS_INDEX.parent, exist_ok=True)
    pd.DataFrame(rows_data).to_csv(_DOCS_INDEX, index=False)
    print(f"Built docs_index.csv with {len(rows_data)} rows")
    
    return docs

def build_chroma_db(docs: list):
    """Build vector DB from documents."""
    embeddings = HuggingFaceEmbeddings(model_name=_EMBED_MODEL)
    
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(_CHROMA_DIR)
    )
    
    print("Chroma DB created and persisted")
    return vectorstore

def build_knowledge_base():
    """Build knowledge base from text files."""
    print("Building knowledge base from text files...")
    
    if not _KNOWLEDGE_DIR.exists():
        print(f"No knowledge_base folder found at {_KNOWLEDGE_DIR}")
        return None
    
    text_files = list(_KNOWLEDGE_DIR.glob("*.txt"))
    if not text_files:
        print("No .txt files found in knowledge_base/")
        return None
    
    all_docs = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    for txt_file in text_files:
        text = txt_file.read_text(encoding='utf-8')
        chunks = text_splitter.split_text(text)
        
        for chunk in chunks:
            doc = Document(
                page_content=chunk,
                metadata={"source": str(txt_file)}
            )
            all_docs.append(doc)
    
    print(f"Loaded {len(all_docs)} knowledge chunks from {len(text_files)} files")
    
    embeddings = HuggingFaceEmbeddings(model_name=_EMBED_MODEL)
    
    knowledge_store = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=str(_KNOWLEDGE_CHROMA)
    )
    
    print(f"Knowledge base created with {len(all_docs)} chunks")
    return knowledge_store

# ============
# RETRIEVAL
# ============
def dense_search(q: str, k: int = 3):
    """Vector similarity search."""
    if not _CHROMA_DIR.exists():
        return []
    
    embeddings = HuggingFaceEmbeddings(model_name=_EMBED_MODEL)
    vectorstore = Chroma(
        persist_directory=str(_CHROMA_DIR),
        embedding_function=embeddings
    )
    
    return vectorstore.similarity_search(q, k=k)

def knowledge_search(q: str, k: int = 3):
    """Search knowledge base."""
    if not _KNOWLEDGE_CHROMA.exists():
        return []
    
    embeddings = HuggingFaceEmbeddings(model_name=_EMBED_MODEL)
    knowledge_store = Chroma(
        persist_directory=str(_KNOWLEDGE_CHROMA),
        embedding_function=embeddings
    )
    
    return knowledge_store.similarity_search(q, k=k)

# ============
# BM25
# ============
class BM25Retriever:
    """BM25 retriever for lexical search."""
    
    def __init__(self):
        self.corpus = []
        self.docs = []
        self.bm25 = None
        self._load()
    
    def _load(self):
        """Load documents from index."""
        if not _DOCS_INDEX.exists():
            return
        
        df = pd.read_csv(_DOCS_INDEX)
        
        for _, row in df.iterrows():
            content = f"ROW={row['row']} | NAME={row['name']} | PRICE={row['price']} | RATING={row['rating']}"
            
            doc = Document(
                page_content=content,
                metadata={
                    "row": str(row["row"]),
                    "name": row["name"],
                    "price": str(row["price"]),
                    "rating": str(row["rating"])
                }
            )
            
            self.docs.append(doc)
            self.corpus.append(content.lower().split())
        
        if self.corpus:
            self.bm25 = BM25Okapi(self.corpus)
    
    def search(self, query: str, k: int = 3):
        """Search using BM25."""
        if not self.bm25:
            return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:k]
        
        return [self.docs[i] for i in top_indices]

# Global BM25 retriever instance
bm25_retriever = BM25Retriever()

# ============
# HYBRID
# ============
def hybrid_search(q: str, k: int = 3, alpha: float = 0.6):
    """Combine dense and BM25 with rank fusion."""
    dense_docs = dense_search(q, k=k * 2)
    bm25_docs = bm25_retriever.search(q, k=k * 2)
    
    score_map = {}
    
    for rank, doc in enumerate(dense_docs):
        row = doc.metadata.get("row")
        score_map[row] = score_map.get(row, 0) + alpha * (1.0 / (rank + 1))
    
    for rank, doc in enumerate(bm25_docs):
        row = doc.metadata.get("row")
        score_map[row] = score_map.get(row, 0) + (1 - alpha) * (1.0 / (rank + 1))
    
    sorted_rows = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:k]
    
    all_docs = {d.metadata.get("row"): d for d in dense_docs + bm25_docs}
    return [all_docs[row] for row, _ in sorted_rows if row in all_docs]

# ============
# MAIN
# ============
if __name__ == "__main__":
    docs = build_docs_from_csv()
    build_chroma_db(docs)
    build_knowledge_base()
