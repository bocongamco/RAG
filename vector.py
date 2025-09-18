from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

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
