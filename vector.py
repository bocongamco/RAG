from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# Load csv file
df = pd.read_csv("./training-data/Amazon_Laptop_Specs.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_db"

def needs_build(path: str) -> bool:
    # Chroma writes a chroma.sqlite3 when a collection is persisted
    return not os.path.exists(os.path.join(path, "chroma.sqlite3"))

add_document = needs_build(db_location)

def s(v):
    return "" if (isinstance(v, float) and pd.isna(v)) else str(v)

if add_document:
    documents, ids = [], []
    for i, row in df.iterrows():
        content = " ".join([
            s(row.get("Name")), s(row.get("Price")), s(row.get("Best Sellers Rank")),
            s(row.get("Item Dimensions LxWxH")), s(row.get("Net Quantity")),
            s(row.get("Generic Name")), s(row.get("Number of Ratings")),
            s(row.get("Customer Rating")),
        ])
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "row": i,
                    "name": s(row.get("Name")),
                    "rating": s(row.get("Customer Rating")),
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
