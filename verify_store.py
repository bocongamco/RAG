# verify_store.py
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
