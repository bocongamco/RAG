from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

QUERY  = "price of Acer Aspire 5"
emb = OllamaEmbeddings(model="mxbai-embed-large")
print(f"query:: {QUERY}")
db = Chroma(
    collection_name="laptop-specs",      # or laptop_specs (underscore) if that’s how you built it
    embedding_function=emb,
    persist_directory="./chroma_db",     # <-- corrected path
)

ids = db.get().get("ids", [])
print(f"[info] docs in store: {len(ids)}")

pairs = db.similarity_search_with_score(QUERY, k=3)
if not pairs:
    print("[warn] 0 results. Check PERSIST/COLL or rebuild the index. Is ollama running?")
else:
    for i, (d, score) in enumerate(pairs, 1):
        print(f"\n--- Doc {i} (score={score:.4f}) ---")
        print(d.page_content[:400].replace("\n", " "))
        print(d.metadata)
