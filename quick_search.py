from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

emb = OllamaEmbeddings(model="mxbai-embed-large")
db = Chroma(
    collection_name="laptop-specs",
    embedding_function=emb,
    persist_directory="./chrome_langchain_db",
)

docs = db.similarity_search("price of Acer Aspire 5", k=3)
for i, d in enumerate(docs, 1):
    print(f"\n--- Doc {i} ---")
    print(d.page_content[:400])
    print(d.metadata)