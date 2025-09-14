from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

#Load csv file

df = pd.read_csv("./training-data/Amazon_Laptop_Specs.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_db"
add_document = not os.path.exists(db_location)

def s(v):
    return "" if (isinstance(v, float) and pd.isna(v)) else str(v)

if add_document:
    documents = []
    ids = []

    for i, row in df.iterrows():
        document = Document(
            page_content =
                s(row["Name"]) + " " +
                s(row["Price"]) + " " +
                s(row["Best Sellers Rank"]) + " " +
                s(row["Item Dimensions LxWxH"]) + " " +
                s(row["Net Quantity"]) + " " +
                s(row["Generic Name"]) + " " +
                s(row["Number of Ratings"]) + " " +
                s(row["Customer Rating"]),   # CSV header is singular
            metadata = {"rating": s(row["Customer Rating"])}
        )
        ids.append(str(i))
        documents.append(document)

#Create vector store
vector_store = Chroma(
    collection_name="laptop-specs",
    embedding_function=embeddings,
    persist_directory=db_location
)

if add_document:
    vector_store.add_documents(documents, ids=ids)

#Create retriever from vector store
retriever = vector_store.as_retriever(
    search_kwargs = {"k": 3}
)