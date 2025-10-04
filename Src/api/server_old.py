# api/server.py
# FastAPI backend for Laptop RAG (dense / bm25 / hybrid)

import os, sys
from typing import List, Literal, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

#project root 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#names defined in vector.py
from vector import dense_search, bm25_retriever, hybrid_search

app = FastAPI(title="Laptop RAG API")

# CORS: adding frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

# LLM + prompt (Ollama must be running and models pulled)
_llm = OllamaLLM(model="llama3.2")
_PROMPT = ChatPromptTemplate.from_template(
    """
You are a laptop shopping assistant.
Use ONLY the given CONTEXT. Do not invent facts.

Return the answer in EXACTLY this format:
Answer: <two short sentences with the exact MODEL and PRICE from context>
Citations: row=<ROW>, name="<MODEL>"

If the answer is not in the context, reply exactly:
Answer: I don't know
Citations: (none)

CONTEXT:
{ctx}

QUESTION: {q}
"""
)
_chain = _PROMPT | _llm

def fmt_ctx(docs):
    return "\n".join(d.page_content[:350] for d in docs)

def docs_to_json(docs):
    return [
        {
            "row": d.metadata.get("row"),
            "name": d.metadata.get("name"),
            "preview": d.page_content[:500],
        }
        for d in docs
    ]

class AskReq(BaseModel):
    question: str
    mode: Optional[Literal["dense", "bm25", "hybrid"]] = "hybrid"
    k: int = 3
    alpha: Optional[float] = 0.6  # used only for hybrid

class AskResp(BaseModel):
    answer: str
    mode: str
    documents: List[dict]

@app.post("/ask", response_model=AskResp)
def ask(req: AskReq):
    q = req.question.strip()
    mode = (req.mode or "hybrid").lower()
    k = max(1, min(20, int(req.k)))
    alpha = 0.6 if req.alpha is None else float(req.alpha)
    alpha = max(0.0, min(1.0, alpha))  # clamp for safety

    #retriever choice 
    if mode == "dense":
        docs = dense_search(q, k=k) or []
    elif mode == "bm25":
        docs = bm25_retriever.search(q, k=k) or []
    else:  # hybrid
        docs = hybrid_search(q, k=k, alpha=alpha) or []

    if not docs:
        return AskResp(answer="Answer: I don't know\nCitations: (none)", mode=mode, documents=[])

    ctx = fmt_ctx(docs)
    ans = _chain.invoke({"ctx": ctx, "q": q})
    return AskResp(answer=str(ans), mode=mode, documents=docs_to_json(docs))
