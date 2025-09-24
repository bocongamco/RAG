# api/server.py
import os, sys
from typing import List, Literal, Optional
from fastapi.middleware.cors import CORSMiddleware

# allow imports from project root (so vector.py is found)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# import multiple retrievers from vector.py (dense, bm25, hybrid)
from vector import retriever_dense, retriever_bm25, hybrid_search

# -------------------
# FastAPI setup
# -------------------
app = FastAPI(title="Laptop RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # adjust if UI runs on another port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    """Simple health check endpoint"""
    return {"ok": True}

# -------------------
# LLM + Prompt setup
# -------------------
_llm = OllamaLLM(model="llama3.2")
_PROMPT = ChatPromptTemplate.from_template("""
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
""")
_chain = _PROMPT | _llm

def _fmt_ctx(docs):
    """Format retrieved docs into context string"""
    return "\n".join(d.page_content[:350] for d in docs)

def _docs_to_json(docs):
    """Serialize docs for JSON response"""
    return [
        {
            "row": d.metadata.get("row"),
            "name": d.metadata.get("name"),
            "preview": d.page_content[:500]
        }
        for d in docs
    ]

# -------------------
# API models
# -------------------
class AskReq(BaseModel):
    question: str
    mode: Optional[Literal["dense", "bm25", "hybrid"]] = "hybrid"
    k: int = 3
    alpha: float = 0.6  # hybrid weight (0 = pure bm25, 1 = pure dense)

class AskResp(BaseModel):
    answer: str
    mode: str
    documents: List[dict]

# -------------------
# API endpoint
# -------------------
@app.post("/ask", response_model=AskResp)
def ask(req: AskReq):
    """Answer a laptop question using chosen retriever mode"""
    q, mode, k, alpha = req.question.strip(), req.mode.lower(), req.k, req.alpha

    # pick retriever
    if mode == "dense":
        docs = retriever_dense.invoke(q)
    elif mode == "bm25":
        docs = retriever_bm25.search(q, k=k)
    else:  # default = hybrid
        docs = hybrid_search(q, k=k, alpha=alpha)

    if not docs:
        return AskResp(
            answer="Answer: I don't know\nCitations: (none)",
            mode=mode,
            documents=[]
        )

    ctx = _fmt_ctx(docs)
    ans = _chain.invoke({"ctx": ctx, "q": q})

    return AskResp(
        answer=str(ans),
        mode=mode,
        documents=_docs_to_json(docs)
    )
