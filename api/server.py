# api/server.py
import os, sys
from typing import List
from fastapi.middleware.cors import CORSMiddleware

# make sure we can import vector.py from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

app = FastAPI(title="Laptop RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# allow simple health check
@app.get("/health")
def health():
    return {"ok": True}

# LLM chain (same as Streamlit prompt)
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
    return "\n".join(d.page_content[:350] for d in docs)

def _docs_to_json(docs):
    out = []
    for d in docs:
        out.append({
            "row": d.metadata.get("row"),
            "name": d.metadata.get("name"),
            "preview": d.page_content[:500]
        })
    return out

class AskReq(BaseModel):
    question: str

class AskResp(BaseModel):
    answer: str
    documents: List[dict]

@app.post("/ask", response_model=AskResp)
def ask(req: AskReq):
    docs = retriever.invoke(req.question)
    if not docs:
        return AskResp(answer="Answer: I don't know\nCitations: (none)", documents=[])
    ctx = _fmt_ctx(docs)
    ans = _chain.invoke({"ctx": ctx, "q": req.question})
    return AskResp(answer=str(ans), documents=_docs_to_json(docs))
