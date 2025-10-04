# Src/api/server.py
# FastAPI backend for Laptop RAG (dense / bm25 / hybrid)

from typing import List, Literal, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import re

# IMPORTANT: use the exact casing of your package name
from Src.search.vector import (
    dense_search,
    hybrid_search,
    bm25_get,              # <-- SAFE wrapper (calls init_stores)
    _load_learned_alpha,
)

app = FastAPI(title="Laptop RAG API")

# CORS (adjust to your UI origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/alpha")
def get_alpha():
    return {"alpha": _load_learned_alpha(0.6)}

# ---------------- LLM prompt ----------------
_llm = OllamaLLM(model="llama3.2")
_PROMPT = ChatPromptTemplate.from_template(
    """
You are a laptop shopping assistant.
Use ONLY the given CONTEXT. Do not invent facts.

CRITICAL RULES:
- When you mention the price, COPY the substring that appears AFTER 'PRICE=' in the context, up to the next '|' (if any).
- DO NOT add or change currency symbols. DO NOT format, round, or shorten numbers.

Return the answer in EXACTLY this format:
Answer: <two short sentences with the exact MODEL and the exact PRICE substring copied verbatim>
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

# ---------------- Deterministic helpers ----------------
PRICE_RE = re.compile(r"PRICE\s*=\s*([^|]+)")           # text after PRICE= up to next '|'
ROW_IN_ANSWER_RE = re.compile(r"row\s*=\s*(\d+)", re.IGNORECASE)

def extract_price_text(doc_text: str) -> Optional[str]:
    m = PRICE_RE.search(doc_text)
    return m.group(1).strip() if m else None

def parse_price_value(price_text: Optional[str]) -> Optional[float]:
    if not price_text:
        return None
    s = price_text.strip()
    s = s.replace(",", "").replace(" ", "")
    for sym in ("₹", "$", "€", "Rs.", "rs.", "INR", "USD", "EUR"):
        s = s.replace(sym, "")
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    try:
        return float(m.group(1)) if m else None
    except Exception:
        return None

def find_doc_by_row(docs, row_str: str):
    for d in docs:
        if str(d.metadata.get("row")) == str(row_str):
            return d
    return None

def is_max_price_query(q: str) -> bool:
    q = q.lower()
    return any(k in q for k in ["highest price", "most expensive", "max price", "costliest"])

def is_min_price_query(q: str) -> bool:
    q = q.lower()
    return any(k in q for k in ["lowest price", "cheapest", "min price"])

def is_best_query(q: str) -> bool:
    q = q.lower()
    return any(k in q for k in ["best laptop", "best one", "top laptop", "recommend", "which is best", "top one"])

# ---------------- API schema ----------------
class AskReq(BaseModel):
    question: str
    mode: Optional[Literal["dense", "bm25", "hybrid"]] = "hybrid"
    k: int = 3
    alpha: Optional[float] = None  # None → vector.hybrid_search loads learned α

class AskResp(BaseModel):
    answer: str
    mode: str
    documents: List[dict]

# ---------------- Main endpoint ----------------
@app.post("/ask", response_model=AskResp)
def ask(req: AskReq):
    q = req.question.strip()
    mode = (req.mode or "hybrid").lower()
    k = max(1, min(20, int(req.k)))

    # retrieve (use safe wrappers)
    if mode == "dense":
        docs = dense_search(q, k=k) or []
    elif mode == "bm25":
        docs = bm25_get(q, k=k) or []            # <-- FIXED: no crash when stores uninitialized
    else:
        docs = hybrid_search(q, k=k, alpha=req.alpha) or []

    if not docs:
        return AskResp(answer="Answer: I don't know\nCitations: (none)", mode=mode, documents=[])

    # ====== Deterministic handling for vague "best" queries ======
    if is_best_query(q):
        top = docs[0]
        name = top.metadata.get("name")
        row  = top.metadata.get("row")
        ptxt = extract_price_text(top.page_content)
        if ptxt:
            answer = f"Answer: Based on the retrieved results, the top match is {name}, priced at {ptxt}."
        else:
            answer = f"Answer: Based on the retrieved results, the top match is {name}."
        cites  = f'Citations: row={row}, name="{name}"'
        return AskResp(answer=f"{answer}\n{cites}", mode=mode, documents=docs_to_json(docs))
    # =============================================================

    # ====== Deterministic handling for price-extrema questions ======
    if is_max_price_query(q) or is_min_price_query(q):
        best_doc = None
        best_price_val = None
        best_price_txt = None

        for d in docs:
            ptxt = extract_price_text(d.page_content)
            pval = parse_price_value(ptxt)
            if pval is None:
                continue
            if is_max_price_query(q):
                take = (best_price_val is None) or (pval > best_price_val)
            else:
                take = (best_price_val is None) or (pval < best_price_val)
            if take:
                best_doc, best_price_val, best_price_txt = d, pval, ptxt

        if best_doc and best_price_txt:
            row = best_doc.metadata.get("row")
            name = best_doc.metadata.get("name")
            phr = "highest" if is_max_price_query(q) else "lowest"
            answer = f"Answer: {name} is the laptop with the {phr} price, priced at {best_price_txt}."
            cites  = f'Citations: row={row}, name="{name}"'
            return AskResp(answer=f"{answer}\n{cites}", mode=mode, documents=docs_to_json(docs))
        # fall through to LLM if we couldn't parse prices
    # ================================================================

    # LLM answer (with safety fix-up to copy exact PRICE from cited row)
    ctx = fmt_ctx(docs)
    try:
        ans = _chain.invoke({"ctx": ctx, "q": q})
    except Exception:
        # If Ollama/LLM is unavailable, return a graceful non-LLM fallback using top doc
        top = docs[0]
        name = top.metadata.get("name")
        row  = top.metadata.get("row")
        ptxt = extract_price_text(top.page_content)
        if ptxt:
            ans = f"Answer: {name}. Example price: {ptxt}\nCitations: row={row}, name=\"{name}\""
        else:
            ans = f"Answer: {name}\nCitations: row={row}, name=\"{name}\""
        return AskResp(answer=str(ans), mode=mode, documents=docs_to_json(docs))

    # If the model cited a row, ensure the price string is exact
    mrow = ROW_IN_ANSWER_RE.search(str(ans))
    if mrow:
        d = find_doc_by_row(docs, mrow.group(1))
        if d:
            ptxt = extract_price_text(d.page_content)
            if ptxt:
                ans = re.sub(r"priced at [^\.\n]+", f"priced at {ptxt}", str(ans))

    return AskResp(answer=str(ans), mode=mode, documents=docs_to_json(docs))
