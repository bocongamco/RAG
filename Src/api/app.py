from fastapi import FastAPI, Query
from pathlib import Path
import json

app = FastAPI(title="RAG Laptops (no-IDF)")
ART = Path("artifacts"); EDA_SUM = ART/"eda/eda_summary.json"; ALPHA = ART/"eval/alpha.json"

@app.get("/alpha")
def get_alpha():
    if ALPHA.exists(): return json.loads(ALPHA.read_text(encoding="utf-8"))
    return {"alpha": 0.6, "k": 5}

@app.get("/search-mode")
def get_mode(default: str = Query("hybrid", pattern="^(bm25|dense|hybrid)$")):
    return {"mode": default}

@app.get("/eda/summary")
def eda_summary():
    return json.loads(EDA_SUM.read_text(encoding="utf-8")) if EDA_SUM.exists() else {}
