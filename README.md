# WIL Project 32
# Laptop Shopping RAG (Local)

A tiny Retrieval-Augmented Generation (RAG) prototype that answers questions about **laptops** from a CSV.
- **LLM/Embeddings:** [Ollama](https://ollama.com/) (`llama3.2`, `mxbai-embed-large`)
- **Vector store:** Chroma (persisted locally)
- **Orchestration:** LangChain
- **UIs:** (A) Streamlit quick demo, or (B) FastAPI + React (Vite) chat app

> Works **fully offline** once models are pulled with Ollama.

---

## Quick Setup

### 0) Requirements
- Python **3.10–3.12** (repo tested on 3.11)
- Node **18+** (only for the React UI option)
- [Ollama](https://ollama.com/download) installed and running
- Git (optional)

### 1) Clone & Python deps
```powershell
git clone https://github.com/bocongamco/RAG.git
cd RAG

# (recommended) create venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # macOS/Linux: source .venv/bin/activate

# install Python deps
pip install -r requirements.txt
pip install fastapi uvicorn
```

### 2) Ollama models
Make sure the Ollama service is running (`ollama serve`) or the Windows service is started, then pull the models:
```powershell
ollama pull mxbai-embed-large
ollama pull llama3.2
ollama list
```
If needed:
```powershell
$env:OLLAMA_HOST = "http://localhost:11434"
```

### 3) Build the vector store
This embeds the CSV and persists a local Chroma DB to `./chrome_langchain_db/`.
```powershell
python vector.py
```
You should see a `chroma.sqlite3` inside `chrome_langchain_db/` afterward.

---

## Run a UI

### FastAPI + React

**Backend (FastAPI)**
```powershell
python -m uvicorn api.server:app --reload --port 8000
```
Health check: http://localhost:8000/health

**Frontend (Vite React in `rag-ui/`)**
```powershell
cd rag-ui
# create .env with API base URL
echo VITE_API_BASE=http://localhost:8000 > .env

npm install
npm run dev
```
Open the printed URL (usually `http://localhost:5173`).

> CORS: `api/server.py` already enables CORS for `http://localhost:5173`. If you use a different port, add it to `allow_origins`.

---

## Project Structure

```
RAG/
├── api/
│   └── server.py            # FastAPI backend exposing POST /ask
├── chrome_langchain_db/     # Chroma persisted DB (created by vector.py)
├── rag-ui/                  # React (Vite) frontend
├── training-data/
│   └── Amazon_Laptop_Specs.csv
├── main.py                  # CLI loop (optional)
├── vector.py                # builds/opens retriever (Chroma + Ollama embeddings)
├── quick_search.py          # simple retrieval sanity check
├── verify_store.py          # counts docs in Chroma
├── requirements.txt
└── README.md
```

---

## How it works (short)

1. `vector.py` reads `training-data/Amazon_Laptop_Specs.csv`, converts each row into a **Document**, embeds with **`mxbai-embed-large`**, and persists to Chroma.
2. The retriever returns the top-k docs for a query.
3. `llama3.2` generates the answer from the retrieved context and includes light **citations** (row + model).

---

## Troubleshooting

**`ollama: not recognized`**
- Reopen your terminal, or add Ollama to PATH. Test with `ollama --version`.  
- On Windows you can run it directly: `& "C:\Users\<YOU>\AppData\Local\Programs\Ollama\ollama.exe" --version`.

**`model "... not found"`**
- Pull it: `ollama pull mxbai-embed-large` (and `ollama pull llama3.2`).

**Vector store empty / no results**
- Delete `chrome_langchain_db/` and run `python vector.py` again.
- Ensure `training-data/Amazon_Laptop_Specs.csv` exists and has data.

**Streamlit says it can’t import `vector`**
- Run `streamlit run app/app.py` from the **repo root** (same folder as `vector.py`).
- We add the parent folder to `sys.path` in the app to help, but running from the root is safest.

**React UI CORS error**
- In `api/server.py`, update the `allow_origins` list to match your frontend origin (port).

**Can’t reach API from React**
- Verify FastAPI is running on `http://localhost:8000/health`.
- Ensure `.env` in `rag-ui/` contains `VITE_API_BASE=http://localhost:8000` and restart `npm run dev`.

---

## Optional: Offline Evaluation (for the assignment)

If you add a small gold set `eval/qa_gold.csv`, you can run a quick script to compute:
- **Answered %** (abstentions)
- **Exact match %** (vs known answers)
- **Attribution %** (cited row/model appears in retrieved context)
- **Latency**

Skeleton (create `eval/run_eval.py`) is easy to add later.

---

## License