# WIL-Project-W32
WIL-Project-W32

# RAG (Laptops) — Mac Setup

This is a **local Retrieval-Augmented Generation (RAG)** prototype that indexes a CSV of laptop specs and lets you query it using **Ollama** for embeddings and generation.

> Stack: Python 3.11 + virtualenv, LangChain (community split), Chroma as vector store, Ollama (`mxbai-embed-large` for embeddings).

---

## Folder Structure

```
RAG/
├─ docs-projec-description/    # docs
├─ training-data/
│  └─ Amazon_Laptop_Specs.csv  # your dataset
├─ .gitignore
├─ main.py                     # simple retriever/CLI
├─ requirements.txt            # Python deps
└─ vector.py                   # builds the vector store from CSV
```

---

## Prerequisites (Mac)

1. **Install Python 3.11+**  
   - `python3 --version` should show 3.11+

2. **Install & start Ollama**  
   - Download: https://ollama.com  
   - After installing, start the app (it runs a local server at `http://localhost:11434`).  
   - Pull the embedding model:
     ```bash
     ollama pull mxbai-embed-large
     ```
   - (Optional) pull a chat model if you plan to generate answers:
     ```bash
     ollama pull llama3.1
     ```

---

## One‑time Project Setup

From the repo root (the folder that contains `requirements.txt`):

```bash
# 0) Optional: ensure Auto Save enabled in VS Code
# File → Auto Save

# 1) Create & activate a virtualenv
python3 -m venv venv
source venv/bin/activate

# 2) Upgrade pip and install deps
python -m pip install -U pip
python -m pip install -r requirements.txt

# 3) (If not already running) make sure Ollama is up
curl -s http://localhost:11434/api/tags | head
# if empty, launch the Ollama app or run: ollama serve
```

> **Tip:** If you see “No module named …”, double‑check you’re in the venv: `which python` should point to `.../RAG/venv/bin/python`.

---

## Index the CSV (build the vector store)

`vector.py` reads `training-data/Amazon_Laptop_Specs.csv`, converts each row into a `Document`, and persists the embeddings into `chrome_langchain_db/`.

```bash
# from repo root, with venv active and Ollama running
python vector.py
# Expected: "Indexed N rows into chrome_langchain_db"
```

If you change the CSV, re-run `python vector.py` to rebuild the index.

---

## Query the Index (simple demo)

`main.py` loads the Chroma DB and does similarity search. Replace with your own logic as needed.

```bash
python main.py
# You> thin and light laptop under 1.7 kg
# (prints top hits with row names / indices)
```

---

## Configuration Notes

- **CSV path**: `training-data/Amazon_Laptop_Specs.csv` (relative to repo root).  
- **Chroma DB path**: `chrome_langchain_db/` (auto‑created).  
- **Embedding model**: `mxbai-embed-large` (via Ollama). Modify in code if you prefer a different model.  

---

## Git Hygiene (recommended)

`.gitignore` should include the following to keep the repo clean:

```
venv/
__pycache__/
*.pyc
.DS_Store
chrome_langchain_db/
```

If you accidentally committed the venv or DB in the past, untrack them:

```bash
git rm -r --cached venv chrome_langchain_db
git add -A
git commit -m "Ignore venv and Chroma DB"
git push
```

---

## Troubleshooting

- **`FileNotFoundError` for CSV**  
  Ensure the file exists at `training-data/Amazon_Laptop_Specs.csv`. Use:
  ```bash
  ls -l training-data/Amazon_Laptop_Specs.csv
  ```

- **`TypeError: can only concatenate str (not "float") to str`**  
  Some CSV columns are numeric/NaN. The provided `vector.py` uses a small `s()` helper to safely stringify values.

- **Ollama not reachable**  
  - Start the app or run `ollama serve`.
  - Verify: `curl -s http://localhost:11434/api/tags | head`
  - Pull the embed model: `ollama pull mxbai-embed-large`

- **Wrong Python / pip**  
  - Re‑activate venv: `source venv/bin/activate`  
  - Confirm: `which python`, `python -V`, `python -m pip --version`

---

## Commands Cheat‑Sheet

```bash
# create venv & install deps
python3 -m venv venv
source venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt

# pull models
ollama pull mxbai-embed-large
ollama pull llama3.1

# build index
python vector.py

# query
python main.py

# deactivate venv
deactivate
```

---

## What’s next?

- Add richer chunking from multiple sources (specs/reviews/policies).  
- Swap in a persistent vector DB server (Qdrant/Weaviate/Chroma server) if desired.  
- Build a small web API (FastAPI/ASP.NET Core) or a GUI (Streamlit/React) on top.

---

**Author notes:** macOS‑focused setup. Adjust shell paths for Linux/Windows (e.g., `venv\Scripts\activate` on Windows PowerShell).

