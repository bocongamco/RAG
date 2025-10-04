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
conda create -n ragenv python=3.11 -y
conda activate ragenv

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

start ollama
ollama --version    =>  ollama version is 0.12.0
curl http://127.0.0.1:11434/api/tags
ollama pull mxbai-embed-large
ollama pull llama3.2
ollama list

sample output ::
C:\Users\gaura>ollama list
NAME                        ID              SIZE      MODIFIED
llama3.2:latest             a80c4f17acd5    2.0 GB    2 days ago
mxbai-embed-large:latest    468836162de7    669 MB    2 days ago
nomic-embed-text:latest     0a109f422b47    274 MB    7 days ago
=======
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
python Src\search\vector.py --init
```
You should see a `chroma.sqlite3` inside `chrome_langchain_db/` afterward.
data_index folder having docs_index, docid, doc/P01/0d498ada3f6c.md like file, meta.json
```
python -m Src.Eda.Cli --rebuild --csv Data\Training-Data\Amazon_Laptop_Specs_utf8.csv --out Outputs\eda
python -m Src.Eda.Cli --make-qrels Data\qrels.csv --csv Data\Training-Data\Amazon_Laptop_Specs_utf8.csv

python -m Src.alpha.cli --config config\eval.yaml --val-frac 0.20 --test-frac 0.15 --out Outputs\alpha_eval --select cv --cv-folds 5 --one-std-err 0 --tie-break min_alpha```

---

## Run a UI

### FastAPI + React

**Backend (FastAPI)**
```powershell
python -m uvicorn Src.api.server:app --reload --port 8000
```
Health check: http://localhost:8000/health

**Frontend (Vite React in `rag-ui/`)**
```powershell
cd UI
# create .env with API base URL
echo VITE_API_BASE=http://localhost:8000 > .env
conda install nodejs  (one time only)
npm install
npm run dev
```
Open the printed URL (usually `http://localhost:5173`).

> CORS: `api/server.py` already enables CORS for `http://localhost:5173`. If you use a different port, add it to `allow_origins`.

---

## Project Structure

```
RAG/
chroma_db
  |__ a6767a01-fc9f-4380-85c2-f617ffa70606
      |__ data_level0.bin
      |__ header.bin
      |__ length.bin
      |__ link_lists.bin
  |__ chroma.sqlite3
Config
  |__ eval.yaml
Data
  |__ qrels.csv
  |__ Training-Data
      |__ Amazon_Laptop_Specs.csv
      |__ Amazon_Laptop_Specs_utf8.csv
      |__ amazon_products_wide.csv
data_index
  |__ docid
  |__ docs
      |__ P01
      |__ P02
      |__ P03
      |__ P04
      |__ P05
      |__ P06
  |__ docs_index.csv
  |__ meta.json
Outputs
  |__ alpha_eval
      |__ cv_curve.csv
      |__ splits.json
      |__ summary.json
  |__ eda
      |__ corr_heatmap.png
      |__ dist_pricenum.png
      |__ eda_summary.json
RAG_OLD.zip
README.md
Readme_Txt_Steps.txt
Report-Docs
  |__ ANSWER-2-GUI
  |__ FastAPI + React UI Running Status.png
  |__ GUI-2-RAG-POST_QUERY
  |__ Indexing-LOAD-SPLIT-EMBED-STORE
  |__ Ollama_checking_IN_GUI.png
  |__ RAG Query Setup
  |__ REACT GUI-BROWSER for query.png
  |__ Retrieval and Answer Generation
requirements.txt
Src
  |__ alpha
      |__ cli.py
      |__ data.py
      |__ metrics.py
      |__ train.py
      |__ __pycache__
  |__ api
      |__ app.py
      |__ server.py
      |__ __pycache__
  |__ cli
      |__ main.py
      |__ Main.py-Steps.txt
      |__ quick_search_All.py
      |__ quick_search_All.py-Steps.txt
      |__ quick_search_dense.py
      |__ Verify-store.py-steps.txt
      |__ verify_store.py
  |__ Eda
      |__ Cli.py
      |__ qrels.py
      |__ summary.py
      |__ __pycache__
  |__ eval
      |__ analysis.py
      |__ eval performance.txt
      |__ eval.py
      |__ eval2.py
      |__ evaluation_results.json
      |__ evaluation_results.png
      |__ ground_truth.json
      |__ RESULTS.md
  |__ search
      |__ vector.py
      |__ __pycache__
tree.txt
UI
  |__ .env
  |__ .gitignore
  |__ eslint.config.js
  |__ index.html
  |__ node_modules
      |__ .bin
      |__ .package-lock.json
      |__ .vite
      |__ .vite-temp
      |__ @babel
      |__ @esbuild
      |__ @eslint
      |__ @eslint-community
      |__ @humanfs
      |__ @humanwhocodes
      |__ @jridgewell
      |__ @nodelib
      |__ @rolldown
      |__ @rollup
      |__ @types
      |__ @typescript-eslint
      |__ @vitejs
      |__ acorn
      |__ acorn-jsx
      |__ ajv
      |__ ansi-styles
      |__ argparse
      |__ balanced-match
      |__ baseline-browser-mapping
      |__ brace-expansion
      |__ braces
      |__ browserslist
      |__ callsites
      |__ caniuse-lite
      |__ chalk
      |__ color-convert
      |__ color-name
      |__ concat-map
      |__ convert-source-map
      |__ cross-spawn
      |__ csstype
      |__ debug
      |__ deep-is
      |__ electron-to-chromium
      |__ esbuild
      |__ escalade
      |__ escape-string-regexp
      |__ eslint
      |__ eslint-plugin-react-hooks
      |__ eslint-plugin-react-refresh
      |__ eslint-scope
      |__ eslint-visitor-keys
      |__ espree
      |__ esquery
      |__ esrecurse
      |__ estraverse
      |__ esutils
      |__ fast-deep-equal
      |__ fast-glob
      |__ fast-json-stable-stringify
      |__ fast-levenshtein
      |__ fastq
      |__ file-entry-cache
      |__ fill-range
      |__ find-up
      |__ flat-cache
      |__ flatted
      |__ gensync
      |__ glob-parent
      |__ globals
      |__ graphemer
      |__ has-flag
      |__ ignore
      |__ import-fresh
      |__ imurmurhash
      |__ is-extglob
      |__ is-glob
      |__ is-number
      |__ isexe
      |__ js-tokens
      |__ js-yaml
      |__ jsesc
      |__ json-buffer
      |__ json-schema-traverse
      |__ json-stable-stringify-without-jsonify
      |__ json5
      |__ keyv
      |__ levn
      |__ locate-path
      |__ lodash.merge
      |__ lru-cache
      |__ merge2
      |__ micromatch
      |__ minimatch
      |__ ms
      |__ nanoid
      |__ natural-compare
      |__ node-releases
      |__ optionator
      |__ p-limit
      |__ p-locate
      |__ parent-module
      |__ path-exists
      |__ path-key
      |__ picocolors
      |__ picomatch
      |__ postcss
      |__ prelude-ls
      |__ punycode
      |__ queue-microtask
      |__ react
      |__ react-dom
      |__ react-refresh
      |__ resolve-from
      |__ reusify
      |__ rollup
      |__ run-parallel
      |__ scheduler
      |__ semver
      |__ shebang-command
      |__ shebang-regex
      |__ source-map-js
      |__ strip-json-comments
      |__ supports-color
      |__ tinyglobby
      |__ to-regex-range
      |__ ts-api-utils
      |__ type-check
      |__ typescript
      |__ typescript-eslint
      |__ update-browserslist-db
      |__ uri-js
      |__ vite
      |__ which
      |__ word-wrap
      |__ yallist
      |__ yocto-queue
  |__ package-lock.json
  |__ package.json
  |__ public
      |__ vite.svg
  |__ README.md
  |__ src
      |__ App.css
      |__ App.tsx
      |__ assets
      |__ index.css
      |__ main.tsx
      |__ vite-env.d.ts
  |__ tsconfig.app.json
  |__ tsconfig.json
  |__ tsconfig.node.json
  |__ vite.config.ts
  
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