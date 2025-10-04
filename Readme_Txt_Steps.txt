
start ollama
ollama --version
ollama version is 0.12.3

curl http://127.0.0.1:11434/api/tags
ollama pull mxbai-embed-large
ollama pull llama3.2
ollama list

conda create -n ragenv python=3.11 -y
conda activate ragenv
pip install -r requirements.txt
pip install fastapi uvicorn


(ragenv) C:\Users\gaura\Desktop\RAG_F\RAG>python Src\search\vector.py --init
[cli] init_stores()
[vector] No Chroma DB found → building index from CSV and creating Chroma
[vector] Built docs_index.csv with 551 rows from Amazon_Laptop_Specs_utf8.csv
[vector] Chroma DB created and persisted

python -m Src.Eda.Cli --rebuild --csv Data\Training-Data\Amazon_Laptop_Specs_utf8.csv --out Outputs\eda
EDA → C:\Users\gaura\Desktop\RAG_F\RAG\Outputs\eda\eda_summary.json + charts (overwritten)

(ragenv) C:\Users\gaura\Desktop\RAG_F\RAG>python -m Src.Eda.Cli --make-qrels Data\qrels.csv --csv Data\Training-Data\Amazon_Laptop_Specs_utf8.csv
EDA → C:\Users\gaura\Desktop\RAG_F\RAG\Outputs\eda\eda_summary.json + charts
Wrote 551 rows to C:\Users\gaura\Desktop\RAG_F\RAG\Data\qrels.csv (from C:\Users\gaura\Desktop\RAG_F\RAG\data_index\docs_index.csv).


python -m Src.alpha.cli --config config\eval.yaml --val-frac 0.20 --test-frac 0.15 --out Outputs\alpha_eval --select cv --cv-folds 5 --one-std-err 0 --tie-break min_alpha

[vector] Using existing Chroma DB
[vector] Loaded 551 docs from docs_index.csv
[alpha] split: non-test=468 | test=83 | total=551
[alpha] building candidates for non-test (468 queries) ...
[alpha]   (non-test) built candidates for 20/468 queries
[alpha]   (non-test) built candidates for 40/468 queries
[alpha]   (non-test) built candidates for 60/468 queries
[alpha]   (non-test) built candidates for 80/468 queries
[alpha]   (non-test) built candidates for 100/468 queries
[alpha]   (non-test) built candidates for 120/468 queries
[alpha]   (non-test) built candidates for 140/468 queries
[alpha]   (non-test) built candidates for 160/468 queries
[alpha]   (non-test) built candidates for 180/468 queries
[alpha]   (non-test) built candidates for 200/468 queries
[alpha]   (non-test) built candidates for 220/468 queries
[alpha]   (non-test) built candidates for 240/468 queries
[alpha]   (non-test) built candidates for 260/468 queries
[alpha]   (non-test) built candidates for 280/468 queries
[alpha]   (non-test) built candidates for 300/468 queries
[alpha]   (non-test) built candidates for 320/468 queries
[alpha]   (non-test) built candidates for 340/468 queries
[alpha]   (non-test) built candidates for 360/468 queries
[alpha]   (non-test) built candidates for 380/468 queries
[alpha]   (non-test) built candidates for 400/468 queries
[alpha]   (non-test) built candidates for 420/468 queries
[alpha]   (non-test) built candidates for 440/468 queries
[alpha]   (non-test) built candidates for 460/468 queries
[alpha] selection = CV (5 folds) + one-std-err=False
[alpha] CV fold 1/5 computed.
[alpha] CV fold 2/5 computed.
[alpha] CV fold 3/5 computed.
[alpha] CV fold 4/5 computed.
[alpha] CV fold 5/5 computed.
[alpha] chosen α=0.35
[alpha] evaluating TEST (83 queries)
α=0.35 | train nDCG@k=0.978 | val nDCG@k=0.980 | test nDCG@k=0.952
[alpha] artifacts saved to: Outputs\alpha_eval


python -m uvicorn Src.api.server:app --reload --port 8000
←[32mINFO←[0m:     Will watch for changes in these directories: ['C:\\Users\\gaura\\Desktop\\RAG_F\\RAG']
←[32mINFO←[0m:     Uvicorn running on ←[1mhttp://127.0.0.1:8000←[0m (Press CTRL+C to quit)
←[32mINFO←[0m:     Started reloader process [←[36m←[1m5928←[0m] using ←[36m←[1mWatchFiles←[0m
INFO:     Started server process [8044]
INFO:     Waiting for application startup.
INFO:     Application startup complete.

http://localhost:8000/health

conda install nodejs
cd UI 
echo VITE_API_BASE=http://localhost:8000 > .env
npm install
npm run dev

http://localhost:5173/


(ragenv) C:\Users\gaura\Desktop\RAG_F\RAG>python -m Src.eval.eval
(ragenv) C:\Users\gaura\Desktop\RAG_F\RAG>python -m Src.eval.eval2
(ragenv) C:\Users\gaura\Desktop\RAG_F\RAG>python -m Src.eval.analysis







(base) C:\Users\gaura\Desktop\RAG_F\RAG>for /f "delims=" %A in ('dir /b') do @echo %A & if exist "%A\" (for /f "delims=" %B in ('dir /b "%A"') do @echo   ^|__ %B & if exist "%A\%B\" (for /f "delims=" %C in ('dir /b "%A\%B"') do @echo       ^|__ %C))
.env
.gitignore
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
      |__ server_old.py
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
      |__ vector_1.py
      |__ vector_2.py
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
  
git status
git add -A
git commit -m "run-time fix"
git push origin main