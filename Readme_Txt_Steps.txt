
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



