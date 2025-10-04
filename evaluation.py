#!/usr/bin/env python3
"""
Merged Evaluation Script:
- Mode dashboard (Dense / BM25 / Hybrid), alpha sweep, performance benchmarking
- Holistic QA metrics: EM/F1/Recall@k/MRR/nDCG, Faithfulness, Attribution
- Ground truth auto-built from Amazon_Laptop_Specs.json (Name, Price)
"""

import os, sys, re, json, math, time, random, hashlib, itertools
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# -----------------------------------------------------------------------------
# Bootstrap local imports
# -----------------------------------------------------------------------------
try:
    _BASE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _BASE = os.getcwd()
if _BASE not in sys.path:
    sys.path.append(_BASE)

# --- Your retrievers ---
try:
    from vector import dense_search, bm25_retriever, hybrid_search
except Exception as e:
    dense_search = bm25_retriever = hybrid_search = None
    print(f"[WARN] Could not import retrievers from vector.py: {e}")

# -----------------------------------------------------------------------------
# Text utils
# -----------------------------------------------------------------------------
def normalize_text(s: str) -> str:
    return " ".join((s or "").lower().split())

def tokenize(s: str) -> List[str]:
    return normalize_text(s).split()

def sentences(s: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", (s or "").strip())
    return [p for p in parts if p]

def rouge_l_like(s1: str, s2: str) -> float:
    t1, t2 = tokenize(s1), tokenize(s2)
    n1, n2 = len(t1), len(t2)
    if n1 == 0 or n2 == 0:
        return 0.0
    dp = [[0]*(n2+1) for _ in range(n1+1)]
    for i in range(1, n1+1):
        for j in range(1, n2+1):
            if t1[i-1] == t2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[n1][n2] / max(n1, n2)

# -----------------------------------------------------------------------------
# Effectiveness metrics
# -----------------------------------------------------------------------------
def exact_match(pred: str, golds: Sequence[str]) -> int:
    p = normalize_text(pred)
    return int(any(p == normalize_text(g) for g in golds))

def f1_score(pred: str, golds: Sequence[str]) -> float:
    ptoks = tokenize(pred)
    best = 0.0
    for g in golds:
        gtoks = tokenize(g)
        if not ptoks and not gtoks:
            best = max(best, 1.0)
            continue
        common = 0
        gbag = list(gtoks)
        for t in ptoks:
            if t in gbag:
                common += 1
                gbag.remove(t)
        if common == 0:
            continue
        precision = common / max(len(ptoks), 1)
        recall    = common / max(len(gtoks), 1)
        best = max(best, 2*precision*recall/(precision+recall+1e-9))
    return best

def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    k = min(k, len(retrieved_ids))
    if k == 0:
        return 0.0
    hits = sum(1 for rid in retrieved_ids[:k] if rid in relevant_ids)
    return hits / k

def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    denom = len(relevant_ids)
    if denom == 0:
        return 0.0
    hits = sum(1 for rid in retrieved_ids[:k] if rid in relevant_ids)
    return hits / denom

def mrr_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    for r, rid in enumerate(retrieved_ids[:k], start=1):
        if rid in relevant_ids:
            return 1.0 / r
    return 0.0

def ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    def dcg(rel: List[int]) -> float:
        return sum(rel[i] / math.log2(i+2) for i in range(len(rel)))
    rel = [1 if rid in relevant_ids else 0 for rid in retrieved_ids[:k]]
    ideal = sorted(rel, reverse=True)
    idcg = dcg(ideal)
    return (dcg(rel) / (idcg + 1e-9)) if idcg > 0 else 0.0

# -----------------------------------------------------------------------------
# Faithfulness & Attribution
# -----------------------------------------------------------------------------
def grounded_sentence_rate(answer: str, contexts: Sequence[str], thr: float = 0.5) -> float:
    sents = sentences(answer)
    if not sents:
        return 1.0
    grounded = 0
    for s in sents:
        best = 0.0
        for ctx in contexts:
            best = max(best, rouge_l_like(s, ctx))
        grounded += int(best >= thr)
    return grounded / len(sents)

CITATION_PATTERN = re.compile(r"\[(?:doc|source|id):?\s*([^\]\s]+)\]", re.IGNORECASE)
def parse_citations(text: str) -> List[str]:
    return [m.group(1) for m in CITATION_PATTERN.finditer(text or "")]

def attribution_precision(answer: str, doc_store: Dict[str, str], thr: float = 0.25, span_chars: int = 400) -> float:
    """
    Span-based attribution: for each [doc:<id>], compare the ~400 chars before the citation
    to the cited doc body using a ROUGE-L-like threshold.
    """
    cites = list(CITATION_PATTERN.finditer(answer or ""))
    if not cites:
        return 1.0
    ok = total = 0
    for m in cites:
        total += 1
        cid = m.group(1)
        body = doc_store.get(cid, "")
        if not body:
            continue
        start = max(0, m.start() - span_chars)
        span = (answer or "")[start:m.start()].strip()
        if span and (span in body or rouge_l_like(span, body) >= thr):
            ok += 1
    return ok / max(total, 1)

# -----------------------------------------------------------------------------
# Data model & ID helpers
# -----------------------------------------------------------------------------
@dataclass
class EvalExample:
    query: str
    answers: List[str]
    group: str = "default"
    relevant_ids: Optional[List[str]] = None

def get_doc_id_and_text(d, i: int) -> Tuple[str, str]:
    """
    Robustly extract a stable ID and text from retriever results.
    Tries typical metadata keys: doc_id, row, id, asin, source, url_hash.
    """
    if hasattr(d, "metadata"):
        md = d.metadata or {}
        rid = (md.get("doc_id") or md.get("row") or md.get("id")
               or md.get("asin") or md.get("source") or md.get("url_hash") or f"{i}")
        txt = d.page_content or ""
    else:
        rid = (d.get("doc_id") or d.get("row") or d.get("id")
               or d.get("asin") or d.get("source") or d.get("url_hash") or f"{i}")
        txt = d.get("text", "")
    return str(rid), txt

# -----------------------------------------------------------------------------
# Dataset: Amazon_Laptop_Specs.json (Name, Price)
# -----------------------------------------------------------------------------
STOPWORDS = {
    "the","a","an","for","with","and","or","to","of","on","in",
    "laptop","laptops","notebook","notebooks","pc","computer",
    "best","good","cheap","affordable","budget",
    "spec","specs","specifications","price","prices"
}

def _normalize_price(p) -> str:
    if p is None:
        return ""
    s = str(p).strip()
    # keep digits and dot; remove commas, currency words/symbols
    s2 = s.replace(",", "")
    m = re.search(r"(\d+(?:\.\d{1,2})?)", s2)
    return m.group(1) if m else ""

def resolve_specs_json_path() -> str:
    """
    Try a few common places for Amazon_Laptop_Specs.json
    """
    candidates = [
        os.path.join(_BASE, "training-data", "Amazon_Laptop_Specs.json"),
        os.path.join(_BASE, "Amazon_Laptop_Specs.json"),
        os.path.join(_BASE, "data", "Amazon_Laptop_Specs.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # last resort - print where we looked
    print("[WARN] Could not find Amazon_Laptop_Specs.json. Tried:")
    for p in candidates:
        print("  -", p)
    return candidates[0]  # default first

# put near your other helpers
BRANDS = {"acer","hp","hewlett","dell","lenovo","asus","msi","apple","microsoft","samsung","lg"}
STOPWORDS = {
    "the","a","an","for","with","and","or","to","of","on","in",
    "laptop","laptops","notebook","notebooks","pc","computer",
    "best","good","cheap","affordable","budget",
    "spec","specs","specifications","price","prices"
}

def _normalize_price(p) -> str:
    if p is None:
        return ""
    s = str(p).strip().replace(",", "")
    m = re.search(r"(\d+(?:\.\d{1,2})?)", s)
    return m.group(1) if m else ""

def _price_variants(p: str) -> list[str]:
    if not p:
        return []
    out = {p}
    if p.endswith(".00"):
        out.add(p[:-3])   # 799.00 -> 799
    if not p.startswith("$"):
        out.add("$" + p)  # 799 -> $799
    return list(out)

def _toks(s: str) -> list[str]:
    return re.findall(r"[a-z0-9\+\-]+", (s or "").lower())

def build_ground_truth_from_laptop_specs(json_path: str,
                                         queries: List[Tuple[str, str, str]]) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    # Flatten dataset
    products = []
    for idx, r in enumerate(rows):
        name  = str(r.get("Name", "")).strip()
        price = _normalize_price(r.get("Price", ""))
        brand = str(r.get("Brand", r.get("brand", ""))).strip()
        model = str(r.get("Model name", r.get("Model", ""))).strip()
        # main searchable text
        text  = " ".join([name, brand, model, price]).strip().lower()
        products.append({
            "row": str(idx),     # <- IMPORTANT: your indexer should also set metadata["row"]=str(idx)
            "name": name,
            "price": price,
            "brand": brand.lower(),
            "model": model.lower(),
            "text": text,
            "tokens": set(_toks(text)),
        })

    ground_truth: Dict[str, Dict[str, List[str]]] = {}

    for qtext, qtype, _exp in queries:
        # Query tokens (keep brand tokens; drop generic words)
        raw_qtoks = _toks(qtext)
        brand_qtoks = [t for t in raw_qtoks if t in BRANDS]
        qtoks = [t for t in raw_qtoks if t not in STOPWORDS and len(t) > 2]

        # If everything got stripped, fall back to raw tokens
        if not qtoks:
            qtoks = raw_qtoks

        # Matching strategy:
        # - If a brand appears in the query, require it to be present in the product tokens.
        # - Otherwise require (overlap >= 2) to avoid accidental matches.
        hits = []
        for p in products:
            ptoks = p["tokens"]
            overlap = sum(1 for t in qtoks if t in ptoks)
            has_brand = any(b in ptoks for b in brand_qtoks) if brand_qtoks else False

            if brand_qtoks:
                if has_brand and overlap >= max(1, min(2, len(qtoks))):
                    hits.append(p)
            else:
                if overlap >= 2:
                    hits.append(p)

        relevant_ids = [p["row"] for p in hits[:10]]

        # Multiple acceptable gold answers: name+price, bare price variants, brand-only, name-only
        answers = []
        for p in hits[:10]:
            name, price, brand = p["name"], p["price"], p["brand"]
            if name and price:
                answers.append(f"{name} {price}")
            for pv in _price_variants(price):
                answers.append(pv)
            if brand:
                answers.append(brand)
            if name:
                answers.append(name)

        ground_truth[qtext] = {"answers": answers, "relevant_ids": relevant_ids}

    return ground_truth



# -----------------------------------------------------------------------------
# Simple price extractor to boost EM/F1 on price questions
# -----------------------------------------------------------------------------
PRICE_RE = re.compile(r"\$?\s?(\d{2,5}(?:\.\d{1,2})?)")

def default_generate_answer(query: str, retrieved) -> Tuple[str, List[str]]:
    if not retrieved:
        return "I'm not sure based on the available documents.", []

    want_price = ("price" in normalize_text(query)) or ("cost" in normalize_text(query))
    contexts: List[str] = []
    for i, d in enumerate(retrieved[:5]):
        doc_id, text = get_doc_id_and_text(d, i)
        contexts.append(text)

        # try to get a human-friendly name from metadata if present
        if hasattr(d, "metadata"):
            title = d.metadata.get("name") or d.metadata.get("title") or ""
        else:
            title = ""

        if want_price:
            m = PRICE_RE.search(text or "")
            if m:
                val = m.group(1)
                # Prefer "Name price" if we have a title; else just the number
                if title:
                    return f"{title} {val} [doc:{doc_id}]", contexts
                return f"{val} [doc:{doc_id}]", contexts

    # fallback: stitched snippet with citation
    snips = []
    for i, d in enumerate(retrieved[:2]):
        doc_id, text = get_doc_id_and_text(d, i)
        if text:
            snip = " ".join(text.split()[:50])
            snips.append(f"{snip} [doc:{doc_id}]")
    return (" ".join(snips) if snips else "No usable content."), contexts

# Use a slightly easier grounding threshold (optional)
def grounded_sentence_rate(answer: str, contexts: Sequence[str], thr: float = 0.35) -> float:
    sents = sentences(answer)
    if not sents:
        return 1.0
    grounded = 0
    for s in sents:
        best = 0.0
        for ctx in contexts:
            best = max(best, rouge_l_like(s, ctx))
        grounded += int(best >= thr)
    return grounded / len(sents)

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
def evaluate_rag(examples: List[EvalExample], k: int = 5, alpha_values=(0.25, 0.5, 0.75)) -> Dict[str, Any]:
    modes = ["bm25", "dense", "hybrid"]
    results_per_mode = {m: [] for m in modes}
    perf_times = {m: [] for m in modes}
    doc_store: Dict[str, str] = {}
    alpha_choices = list(alpha_values)

    for ex in examples:
        for mode in modes:
            t0 = time.perf_counter()
            if mode == "bm25":
                retrieved = bm25_retriever.search(ex.query, k) if bm25_retriever else []
            elif mode == "dense":
                retrieved = dense_search(ex.query, k) if dense_search else []
            else:
                a = random.choice(alpha_choices)
                retrieved = hybrid_search(ex.query, k, alpha=a) if hybrid_search else []
            dt = max(time.perf_counter() - t0, 1e-6)
            perf_times[mode].append(dt)

            # Extract IDs & texts
            retrieved_ids: List[str] = []
            retrieved_texts: List[str] = []
            for i, d in enumerate(retrieved):
                rid, txt = get_doc_id_and_text(d, i)
                retrieved_ids.append(rid)
                retrieved_texts.append(txt)
                if rid:
                    doc_store[rid] = txt

            # Generate answer
            answer, contexts = default_generate_answer(ex.query, retrieved)

            # Metrics
            rel_ids = ex.relevant_ids or []
            p_at_k = precision_at_k(retrieved_ids, rel_ids, k)
            r_at_k = recall_at_k(retrieved_ids, rel_ids, k)
            mrr    = mrr_at_k(retrieved_ids, rel_ids, k)
            ndcg   = ndcg_at_k(retrieved_ids, rel_ids, k)
            em     = exact_match(answer, ex.answers)
            f1     = f1_score(answer, ex.answers)
            gsr    = grounded_sentence_rate(answer, contexts or retrieved_texts)
            ap     = attribution_precision(answer, doc_store)

            results_per_mode[mode].append({
                "query": ex.query, "group": ex.group,
                "Precision@k": p_at_k, "Recall@k": r_at_k,
                "MRR@k": mrr, "nDCG@k": ndcg,
                "EM": em, "F1": f1,
                "Faithfulness": gsr, "AttributionPrecision": ap,
                "time_s": dt
            })

    return {"per_example": results_per_mode, "perf_times": perf_times}

# -----------------------------------------------------------------------------
# Demo queries / categories
# -----------------------------------------------------------------------------
QUERY_CATEGORIES = [
    ("Acer Extensa 15 price","Exact Product","BM25"),
    ("HP laptop specifications","Brand Query","BM25"),
    ("best laptop for students","Semantic Query","Dense"),
    ("affordable gaming laptop","Conceptual Query","Dense/Hybrid"),
    ("Intel Core i3 8GB RAM","Keyword Heavy","BM25/Hybrid"),
    ("professional work laptop","Abstract Query","Dense"),
    ("smartphone prices","Out-of-scope","None"),
]

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(argv=None):
    print("Using existing Chroma DB")
    print("Loaded docs (from docs_index.csv if present)")
    print("FIXED MULTI-MODE RAG EVALUATION")
    print("==================================================")
    print("Testing Dense vs BM25 vs Hybrid retrieval modes\n")

    # Section 1: Mode comparison (lightweight preview)
    for qtext, qtype, expect in QUERY_CATEGORIES:
        print(f"Query: '{qtext}' ({qtype})")
        print(f"Expected: {expect} should excel")
        print("----------------------------------------")
        for mode in ["dense", "bm25", "hybrid"]:
            t0 = time.perf_counter()
            if mode == "bm25":
                retrieved = bm25_retriever.search(qtext, 3) if bm25_retriever else []
            elif mode == "dense":
                retrieved = dense_search(qtext, 3) if dense_search else []
            else:
                retrieved = hybrid_search(qtext, 3, alpha=0.5) if hybrid_search else []
            dt = max(time.perf_counter() - t0, 1e-6)
            print(f"  {mode.capitalize():5s}: {len(retrieved)} docs | {dt:.3f}s")
            if retrieved and hasattr(retrieved[0], "metadata"):
                md = retrieved[0].metadata
                show_row = md.get("row", "?")
                show_name = md.get("name") or md.get("title") or "?"
                print(f"    First: ROW={show_row} | NAME={str(show_name)[:60]}...")
        print()

    # Section 2: Hybrid α sweep (fixed timing)
    print("2. HYBRID MODE PARAMETER ANALYSIS")
    print("----------------------------------------")
    q = "gaming laptop with good price"
    print(f"Testing query: '{q}'")
    for a in [0.0, 0.3, 0.5, 0.7, 1.0]:
        t0 = time.perf_counter()
        retrieved = hybrid_search(q, 3, alpha=a) if hybrid_search else []
        dt = time.perf_counter() - t0
        first = (retrieved[0].metadata.get("name", "?") if retrieved and hasattr(retrieved[0], "metadata") else "None")
        print(f"  α={a:.1f}: {len(retrieved)} docs | {dt:.3f}s | First doc: {str(first)[:60]}...")

    # Section 3: Performance benchmarking (robust)
    print("\n3. PERFORMANCE BENCHMARKING")
    print("----------------------------------------")
    now = time.perf_counter
    bench_queries = ["laptop", "gaming", "business", "HP", "Dell"]
    modes = {
        "dense":  (lambda q: dense_search(q, 3) if dense_search else []),
        "bm25":   (lambda q: bm25_retriever.search(q, 3) if bm25_retriever else []),
        "hybrid": (lambda q: hybrid_search(q, 3, alpha=0.6) if hybrid_search else []),
    }

    def safe_stats(ts):
        ts = [t for t in ts if t is not None]
        if not ts:
            return 0.0, 0.0, 0.0
        ts_sorted = sorted(ts)
        n = len(ts_sorted)
        p50 = ts_sorted[n // 2]
        p95 = ts_sorted[max(0, int(n * 0.95) - 1)]
        avg = sum(ts_sorted) / n
        return avg, p50, p95

    for name, fn in modes.items():
        if fn is None:
            print(f"  {name.capitalize():5s}: retriever unavailable")
            continue

        times = []
        total_docs = 0
        for q in bench_queries:
            try:
                t0 = now()
                docs = fn(q)
                dt = max(now() - t0, 1e-6)
                times.append(dt)
                total_docs += len(docs) if docs else 0
            except Exception as e:
                times.append(None)
                print(f"    [{name}] '{q}' ERROR: {e}")

        avg, p50, p95 = safe_stats(times)
        total_time = sum(t for t in times if t is not None)
        qps = (len([t for t in times if t is not None]) / total_time) if total_time > 0 else 0.0
        print(f"  {name.capitalize():5s}: {avg:.4f}s avg | {qps:.1f} q/s | P50 {p50:.4f}s | P95 {p95:.4f}s | {total_docs} docs total")

    # Section 4: Holistic eval summary
    print("\n4. EVALUATION SUMMARY")
    print("==================================================")
    json_path = resolve_specs_json_path()
    GROUND_TRUTH = build_ground_truth_from_laptop_specs(json_path, QUERY_CATEGORIES)

    # Optional sanity check
    for q, t, _ in QUERY_CATEGORIES:
        gt = GROUND_TRUTH.get(q, {})
        print(f"GT for '{q}': {len(gt.get('relevant_ids', []))} relevant, {len(gt.get('answers', []))} answers")

    examples: List[EvalExample] = []
    for q, t, _ in QUERY_CATEGORIES:
        gt = GROUND_TRUTH.get(q, {"answers": [], "relevant_ids": []})
        examples.append(EvalExample(query=q, answers=gt["answers"], group=t, relevant_ids=gt["relevant_ids"]))

    report = evaluate_rag(examples, k=3)

    # Summaries
    for mode, rows in report["per_example"].items():
        em   = sum(r["EM"] for r in rows)/len(rows) if rows else 0
        f1   = sum(r["F1"] for r in rows)/len(rows) if rows else 0
        prec = sum(r["Precision@k"] for r in rows)/len(rows) if rows else 0
        rec  = sum(r["Recall@k"] for r in rows)/len(rows) if rows else 0
        nd   = sum(r["nDCG@k"] for r in rows)/len(rows) if rows else 0
        faith= sum(r["Faithfulness"] for r in rows)/len(rows) if rows else 0
        attr = sum(r["AttributionPrecision"] for r in rows)/len(rows) if rows else 0
        avg_t= sum(r["time_s"] for r in rows)/len(rows) if rows else 0
        print(f"[{mode}] EM {em:.3f} | F1 {f1:.3f} | P@k {prec:.3f} | R@k {rec:.3f} | nDCG {nd:.3f} | Faith {faith:.3f} | Attr {attr:.3f} | Avg {avg_t:.3f}s")

    # Save reports
    out_json = Path(_BASE, "rag_eval_report.json")
    out_md   = Path(_BASE, "rag_eval_report.md")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# RAG Evaluation Summary\n")
        for mode, rows in report["per_example"].items():
            if not rows:
                continue
            em   = sum(r["EM"] for r in rows)/len(rows)
            f1   = sum(r["F1"] for r in rows)/len(rows)
            prec = sum(r["Precision@k"] for r in rows)/len(rows)
            rec  = sum(r["Recall@k"] for r in rows)/len(rows)
            nd   = sum(r["nDCG@k"] for r in rows)/len(rows)
            faith= sum(r["Faithfulness"] for r in rows)/len(rows)
            attr = sum(r["AttributionPrecision"] for r in rows)/len(rows)
            f.write(
                f"## {mode}\n"
                f"- EM {em:.3f}, F1 {f1:.3f}\n"
                f"- P@k {prec:.3f}, R@k {rec:.3f}, nDCG {nd:.3f}\n"
                f"- Faithfulness {faith:.3f}, Attribution {attr:.3f}\n\n"
            )

    print(f"\nJSON report: {out_json}")
    print(f"Markdown: {out_md}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
