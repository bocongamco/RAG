# src/alpha/train.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import numpy as np
import pandas as pd

# Your retrieval wrappers
from ..search.vector import bm25_get, dense_get_with_score, init_stores


# ----------------------- Data structures & I/O -----------------------
@dataclass
class QrelsItem:
    query_id: str
    query_text: str
    doc_id: str
    label: int


def _load_qrels(path: str | Path) -> Tuple[List[str], Dict[str, List[QrelsItem]]]:
    """Load qrels CSV with columns: query_id, query_text, doc_id[, label]."""
    df = pd.read_csv(path)
    if "label" not in df.columns:
        df["label"] = 1
    df["query_id"] = df["query_id"].astype(str)
    df["doc_id"] = df["doc_id"].astype(str)
    df["query_text"] = df["query_text"].astype(str)

    qrels_by_qid: Dict[str, List[QrelsItem]] = {}
    for _, r in df.iterrows():
        qid = r["query_id"]
        qrels_by_qid.setdefault(qid, []).append(
            QrelsItem(
                query_id=qid,
                query_text=r["query_text"],
                doc_id=r["doc_id"],
                label=int(r["label"]),
            )
        )
    return list(qrels_by_qid.keys()), qrels_by_qid


# ----------------------- Retrieval candidates cache -----------------------
def _build_candidates_for_query(qtext: str, k_each: int) -> Tuple[List[str], Dict[str, int], Dict[str, int]]:
    """
    Returns:
      union_ids: union of doc_ids from dense and BM25
      dense_rank: doc_id -> rank position (0..), missing => large
      bm25_rank : doc_id -> rank position (0..), missing => large
    """
    dense_pairs = dense_get_with_score(qtext, k=k_each)  # [(Document, score)]
    dense_docs = [d for (d, _) in dense_pairs]
    dense_ids = [d.metadata.get("doc_id", "") for d in dense_docs]

    bm25_docs = bm25_get(qtext, k=k_each)  # [Document]
    bm25_ids = [d.metadata.get("doc_id", "") for d in bm25_docs]

    union_ids = list(dict.fromkeys(dense_ids + bm25_ids))
    big = k_each + 5
    dense_rank = {did: (dense_ids.index(did) if did in dense_ids else big) for did in union_ids}
    bm25_rank = {did: (bm25_ids.index(did) if did in bm25_ids else big) for did in union_ids}
    return union_ids, dense_rank, bm25_rank


def _build_cache_for_queries(qids: List[str],
                             qrels: Dict[str, List[QrelsItem]],
                             k_each: int,
                             tag: str,
                             log_every: int = 20) -> Dict[str, Tuple[List[str], Dict[str, int], Dict[str, int]]]:
    cache: Dict[str, Tuple[List[str], Dict[str, int], Dict[str, int]]] = {}
    for i, qid in enumerate(qids, 1):
        qtext = qrels[qid][0].query_text
        cache[qid] = _build_candidates_for_query(qtext, k_each=k_each)
        if (i % log_every) == 0:
            print(f"[alpha]   ({tag}) built candidates for {i}/{len(qids)} queries")
    return cache


# ----------------------- Metrics & fusion -----------------------
def _dcg_at_k(gains: np.ndarray, k: int) -> float:
    k = min(k, gains.shape[0])
    if k <= 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    return float((gains[:k] * discounts).sum())


def _ndcg_for_query(relevant_set: Dict[str, int], ranked_ids: List[str], k: int) -> float:
    gains = np.array([relevant_set.get(did, 0) for did in ranked_ids], dtype=float)
    dcg = _dcg_at_k(gains, k)
    ideal = np.sort(list(relevant_set.values()))[::-1]
    if len(ideal) == 0:
        return 0.0
    idcg = _dcg_at_k(np.array(ideal, dtype=float), k)
    return 0.0 if idcg <= 0 else dcg / idcg


def _fuse_and_rank(union_ids: List[str],
                   dense_rank: Dict[str, int],
                   bm25_rank: Dict[str, int],
                   alpha: float,
                   k_score: int) -> List[str]:
    """
    Rank docs by fused score: s = α*(K - rd) + (1-α)*(K - rb),
    where rd/rb are rank positions; K is k_score (use k_each or similar).
    """
    K = k_score

    def score(did: str) -> float:
        rd = dense_rank.get(did, K + 5)
        rb = bm25_rank.get(did, K + 5)
        return alpha * (K - rd) + (1.0 - alpha) * (K - rb)

    return sorted(union_ids, key=lambda did: score(did), reverse=True)


def _mean_ndcg_for_list(qids: List[str],
                        rel_sets: Dict[str, Dict[str, int]],
                        cache: Dict[str, Tuple[List[str], Dict[str, int], Dict[str, int]]],
                        alpha: float,
                        k: int,
                        k_each: int) -> float:
    if not qids:
        return 0.0
    scores = []
    for qid in qids:
        union_ids, d_rank, b_rank = cache[qid]
        ranked = _fuse_and_rank(union_ids, d_rank, b_rank, alpha, k_score=k_each)
        scores.append(_ndcg_for_query(rel_sets[qid], ranked, k))
    return float(np.mean(scores))


# ----------------------- Selection helpers -----------------------
def _choose_alpha_val(alphas: List[float],
                      train_means: List[float],
                      val_means: List[float],
                      tie_break: str = "min_alpha") -> Tuple[float, float, float]:
    """
    Choose α by highest validation mean; tie-break by:
      - 'min_alpha'  -> smallest α among equals
      - 'max_train'  -> highest train mean among equals
    Returns (best_alpha, best_train_mean, best_val_mean).
    """
    best_val = max(val_means) if val_means else -1.0
    candidates = [i for i, v in enumerate(val_means) if abs(v - best_val) <= 1e-12]
    if len(candidates) == 1:
        i = candidates[0]
        return alphas[i], train_means[i], val_means[i]

    if tie_break == "max_train":
        # among ties on val, pick the one with max train
        best_train = -1.0
        best_i = candidates[0]
        for i in candidates:
            if train_means[i] > best_train + 1e-12:
                best_train = train_means[i]
                best_i = i
        return alphas[best_i], train_means[best_i], val_means[best_i]
    else:
        # default: smallest alpha among equals
        i = min(candidates, key=lambda j: alphas[j])
        return alphas[i], train_means[i], val_means[i]


def _choose_alpha_cv(alphas: List[float],
                     fold_means: List[List[float]],
                     one_std_err: bool = True,
                     tie_break: str = "min_alpha") -> Tuple[float, float, float]:
    """
    alphas: list of candidate α
    fold_means: shape [A][F] (val means per alpha per fold)
    Returns (alpha*, mean_val_at_alpha*, std_val_at_alpha*).
    """
    A = len(alphas)
    F = len(fold_means[0]) if A > 0 else 0
    means = [float(np.mean(fold_means[a])) for a in range(A)]
    stds = [float(np.std(fold_means[a], ddof=1)) if F > 1 else 0.0 for a in range(A)]
    sems = [std / np.sqrt(F) if F > 0 else 0.0 for std in stds]

    best_idx = int(np.argmax(means)) if means else 0
    if one_std_err and A > 0:
        thresh = means[best_idx] - sems[best_idx]
        eligible = [i for i in range(A) if means[i] >= thresh - 1e-12]
        if tie_break == "min_alpha":
            pick = min(eligible, key=lambda i: alphas[i])
        else:
            # highest mean among eligible, then smallest alpha
            top = max(means[i] for i in eligible)
            tie = [i for i in eligible if abs(means[i] - top) <= 1e-12]
            pick = min(tie, key=lambda i: alphas[i])
        best_idx = pick
    else:
        # simple argmax; tie-break by smallest alpha
        top = max(means) if means else -1.0
        tie = [i for i, m in enumerate(means) if abs(m - top) <= 1e-12]
        best_idx = min(tie, key=lambda i: alphas[i])

    return alphas[best_idx], means[best_idx], stds[best_idx]


# ----------------------- Main API -----------------------
def fit_and_eval(qrels_path: str | Path,
                 k: int = 5,
                 seed: int = 42,
                 grid: Optional[List[float]] = None,
                 k_each: int = 25,
                 val_frac: float = 0.20,
                 test_frac: float = 0.15,
                 out_dir: Optional[str | Path] = None,
                 select: str = "cv",             # "cv" or "val"
                 cv_folds: int = 5,
                 one_std_err: bool = False,
                 tie_break: str = "min_alpha"    # "min_alpha" or "max_train"
                 ) -> Dict[str, float]:

    """
    Best-practice α selection:
      - Hold out TEST (test_frac).
      - On NON-TEST, select α by:
          * K-fold CV on validation means (default), with one-std-err rule, or
          * Single validation split (select by val mean).
      - Finally, evaluate TEST once at chosen α.

    Returns summary dict (also writes artifacts if out_dir is provided).
    """

    init_stores()

    # Prepare output dir
    out_path = Path(out_dir) if out_dir else None
    if out_path:
        out_path.mkdir(parents=True, exist_ok=True)

    # Load qrels and prepare relevance sets
    queries, qrels = _load_qrels(qrels_path)
    rel_sets: Dict[str, Dict[str, int]] = {qid: {it.doc_id: it.label for it in items} for qid, items in qrels.items()}

    # Build splits: test vs non-test
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(queries))
    n = len(queries)
    n_test = int(round(n * test_frac))
    n_val_single = int(round((n - n_test) * val_frac))  # used only in "val" mode reporting
    non_test_idx = perm[: n - n_test]
    test_idx = perm[n - n_test :] if n_test > 0 else np.array([], dtype=int)

    non_test_q = [queries[i] for i in non_test_idx]
    test_q = [queries[i] for i in test_idx]

    print(f"[alpha] split: non-test={len(non_test_q)} | test={len(test_q)} | total={n}")
    if out_path:
        with open(out_path / "splits.json", "w", encoding="utf-8") as f:
            json.dump({"non_test": non_test_q, "test": test_q}, f, indent=2)

    # Candidate cache for NON-TEST (used in selection)
    print(f"[alpha] building candidates for non-test ({len(non_test_q)} queries) ...")
    cache_non_test = _build_cache_for_queries(non_test_q, qrels, k_each=k_each, tag="non-test", log_every=20)

    # Alpha grid
    if not grid:
        grid = [round(x, 2) for x in np.linspace(0.0, 1.0, 21)]  # 0.00..1.00 step 0.05
    alphas = list(grid)

    # Selection: CV or single validation
    chosen_alpha = 0.6
    selection_info: Dict[str, float | int] = {}

    if select.lower() == "cv" and len(non_test_q) >= cv_folds and cv_folds >= 2:
        print(f"[alpha] selection = CV ({cv_folds} folds) + one-std-err={one_std_err}")
        # Build folds on non-test
        rng2 = np.random.default_rng(seed + 13)
        shuffled = list(non_test_q)
        rng2.shuffle(shuffled)
        folds: List[List[str]] = []
        # split into cv_folds approximately-equal chunks
        fold_sizes = [(len(shuffled) + i) // cv_folds for i in range(cv_folds)]
        start = 0
        for fs in fold_sizes:
            folds.append(shuffled[start:start + fs])
            start += fs

        # For each alpha, compute validation nDCG for each fold
        fold_means: List[List[float]] = [[] for _ in alphas]  # [A][F]
        for fi, val_fold in enumerate(folds):
            # val set = this fold; (we don't actually use "train folds" for α since no training)
            for ai, a in enumerate(alphas):
                m = _mean_ndcg_for_list(val_fold, rel_sets, cache_non_test, alpha=a, k=k, k_each=k_each)
                fold_means[ai].append(m)
            print(f"[alpha] CV fold {fi+1}/{cv_folds} computed.")

        chosen_alpha, cv_mean, cv_std = _choose_alpha_cv(alphas, fold_means, one_std_err=one_std_err, tie_break=tie_break)
        selection_info = {"cv_mean_val": cv_mean, "cv_std_val": cv_std, "cv_folds": cv_folds}

        # Save CV curve
        if out_path:
            rows = []
            for ai, a in enumerate(alphas):
                for fi, m in enumerate(fold_means[ai]):
                    rows.append({"alpha": a, "fold": fi, "val_ndcg": m})
            pd.DataFrame(rows).to_csv(out_path / "cv_curve.csv", index=False, encoding="utf-8")

    else:
        # Single validation split inside NON-TEST
        print(f"[alpha] selection = single validation (val_frac={val_frac:.2f})")
        rng3 = np.random.default_rng(seed + 29)
        non_perm = rng3.permutation(len(non_test_q))
        val_idx = non_perm[:n_val_single]
        train_idx = non_perm[n_val_single:]
        val_q = [non_test_q[i] for i in val_idx]
        train_q = [non_test_q[i] for i in train_idx]

        cache_train = {qid: cache_non_test[qid] for qid in train_q}
        cache_val = {qid: cache_non_test[qid] for qid in val_q}

        train_means, val_means = [], []
        for a in alphas:
            t_mean = _mean_ndcg_for_list(train_q, rel_sets, cache_train, alpha=a, k=k, k_each=k_each)
            v_mean = _mean_ndcg_for_list(val_q, rel_sets, cache_val, alpha=a, k=k, k_each=k_each)
            print(f"[alpha] try α={a:.2f} -> train nDCG={t_mean:.3f} | val nDCG={v_mean:.3f}")
            train_means.append(t_mean)
            val_means.append(v_mean)

        chosen_alpha, best_train, best_val = _choose_alpha_val(alphas, train_means, val_means, tie_break=tie_break)
        selection_info = {"val_best": best_val, "train_at_val_best": best_train, "val_frac": val_frac}

        # Save alpha_curve for single-val mode
        if out_path:
            pd.DataFrame({"alpha": alphas, "train_ndcg": train_means, "val_ndcg": val_means}) \
                .to_csv(out_path / "alpha_curve.csv", index=False, encoding="utf-8")

    # ----------------------- Final evaluation -----------------------
    print(f"[alpha] chosen α={chosen_alpha:.2f}")

    # For reporting train/val numbers, create a fresh single split on NON-TEST
    rng4 = np.random.default_rng(seed + 97)
    non_perm2 = rng4.permutation(len(non_test_q))
    n_val_report = n_val_single
    val_rep_idx = non_perm2[:n_val_report]
    train_rep_idx = non_perm2[n_val_report:]
    val_rep_q = [non_test_q[i] for i in val_rep_idx]
    train_rep_q = [non_test_q[i] for i in train_rep_idx]

    train_ndcg = _mean_ndcg_for_list(train_rep_q, rel_sets, cache_non_test, alpha=chosen_alpha, k=k, k_each=k_each)
    val_ndcg = _mean_ndcg_for_list(val_rep_q, rel_sets, cache_non_test, alpha=chosen_alpha, k=k, k_each=k_each)

    # Test: build candidates lazily (not cached yet)
    test_ndcg = None
    if len(test_q) > 0:
        print(f"[alpha] evaluating TEST ({len(test_q)} queries)")
        test_scores = []
        for qid in test_q:
            qtext = qrels[qid][0].query_text
            union_ids, d_rank, b_rank = _build_candidates_for_query(qtext, k_each=k_each)
            ranked = _fuse_and_rank(union_ids, d_rank, b_rank, chosen_alpha, k_score=k_each)
            test_scores.append(_ndcg_for_query(rel_sets[qid], ranked, k))
        test_ndcg = float(np.mean(test_scores)) if test_scores else 0.0

    # Save summary/artifacts
    if out_path:
        summary = {
            "alpha": chosen_alpha,
            "train_ndcg": float(train_ndcg),
            "val_ndcg": float(val_ndcg),
            **({"test_ndcg": float(test_ndcg)} if test_ndcg is not None else {}),
            "k": k,
            "k_each": k_each,
            "seed": seed,
            "select": select,
            "grid": alphas,
            "sizes": {"non_test": len(non_test_q), "test": len(test_q), "total": n},
            **selection_info,
        }
        (out_path / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "alpha": chosen_alpha,
        "train_ndcg": float(train_ndcg),
        "val_ndcg": float(val_ndcg),
        **({"test_ndcg": float(test_ndcg)} if test_ndcg is not None else {}),
    }
