from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict
import csv, numpy as np

# expect these to exist in your vector.py:
#   bm25_get(query: str, k: int) -> List[Document]
#   dense_get_with_score(query: str, k: int) -> List[Tuple[Document, float]]

from Src.search.vector import bm25_get, dense_get_with_score

EPS = 1e-9
def _tanh_norm(arr):
    x = np.asarray(arr, dtype=float); mu, sd = x.mean(), x.std() + EPS
    return np.tanh((x - mu) / sd)

@dataclass
class Qrels:
    data: Dict[str, tuple[str, set]]
    def query_ids(self): return list(self.data.keys())
    def pack(self, qids: List[str]): return [(self.data[q][0], Labels(self.data[q][1])) for q in qids]

@dataclass
class Labels:
    pos_ids: set
    def sorted_by_fused(self, candidates, alpha: float):
        if not candidates: return []
        # rank-based fusion using (k - rank) from each list
        k = len(candidates)
        bm = bm25_get("_join_all_", k) or []
        dn = [d for d, _ in dense_get_with_score("_join_all_", k)] or []

        bm_rank = {d.page_content: i for i, d in enumerate(bm)} if bm else {d.page_content: i for i, d in enumerate(candidates)}
        dn_rank = {d.page_content: i for i, d in enumerate(dn)} if dn else {d.page_content: i for i, d in enumerate(candidates)}

        fused=[]
        for d in candidates:
            br = bm_rank.get(d.page_content, k-1); dr = dn_rank.get(d.page_content, k-1)
            fused.append((alpha * (k - dr)) + ((1.0 - alpha) * (k - br)))

        order = sorted(range(len(candidates)), key=lambda i: fused[i], reverse=True)
        y_sorted=[]
        for i in order:
            did = candidates[i].metadata.get("doc_id") or candidates[i].metadata.get("id")
            y_sorted.append(1 if did in self.pos_ids else 0)
        return y_sorted

def load_qrels(path: str) -> Qrels:
    data={}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if int(row.get("label","1")) <= 0: continue
            qid = row["query_id"]; qtext = row["query_text"]; did = row["doc_id"]
            if qid not in data: data[qid] = (qtext, set())
            data[qid][1].add(did)
    return Qrels(data)

_DENSE_CACHE = {}
def _dense(q, k):
    v = _DENSE_CACHE.get((q,k))
    if v is None:
        v = [d for d,_ in dense_get_with_score(q, k)]
        _DENSE_CACHE[(q,k)] = v
    return v
    
def build_candidates_for_query(qtext: str, k_each: int = 10):
    bm = bm25_get(qtext, k_each) or []
    dn = _dense(qtext, k_each) or []
    seen=set(); out=[]
    for d in bm + dn:
        key = d.page_content
        if key not in seen:
            seen.add(key); out.append(d)
    return out
