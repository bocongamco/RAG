import math
def dcg_at_k(gains, k): return sum(g / math.log2(i+2) for i, g in enumerate(gains[:k]))

def ndcg_at_k(labels_sorted, k):
    ideal = sorted(labels_sorted, reverse=True); idcg = dcg_at_k(ideal, k); dcg = dcg_at_k(labels_sorted, k)
    return 0.0 if idcg <= 0 else dcg / idcg

def mrr(labels_sorted):
    for i, g in enumerate(labels_sorted):
        if g > 0: return 1.0 / (i+1)
    return 0.0

def recall_at_k(labels_sorted, k):
    total_pos = sum(1 for g in labels_sorted if g > 0)
    if total_pos == 0: return 0.0
    hit = sum(1 for g in labels_sorted[:k] if g > 0); return hit / total_pos

def top1(labels_sorted): return 1.0 if labels_sorted and labels_sorted[0] > 0 else 0.0
