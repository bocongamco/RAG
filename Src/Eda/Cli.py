from __future__ import annotations

import argparse, os, shutil, stat
from pathlib import Path
import pandas as pd

from .summary import build_eda
from .qrels import make_qrels as make_qrels_from_csv  # CSV-based fallback

# --------- project roots (case-insensitive Windows safe) ----------
ROOT = Path(__file__).resolve().parents[2]

# Defaults (can be overridden by flags)
DATA_DIR        = ROOT / "Data"
DATA_TRAIN_CSV  = DATA_DIR / "Training-Data" / "Amazon_Laptop_Specs.csv"
DATA_FALLBACK   = DATA_DIR / "Amazon_Laptop_Specs.csv"

INDEX_DIR       = ROOT / "data_index"
INDEX_CSV       = INDEX_DIR / "docs_index.csv"

EDA_OUT_DEFAULT = ROOT / "Outputs" / "eda"


def _abs(p: str | Path) -> Path:
    """Resolve `p` to an absolute path. Relative paths are treated as ROOT-relative."""
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)


def _pick_csv(user_csv: str | None) -> Path:
    """Choose the CSV to use when --csv is omitted."""
    if user_csv:
        return _abs(user_csv)
    return DATA_TRAIN_CSV if DATA_TRAIN_CSV.exists() else DATA_FALLBACK


# ---------- robust folder wipe (Windows-safe, clears read-only) ----------
def _on_rm_error(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass  # continue best-effort

def _clean_out_dir(out_dir: Path):
    if out_dir.exists():
        shutil.rmtree(out_dir, onerror=_on_rm_error)
    out_dir.mkdir(parents=True, exist_ok=True)


def _make_qrels_from_index(out_path: Path, index_csv: Path) -> int:
    """
    Make qrels from data_index/docs_index.csv so doc_ids exactly match your retrievers.
    Expects columns: doc_id, content, row, name (as produced by src/search/vector.py).
    """
    if not index_csv.exists():
        raise FileNotFoundError(
            f"Missing {index_csv}. Run your indexing step (BM25/Chroma build) or use --force-from-csv."
        )

    idx = pd.read_csv(index_csv)  # columns: doc_id, content, row, name, price, rating
    rows = []
    for _, r in idx.iterrows():
        row_i = int(r["row"])
        qid   = f"q_{row_i}"
        qtext = str(r.get("name", "")).strip() or f"laptop row {row_i}"
        did   = str(r["doc_id"])
        rows.append({"query_id": qid, "query_text": qtext, "doc_id": did, "label": 1})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["query_id", "query_text", "doc_id", "label"]).to_csv(out_path, index=False)
    return len(rows)


def main():
    p = argparse.ArgumentParser("EDA & qrels (RAG laptops)")
    # EDA
    p.add_argument("--rebuild", action="store_true",
                   help="Overwrite the --out folder: wipe it first, then build EDA summary + charts.")
    p.add_argument("--csv", type=str, default=None,
                   help="Path to Amazon_Laptop_Specs.csv (default: Data/Training-Data/... or Data/...).")
    p.add_argument("--out", type=str, default=str(EDA_OUT_DEFAULT),
                   help="Output folder for EDA artifacts (default: Outputs/eda).")

    # Qrels
    p.add_argument("--make-qrels", type=str, default=None,
                   help="Write qrels CSV to this path (e.g., Data/qrels.csv).")
    p.add_argument("--index-csv", type=str, default=str(INDEX_CSV),
                   help="Path to data_index/docs_index.csv (default: data_index/docs_index.csv).")
    p.add_argument("--force-from-csv", action="store_true",
                   help="Force qrels generation directly from the product CSV "
                        "(IDs may not align with retrievers unless your generator uses data_index).")

    args = p.parse_args()

    csv_path   = _pick_csv(args.csv)
    out_dir    = _abs(args.out)
    index_path = _abs(args.index_csv)

    # ---- EDA ----
    if args.rebuild:
        # OVERWRITE behavior
        _clean_out_dir(out_dir)
        out = build_eda(csv_path=str(csv_path), out_dir=str(out_dir))
        print(f"EDA → {(out_dir / 'eda_summary.json').resolve()} + charts (overwritten)")
    else:
        # idempotent (no wipe)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = build_eda(csv_path=str(csv_path), out_dir=str(out_dir))
        print(f"EDA → {(out_dir / 'eda_summary.json').resolve()} + charts")

    # ---- Qrels ----
    if args.make_qrels:
        out_path = _abs(args.make_qrels)
        if args.force_from_csv:
            n = make_qrels_from_csv(csv_path=str(csv_path), out_path=str(out_path))
            print(f"Wrote {n} rows to {out_path} (from CSV).")
        else:
            try:
                n = _make_qrels_from_index(out_path, index_path)
                print(f"Wrote {n} rows to {out_path} (from {index_path}).")
            except FileNotFoundError as e:
                # fallback if index not present yet
                print(f"[warn] {e}")
                n = make_qrels_from_csv(csv_path=str(csv_path), out_path=str(out_path))
                print(f"Wrote {n} rows to {out_path} (from CSV; consider rebuilding index to keep IDs aligned).")


if __name__ == "__main__":
    main()
