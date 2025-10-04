from __future__ import annotations
import csv, re
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

def _pick(cols, *names) -> Optional[str]:
    low = {c.lower(): c for c in cols}
    for n in names:
        if n in low: return low[n]
    return None

def _first_number(s):
    if pd.isna(s): return np.nan
    m = re.search(r"[-+]?\d*\.?\d+", str(s))
    return float(m.group()) if m else np.nan

def _price_num(s):
    if pd.isna(s): return np.nan
    s = str(s).replace(",", "")
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else np.nan

def _ram_gb(s):
    if pd.isna(s): return np.nan
    s = str(s).lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(gb|g|mb)", s)
    if not m: return _first_number(s)
    val = float(m.group(1)); unit = m.group(2)
    return val/1024.0 if unit == "mb" else val

def _storage_gb(s):
    if pd.isna(s): return np.nan
    s = str(s).lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(tb|t|gb|g)", s)
    if not m: return _first_number(s)
    val = float(m.group(1)); unit = m.group(2)
    return val*1024.0 if unit in ("tb","t") else val

def _screen_in(s):
    if pd.isna(s): return np.nan
    s = str(s).lower().replace("”", '"')
    m = re.search(r"(\d+(?:\.\d+)?)\s*(\"|in|\-inch|inch|inches)", s)
    return float(m.group(1)) if m else _first_number(s)

def _choose_id_column(df: pd.DataFrame) -> Optional[str]:
    for name in ["doc_id","id","unique_id","asin","sku","row","index"]:
        for c in df.columns:
            if c.lower() == name:
                return c
    return None

def make_qrels(csv_path="data/Amazon_Laptop_Specs.csv", out_path="data/qrels.csv",
               max_self=150, max_attr=150) -> int:
    df = pd.read_csv(csv_path)

    col_name   = _pick(df.columns, "name","title","product_name","laptop_name")
    col_brand  = _pick(df.columns, "brand","manufacturer")
    col_cpu    = _pick(df.columns, "cpu","processor","cpu_model","processor_type")
    col_ram    = _pick(df.columns, "ram","memory","system_memory")
    col_store  = _pick(df.columns, "storage","ssd","hdd","drive")
    col_screen = _pick(df.columns, "screen","display","screen_size","display_size","inch")

    work = df.copy()
    price_col = _pick(df.columns, "price","final_price","current_price","offer_price")
    if price_col: work["price_num"] = df[price_col].apply(_price_num)
    work["ram_gb"]     = df[col_ram].apply(_ram_gb)       if col_ram    else np.nan
    work["storage_gb"] = df[col_store].apply(_storage_gb) if col_store  else np.nan
    work["screen_in"]  = df[col_screen].apply(_screen_in) if col_screen else np.nan

    id_col = _choose_id_column(df)
    def doc_id_for_row(i: int) -> str:
        if id_col:
            v = df.iloc[i][id_col]
            return str(v) if not pd.isna(v) else str(i)
        return str(i)

    rows = []; qid = 0

    # 1) self queries
    if col_name:
        for i in range(min(max_self, len(df))):
            title = str(df.iloc[i][col_name]).strip()
            if not title: continue
            rows.append({"query_id": f"q{qid}", "query_text": title, "doc_id": doc_id_for_row(i), "label": 1})
            qid += 1

    # 2) attribute queries (brand/cpu/ram/storage/screen)
    count = 0
    for i in range(len(df)):
        if count >= max_attr: break
        parts = []
        brand = str(df.iloc[i][col_brand]).strip() if col_brand else ""
        cpu   = str(df.iloc[i][col_cpu]).strip() if col_cpu else ""
        ram   = work.iloc[i]["ram_gb"] if "ram_gb" in work else np.nan
        store = work.iloc[i]["storage_gb"] if "storage_gb" in work else np.nan
        inch  = work.iloc[i]["screen_in"] if "screen_in" in work else np.nan

        if brand: parts.append(brand)
        if cpu: parts.append(cpu)
        if not pd.isna(ram): parts.append(f"{int(round(ram))}GB")
        if not pd.isna(store):
            s = int(round(store))
            parts.append(f"{s}GB" if s < 1024 else f"{int(round(s/1024))}TB")
        if not pd.isna(inch): parts.append(f'{inch:g}"')

        qtext = " ".join([p for p in parts if p])
        if not qtext: continue

        rows.append({"query_id": f"q{qid}", "query_text": qtext, "doc_id": doc_id_for_row(i), "label": 1})
        qid += 1; count += 1

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query_id","query_text","doc_id","label"])
        w.writeheader(); w.writerows(rows)
    return len(rows)
