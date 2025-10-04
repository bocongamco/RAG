from __future__ import annotations

import json, re
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

# matplotlib (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------
# Output location
# ---------------------------
DEFAULT_OUT = Path("Outputs/eda")  # change if you prefer
DEFAULT_OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Small helpers
# ---------------------------
def _pick(cols: List[str], *names: str) -> Optional[str]:
    """Pick the first column whose lowercase name matches any of *names*."""
    low = {c.lower(): c for c in cols}
    for n in names:
        if n in low:
            return low[n]
    return None

def _first_number(s) -> float:
    if pd.isna(s): return np.nan
    m = re.search(r"[-+]?\d*\.?\d+", str(s))
    return float(m.group()) if m else np.nan

def _price_num(s) -> float:
    if pd.isna(s): return np.nan
    t = str(s).replace(",", "")
    m = re.search(r"(\d+(?:\.\d+)?)", t)
    return float(m.group(1)) if m else np.nan

def _rating_num(s) -> float:
    # handles "4.2 out of 5", "4.3/5", "4.1"
    return _first_number(s)

def _ratings_count(s) -> float:
    # handles "1,234 ratings" -> 1234
    if pd.isna(s): return np.nan
    t = str(s).replace(",", "")
    m = re.search(r"\d+", t)
    return float(m.group()) if m else np.nan

def _rank_num(s) -> float:
    # handles "Best Sellers Rank: #2,345 in Laptops"
    if pd.isna(s): return np.nan
    t = str(s).replace(",", "")
    m = re.search(r"\d+", t)
    return float(m.group()) if m else np.nan

def _ram_gb(s) -> float:
    if pd.isna(s): return np.nan
    t = str(s).lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(gb|g|mb)", t)
    if not m: return _first_number(t)
    val = float(m.group(1)); unit = m.group(2)
    return val/1024.0 if unit == "mb" else val

def _storage_gb(s) -> float:
    if pd.isna(s): return np.nan
    t = str(s).lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(tb|t|gb|g)", t)
    if not m: return _first_number(t)
    val = float(m.group(1)); unit = m.group(2)
    return val*1024.0 if unit in ("tb","t") else val

def _screen_in(s) -> float:
    if pd.isna(s): return np.nan
    t = str(s).lower().replace("”", '"').replace("“", '"')
    m = re.search(r"(\d+(?:\.\d+)?)\s*(\"|in|\-inch|inch|inches)", t)
    return float(m.group(1)) if m else _first_number(t)

def _weight_kg(s) -> float:
    if pd.isna(s): return np.nan
    t = str(s).lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(kg|g|lbs|lb)", t)
    if not m: return _first_number(t)
    val = float(m.group(1)); unit = m.group(2)
    if unit == "kg": return val
    if unit == "g":  return val/1000.0
    # lbs / lb
    return val * 0.453592

# ----- dimension/volume parsing (unit-aware) -----
_DIM_PAT = r"(?P<a>\d+(?:\.\d+)?)\s*[x×*]\s*(?P<b>\d+(?:\.\d+)?)\s*[x×*]\s*(?P<c>\d+(?:\.\d+)?)"

def _to_cm(value: float, unit: str) -> float:
    u = (unit or "").lower().strip()
    if u in ("cm", "centimeter", "centimeters"): return value
    if u in ('"', "in", "inch", "inches"):       return value * 2.54
    if u in ("mm", "millimeter", "millimeters"): return value / 10.0
    # default assume cm
    return value

def _parse_dims_to_cm3(s) -> float:
    """Return volume in cubic centimeters from strings like '36.1 x 24.5 x 1.8 cm' or '14\" x 9\" x 0.7\"'."""
    if pd.isna(s): return np.nan
    txt = str(s).lower().replace("”", '"').replace("“", '"').replace("′","'")
    # global unit detection
    global_unit = None
    if "cm" in txt: global_unit = "cm"
    elif '"' in txt or " inch" in txt or "inches" in txt or " in " in txt: global_unit = "in"
    elif "mm" in txt: global_unit = "mm"

    m = re.search(_DIM_PAT, txt)
    if not m:
        nums = re.findall(r"\d+(?:\.\d+)?", txt)
        if len(nums) < 3: return np.nan
        a, b, c = map(float, nums[:3])
    else:
        a = float(m.group("a")); b = float(m.group("b")); c = float(m.group("c"))

    a = _to_cm(a, global_unit or "cm")
    b = _to_cm(b, global_unit or "cm")
    c = _to_cm(c, global_unit or "cm")

    vol = a * b * c
    # sanity guard: typical laptop ~ 30×22×2 cm = 1320 cm³; cap out impossible values
    if vol <= 0 or vol > 100000:   # > 100k cm³ ≈ suitcase
        return np.nan
    return vol

# ---------------------------
# Main EDA builder
# ---------------------------
def build_eda(csv_path: str = "Data/Training-Data/Amazon_Laptop_Specs.csv",
              out_dir: str | Path = DEFAULT_OUT) -> Dict[str, Any]:
    """
    Reads the CSV, derives numeric features, writes:
      - {out_dir}/eda_summary.json
      - {out_dir}/corr_heatmap.png
      - {out_dir}/dist_*.png (for each numeric field)
    Returns the summary dict.
    """
    out_path = Path(out_dir); out_path.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)

    # ---- column picks (robust to different schemas) ----
    col_price   = _pick(df.columns, "price","final_price","current_price","offer_price")
    col_rating  = _pick(df.columns, "customer rating","rating","ratings","avg_rating")
    col_count   = _pick(df.columns, "number of ratings","ratings_count","review_count","reviews")
    col_rank    = _pick(df.columns, "best sellers rank","rank","sales_rank")
    col_dims    = _pick(df.columns, "item dimensions lxwxh","dimensions","item dimensions","product dimensions")
    col_ram     = _pick(df.columns, "ram","memory","system_memory")
    col_store   = _pick(df.columns, "storage","ssd","hdd","drive")
    col_screen  = _pick(df.columns, "screen","display","screen_size","display_size","inch")
    col_weight  = _pick(df.columns, "weight","item_weight")
    col_brand   = _pick(df.columns, "brand","manufacturer")
    col_cpu     = _pick(df.columns, "cpu","processor","cpu_model","processor_type")
    col_gpu     = _pick(df.columns, "gpu","graphics","graphics_card")

    work = df.copy()

    # ---- numeric features ----
    if col_price:   work["price_num"]   = work[col_price].apply(_price_num)
    if col_rating:  work["rating_num"]  = work[col_rating].apply(_rating_num)
    if col_count:   work["ratings_count"]= work[col_count].apply(_ratings_count)
    if col_rank:    work["rank_num"]    = work[col_rank].apply(_rank_num)
    if col_dims:    work["volume_cc"]   = work[col_dims].apply(_parse_dims_to_cm3)
    if col_ram:     work["ram_gb"]      = work[col_ram].apply(_ram_gb)
    if col_store:   work["storage_gb"]  = work[col_store].apply(_storage_gb)
    if col_screen:  work["screen_in"]   = work[col_screen].apply(_screen_in)
    if col_weight:  work["weight_kg"]   = work[col_weight].apply(_weight_kg)

    # light outlier clip to make plots readable
    for c, q in (("price_num", 0.995), ("ratings_count", 0.995), ("rank_num", 0.995)):
        if c in work.columns:
            hi = work[c].quantile(q)
            work.loc[work[c] > hi, c] = np.nan

    # ---- numeric summary ----
    num_cols = [c for c in [
        "price_num","rating_num","ratings_count","rank_num","volume_cc",
        "ram_gb","storage_gb","screen_in","weight_kg"
    ] if c in work.columns]

    numeric_summary: Dict[str, Dict[str, float]] = {}
    if num_cols:
        desc = work[num_cols].describe(percentiles=[.1,.25,.5,.75,.9]).T.round(4)
        numeric_summary = {idx: {k: float(v) for k, v in row.items()}  # type: ignore[arg-type]
                           for idx, row in desc.iterrows()}

    # ---- categorical top-k ----
    categorical_top: Dict[str, Dict[str, int]] = {}
    for c in [col_brand, col_cpu, col_gpu]:
        if not c: continue
        vc = work[c].astype(str).str.strip().str.lower().value_counts().head(25)
        categorical_top[c] = {k: int(v) for k, v in vc.items()}

    # ---- correlations + heatmap ----
    correlations: Dict[str, Dict[str, float]] = {}
    if num_cols:
        cm = work[num_cols].corr(numeric_only=True).fillna(0.0)
        correlations = cm.round(3).to_dict(orient="index")
        plt.figure(figsize=(7,6))
        plt.imshow(cm, interpolation="nearest")
        plt.xticks(range(len(num_cols)), num_cols, rotation=45, ha="right")
        plt.yticks(range(len(num_cols)), num_cols)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(out_path / "corr_heatmap.png", dpi=160)
        plt.close()

    # ---- histograms ----
    for c in num_cols:
        s = work[c].dropna().astype(float)
        if s.empty: continue
        plt.figure(figsize=(11,8))
        plt.hist(s, bins=30)
        plt.title(c); plt.xlabel(c); plt.ylabel("count"); plt.tight_layout()
        plt.savefig(out_path / f"dist_{c.replace('_','')}.png", dpi=140)
        plt.close()

    # ---- write summary ----
    out = {
        "csv_path": str(Path(csv_path).resolve()),
        "numeric_summary": numeric_summary,
        "categorical_top": categorical_top,
        "correlations": correlations,
        "columns_used": {
            "price": col_price, "rating": col_rating, "ratings_count": col_count,
            "rank": col_rank, "dimensions": col_dims, "ram": col_ram,
            "storage": col_store, "screen": col_screen, "weight": col_weight,
            "brand": col_brand, "cpu": col_cpu, "gpu": col_gpu
        }
    }
    (out_path / "eda_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out

if __name__ == "__main__":
    # quick local test
    build_eda()
