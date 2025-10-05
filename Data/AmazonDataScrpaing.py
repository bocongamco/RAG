import requests, time, random, csv, json, re
from bs4 import BeautifulSoup

import csv, re
from pathlib import Path

URL_REGEX = re.compile(r"https?://[^\s,'\"]+")

def clean_url_cell(cell: str) -> str | None:
    if not cell:
        return None
    s = cell.strip().strip('"').strip("'")
    m = URL_REGEX.search(s)
    return m.group(0) if m else None

def load_urls_from_csv(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    urls, seen = [], set()
    with open(path, newline="", encoding="utf-8") as f:
        peek = f.readline()
        f.seek(0)
        has_header = "url" in [h.strip().lower() for h in peek.split(",")]

        if has_header:
            reader = csv.DictReader(f)
            for row in reader:
                u = clean_url_cell((row.get("url") or "").strip())
                if u and u not in seen:
                    seen.add(u); urls.append(u)
        else:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if not row:
                    continue
                u = clean_url_cell(row[0])
                if u and u not in seen:
                    seen.add(u); urls.append(u)
    return urls

URLS = load_urls_from_csv("amazon_laptop_urls.csv")

print(f"Loaded {len(URLS)} URLs from amazon_urls.csv")
CSV_PATH  = "amazon_products_wide.csv"
JSON_PATH = "amazon_products.json"

# ------------------ polite session ------------------
session = requests.Session()
session.headers.update({
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/122.0.0.0 Safari/537.36"),
    "Accept-Language": "en-AU,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.google.com/"
})

def polite_get(url, lo=1.5, hi=3.5, timeout=25):
    try:
        r = session.get(url, timeout=timeout)
        print("Fetched", url, "->", r.status_code)
        time.sleep(random.uniform(lo, hi))
        return r.text if r.status_code == 200 else None
    except requests.exceptions.RequestException as e:
        print("Fetch error:", e)
        return None

# ------------------ extract helpers ------------------
def get_text(node):
    return node.get_text(" ", strip=True) if node else ""

def extract_title(soup):
    t = soup.select_one("#productTitle")
    if t: return get_text(t)
    for sel in ('meta[property="og:title"]', 'h1'):
        n = soup.select_one(sel)
        if n and n.get("content"):
            return n["content"].strip()
        if n:
            return get_text(n)
    return "(unknown product)"

def extract_price(soup):
    for sel in (".a-price .a-offscreen", "#corePrice_feature_div .a-offscreen",
                "#priceblock_ourprice", "#priceblock_dealprice", "#priceblock_saleprice"):
        n = soup.select_one(sel)
        if n:
            return get_text(n)
    whole = soup.select_one(".a-price .a-price-whole")
    frac  = soup.select_one(".a-price .a-price-fraction")
    if whole:
        return (get_text(whole) + (("." + get_text(frac)) if frac else "")).replace("\u200f", "")
    m = re.search(r'[$€£]\s?[\d,]+(?:\.\d{2})?', soup.get_text(" ", strip=True))
    return m.group(0) if m else ""

# ---- NEW: rating extractor ----
def _first_float(text: str):
    if not text:
        return None
    m = re.search(r"\d+(?:\.\d+)?", text.replace(",", "."))
    return float(m.group(0)) if m else None

def extract_rating(soup):
    """
    Returns (rating_text, rating_number) e.g. ("4.4 out of 5", 4.4).
    Tries multiple Amazon patterns including the acrPopover block.
    """
    node = soup.select_one('[data-hook="rating-out-of-text"]')  # "4.4 out of 5"
    if node:
        txt = node.get_text(" ", strip=True)
        return txt, _first_float(txt)

    node = soup.select_one('#acrPopover span.a-size-base.a-color-base')  # your screenshot
    if node:
        txt = node.get_text(" ", strip=True)
        return f"{txt} out of 5", _first_float(txt)

    node = soup.select_one('#acrPopover .a-icon-alt, .a-icon-star .a-icon-alt')
    if node:
        txt = node.get_text(" ", strip=True)
        return txt, _first_float(txt)

    txt = soup.get_text(" ", strip=True)
    m = re.search(r"(\d+(?:\.\d+)?)\s*out of\s*5", txt, flags=re.I)
    if m:
        val = float(m.group(1))
        return f"{val} out of 5", val

    return "", None
# --------------------------------

def extract_specs_po_rows(soup):
    specs = {}
    for row in soup.select('tr[role="listitem"]'):
        label_node = row.select_one("td.a-span3, td:nth-of-type(1)")
        value_node = row.select_one("td.a-span9, td:nth-of-type(2)")
        label = get_text(label_node)
        value = get_text(value_node)
        if label and value:
            specs[label.rstrip(":")] = value

    for row in soup.select('[class^="po-"][role="listitem"]'):
        label_node = row.select_one(".a-text-bold")
        value_node = row.select_one(".a-span9, .a-span6, .a-span8")
        label = get_text(label_node) or get_text(row.select_one(".a-span3"))
        value = get_text(value_node)
        if label and value:
            specs[label.rstrip(":")] = value

    return specs

def extract_specs_tech_tables(soup):
    specs = {}
    for tr in soup.select("#productDetails_techSpec_section_1 tr, #productDetails_techSpec_section_2 tr"):
        th, td = tr.find("th"), tr.find("td")
        if th and td:
            k = get_text(th).rstrip(":")
            v = get_text(td)
            if k and v:
                specs[k] = v

    for tr in soup.select("#productDetails_detailBullets_sections1 tr"):
        th, td = tr.find("th"), tr.find("td")
        if th and td:
            k = get_text(th).rstrip(":")
            v = get_text(td)
            if k and v:
                specs[k] = v

    if not specs:
        for tr in soup.select("table tr"):
            th, td = tr.find("th"), tr.find("td")
            if th and td:
                k = get_text(th).rstrip(":")
                v = get_text(td)
                if k and v and len(k) < 50 and len(v) < 2000:
                    specs[k] = v

    return specs

def extract_all_specs(soup):
    specs = extract_specs_po_rows(soup)
    if specs:
        return specs
    return extract_specs_tech_tables(soup)

# ------------------ run & save ------------------
rows = []
for url in URLS:
    html = polite_get(url)
    if not html:
        print("Skipping:", url)
        continue

    soup = BeautifulSoup(html, "html.parser")
    name = extract_title(soup)
    price = extract_price(soup)
    rating_text, rating_num = extract_rating(soup)  # NEW
    specs = extract_all_specs(soup)

    row = {
        "Product Name": name,
        "Price": price,
        "Rating": rating_text,       # NEW
        "RatingNumber": rating_num,  # NEW
        "URL": url
    }
    row.update(specs)
    rows.append(row)

# JSON (debug)
with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
print("Wrote", JSON_PATH)

# WIDE CSV header = union of keys
all_cols = set()
for r in rows:
    all_cols.update(r.keys())
core = ["Product Name", "Price", "Rating", "RatingNumber", "URL"]  # NEW columns included
header = core + [c for c in sorted(all_cols) if c not in core]

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=header)
    w.writeheader()
    for r in rows:
        w.writerow(r)

print("✅ Wrote", CSV_PATH)