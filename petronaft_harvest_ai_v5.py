# -*- coding: utf-8 -*-
"""
Petro Naft — AI-ready site harvester (v5)

Key features
------------
- Generic crawler driven by a sitemap index URL (default: Petro Naft sitemap_index.xml).
- Asks interactively for:
    * Sitemap index URL
    * Output root directory
- Crawls the site via sitemap_index.xml
- Extracts clean, AI-ready content (Products, Articles, Pages, Collections)
- Product pages:
    * Structured sections (description, what is, other names, specs, applications, packing)
- Emits JSONL + CSV + platform metadata:
    * Internet Archive
    * Zenodo
    * Hugging Face
    * Kaggle
    * data.world
    * GitHub
    * Wikidata (QuickStatements)
- Maintains versioned corpora: <output_root>/petronaft_ai_ver1, ver2, ...
- Detects semantic changes via a corpus signature; only creates a new version when needed
- Deterministic record ordering for reproducible outputs
- Writes harvest_errors.log with per-URL error details
- Verbose, human-readable console output:
    * Shows each major step
    * Prints stats and what each file is for
    * Explains versioning and which old files each new one replaces
"""

from __future__ import annotations

import concurrent.futures as cf
import contextlib
import csv
import gzip
import hashlib
import io
import json
import os
import random
import re
import shutil
import string
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import venv
import zipfile

# ----------------------------
# Configuration (with defaults; will be overridden interactively)
# ----------------------------

SITE_ORIGIN = "https://www.petronaftco.com"
SITEMAP_INDEX_URL = f"{SITE_ORIGIN}/sitemap_index.xml"
ALLOWED_NETLOC = urlparse(SITE_ORIGIN).netloc

# Base for versioned output dirs, e.g. C:\Users\<user>\Downloads
OUTPUT_ROOT = Path(os.path.expanduser("~/Downloads"))

MAX_WORKERS = 10
REQUEST_TIMEOUT = 30
RETRY_TOTAL = 5
RETRY_BACKOFF = 0.5
RETRY_STATUS = (429, 500, 502, 503, 504)
LAUNCH_STAGGER = 0.03

USER_AGENT = (
    "PN-Harvester-AI/5.0 (+https://www.petronaftco.com/) "
    "PythonRequests"
)

DOWNLOAD_EXTS = (".pdf", ".doc", ".docx", ".xls", ".xlsx", ".csv", ".zip")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg")

# ----------------------------
# Company fixed info (verbatim, stable ground-truth for Petro Naft)
# ----------------------------

COMPANY_FIXED = {
    "parent_company_profile": {
        "Legal Name": "PETRO NAFT PETROKIMYA MADEN SANAYI TICARET LIMITED SIRKETI",
        "Trade Name": "PETRO NAFT",
        "Company Type": "Limited Company",
        "Registration Number": "507635",
        "MERSIS Number": "0729129928300001",
        "Tax/VAT Number": "7291299283",
        "Authorized Representative": "Vahit Ilhan",
        "Year of Establishment (Brand Origin)": "2011",
        "Registered Address of Parent Company": (
            "Ilkbahar Neighborhood, Guneypark Residential Complex, "
            "No. 3, Internal Door No. 211, Çankaya, Ankara, Turkey"
        ),
        "Contact Information": {
            "Tel/Fax/WhatsApp": "+90 552 693 1510",
            "Email": "info@petronaftco.com",
        },
        "Website": "https://www.petronaftco.com/",
        "Offices": {
            "Turkey Headquarters": (
                "Units 211 & 117, Tower 3, Altinoran-Sinpaş Complex, "
                "Turan Güneş Blvd, Ankara, Turkey"
            ),
            "UAE": "Unit 05-09, Burlington Tower, Business Bay, Dubai",
        },
        "Activities": {
            "Primary": (
                "NACE 19.20.17 — Manufacturing petroleum-based products "
                "like petroleum jelly/vaseline, paraffin wax, petroleum wax, "
                "petroleum coke, petroleum bitumen, etc."
            ),
            "Additional": "46.71.01; 46.90.01; 46.90.04",
        },
        "Certifications": [
            "ISO 9001:2015 — Certificate No: MTS-44332 — IAF Code: 10",
            "ISO 14001:2015 — Certificate No: MTS-44333 — IAF Code: 29",
        ],
        "Awards": [
            "Outstanding Global Exporter",
            "Leading R&D Unit in Industry and Mining",
            "Benchmark Production Unit in the Region",
        ],
        "Legal Verification and Chamber Membership Link": (
            "https://portal.atonet.org.tr/preview/"
            "67dbb266-9a08-407e-94f1-098211a42fa4/"
            "showPage?_tid=9905&dkod=96a41382600e4a209333edc1bca18c7d"
        ),
    },
    "payment_methods": (
        "SWIFT Wire Transfer (T/T); Irrevocable Letter of Credit (at sight or "
        "deferred); Confirmed LC; D/P; D/A; CAD; Open Account (subject to "
        "approval); Escrow; SBLC (as a payment guarantee)."
    ),
    "typical_export_origins": ["Turkey", "UAE", "China", "India"],
    "incoterms_2020": [
        "EXW",
        "FCA",
        "FAS",
        "FOB",
        "CFR",
        "CIF",
        "CPT",
        "CIP",
        "DAP",
        "DPU",
        "DDP",
    ],
    "inspection_companies": [
        "SGS",
        "Bureau Veritas",
        "Intertek",
        "Cotecna",
        "TÜV Rheinland",
        "TÜV SÜD",
        "Geo-Chem",
        "DNV",
        "Alex Stewart",
    ],
    "continents": [
        "Africa",
        "Asia",
        "Europe",
        "North America",
        "South America",
        "Oceania",
        "Antarctica",
    ],
    "un_m49_regions": [
        "Africa",
        "Americas",
        "Antarctica",
        "Asia",
        "Europe",
        "Oceania",
        "Latin America and the Caribbean",
        "Sub-Saharan Africa",
        "Eastern Africa",
        "Middle Africa",
        "Northern Africa",
        "Southern Africa",
        "Western Africa",
        "Caribbean",
        "Central America",
        "South America",
        "Northern America",
        "Central Asia",
        "Eastern Asia",
        "South-Eastern Asia",
        "Southern Asia",
        "Western Asia",
        "Eastern Europe",
        "Northern Europe",
        "Southern Europe",
        "Western Europe",
        "Australia and New Zealand",
        "Melanesia",
        "Micronesia",
        "Polynesia",
    ],
}

# ----------------------------
# Product section patterns
# ----------------------------

SECTION_PATTERNS: Dict[str, List[str]] = {
    "description": [
        r"\bdescription\b",
        r"\bdescription\s+of\b",
        r"\boverview\b",
        r"\bintroduction\b",
        r"\babout\b",
        r"\bproduct\s+overview\b",
    ],
    "what_is": [
        r"\bwhat\s+is\b",
        r"\bwhat\s+is\s+this\b",
        r"\bdefinition\b",
        r"\bdefinition\s+of\b",
    ],
    "other_names": [
        r"\bother\s+names?\b",
        r"\balso\s+known\s+as\b",
        r"\bsynonyms?\b",
        r"\balternative\s+names?\b",
        r"\baka\b",
    ],
    "specifications": [
        r"\bspecification(s)?\b",
        r"\btechnical\s+data\b",
        r"\btechnical\s+specs?\b",
        r"\bproduct\s+data\b",
        r"\btypical\s+properties\b",
        r"\bphysical\s+properties\b",
        r"\bchemical\s+properties\b",
        r"\bgrades?\b",
    ],
    "applications_uses": [
        r"\bapplications?\s+and\s+uses\b",
        r"\bapplications?\b",
        r"\buses\b",
        r"\busage\b",
        r"\bfields?\s+of\s+application\b",
        r"\bindustr(y|ies)\b",
        r"\bwhere\s+to\s+use\b",
    ],
    "packing": [
        r"\bpacking\b",
        r"\bpackaging\b",
        r"\bavailable\s+packing\b",
        r"\bpacking\s+type\b",
        r"\bpacking\s+and\s+storage\b",
        r"\bpackaging\s+and\s+storage\b",
        r"\bpacking\s+and\s+delivery\b",
    ],
}

# ----------------------------
# Utility helpers
# ----------------------------

def ensure_venv(path: Path = Path(".venv")) -> None:
    """
    Ensure a local virtual environment exists (created once; not auto-activated).
    """
    if path.exists():
        return
    venv.create(str(path), with_pip=True)


def build_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "*/*"})
    retry = Retry(
        total=RETRY_TOTAL,
        read=RETRY_TOTAL,
        connect=RETRY_TOTAL,
        backoff_factor=RETRY_BACKOFF,
        status_forcelist=RETRY_STATUS,
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=MAX_WORKERS,
        pool_maxsize=MAX_WORKERS,
    )
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


SESSION = build_session()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def gzip_bytes(raw: bytes) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(raw)
    return buf.getvalue()


def normalize_ws(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()


def is_internal(href: str) -> bool:
    u = urlparse(href)
    return (not u.netloc) or (u.netloc.lower() == ALLOWED_NETLOC.lower())


def make_abs(base: str, href: Optional[str]) -> str:
    return urljoin(base, href or "")


def rand_suffix(n: int = 6) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(n))


def configure_harvester() -> None:
    """
    Ask the user for sitemap index URL and output root, so the script
    can run generically on any site. Defaults are set for Petro Naft.
    """
    global SITE_ORIGIN, SITEMAP_INDEX_URL, ALLOWED_NETLOC, OUTPUT_ROOT

    print("=== Petro Naft AI Corpus Harvester (v5) ===")
    print("This script will:")
    print("  • Discover pages from a sitemap index URL")
    print("  • Fetch and clean content (products, articles, pages, collections)")
    print("  • Generate JSONL + CSV + metadata files ready for AI platforms")
    print()

    default_sitemap = SITEMAP_INDEX_URL
    sitemap_input = input(f"[1/2] Enter sitemap index URL "
                          f"(press Enter for default: {default_sitemap}): ").strip()
    if sitemap_input:
        SITEMAP_INDEX_URL = sitemap_input

    parsed = urlparse(SITEMAP_INDEX_URL)
    if not parsed.scheme or not parsed.netloc:
        print(f"ERROR: Invalid sitemap index URL: {SITEMAP_INDEX_URL}")
        raise SystemExit(1)

    SITE_ORIGIN = f"{parsed.scheme}://{parsed.netloc}"
    ALLOWED_NETLOC = parsed.netloc

    default_output = str(OUTPUT_ROOT)
    out_input = input(f"[2/2] Enter output root directory "
                      f"(press Enter for default: {default_output}): ").strip()
    if out_input:
        OUTPUT_ROOT = Path(out_input).expanduser().resolve()

    print()
    print("Configuration:")
    print(f"  SITE_ORIGIN       = {SITE_ORIGIN}")
    print(f"  SITEMAP_INDEX_URL = {SITEMAP_INDEX_URL}")
    print(f"  OUTPUT_ROOT       = {OUTPUT_ROOT}")
    print(f"  MAX_WORKERS       = {MAX_WORKERS}")
    print()


# ----------------------------
# HTTP + sitemap discovery
# ----------------------------

def safe_get(url: str) -> Tuple[int, str, str, Dict[str, str]]:
    try:
        r = SESSION.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        status = r.status_code
        final_url = r.url
        text = r.text or ""
        headers = {k.lower(): v for k, v in r.headers.items()}
        return status, final_url, text, headers
    except Exception:
        return 0, url, "", {}


def parse_xml_for_locs(xml_text: str) -> List[str]:
    from xml.etree import ElementTree as ET

    locs: List[str] = []
    try:
        root = ET.fromstring(xml_text.encode("utf-8"))
    except Exception:
        return re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", xml_text, flags=re.I)

    if root.tag.startswith("{"):
        uri = root.tag.split("}")[0].strip("{")
        ns = {"sm": uri}
        xpath = ".//sm:loc"
    else:
        ns = {}
        xpath = ".//loc"

    for loc in root.findall(xpath, ns):
        if loc.text:
            locs.append(loc.text.strip())
    return locs


def discover_urls() -> List[str]:
    status, _, text, _ = safe_get(SITEMAP_INDEX_URL)
    if status != 200 or not text:
        return []
    child_sitemaps = parse_xml_for_locs(text)
    urls: List[str] = []
    for sm in child_sitemaps:
        if urlparse(sm).netloc.lower() != ALLOWED_NETLOC.lower():
            continue
        s, _, t, _ = safe_get(sm)
        if s == 200 and t:
            urls.extend(parse_xml_for_locs(t))
        time.sleep(0.02)
    urls = [u for u in urls if urlparse(u).netloc.lower() == ALLOWED_NETLOC.lower()]
    urls = sorted(dict.fromkeys(urls).keys())
    return urls

# ----------------------------
# JSON-LD + head/meta extraction
# ----------------------------

def parse_jsonld(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for tag in soup.find_all("script", attrs={"type": re.compile(r"application/ld\+json", re.I)}):
        raw = tag.string or tag.get_text() or ""
        raw = raw.strip()
        if not raw:
            continue
        data = None
        with contextlib.suppress(Exception):
            data = json.loads(raw)
        if data is None:
            cleaned = raw.replace("\ufeff", "").strip()
            with contextlib.suppress(Exception):
                data = json.loads(cleaned)
        if data is None:
            continue
        objs: List[Dict[str, Any]] = []
        if isinstance(data, dict):
            if "@graph" in data and isinstance(data["@graph"], list):
                objs = [o for o in data["@graph"] if isinstance(o, dict)]
            else:
                objs = [data]
        elif isinstance(data, list):
            objs = [o for o in data if isinstance(o, dict)]
        out.extend(objs)
    return out


def flatten_types(jsonld: List[Dict[str, Any]]) -> List[str]:
    types: List[str] = []
    for obj in jsonld:
        t = obj.get("@type")
        if isinstance(t, list):
            types.extend([str(x) for x in t])
        elif isinstance(t, str):
            types.append(t)
    return [x.strip() for x in types if x]


def extract_head_meta(soup: BeautifulSoup, base_url: str) -> Dict[str, Any]:
    head: Dict[str, Any] = {}
    title_el = soup.find("title")
    head["title"] = normalize_ws(title_el.get_text()) if title_el else None
    canon_el = soup.find("link", attrs={"rel": re.compile(r"\bcanonical\b", re.I)})
    canonical = canon_el.get("href").strip() if canon_el and canon_el.get("href") else None
    if canonical:
        canonical = make_abs(base_url, canonical)
    head["canonical"] = canonical or base_url
    md = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
    head["meta_description"] = normalize_ws(md.get("content")) if md else None
    robots_meta = soup.find("meta", attrs={"name": re.compile(r"robots", re.I)})
    head["robots"] = normalize_ws(robots_meta.get("content")) if robots_meta else None
    og: Dict[str, str] = {}
    for m in soup.find_all("meta", attrs={"property": re.compile(r"^og:", re.I)}):
        k = m.get("property", "").lower()
        v = m.get("content")
        if k and v:
            og[k] = v
    head["og"] = og or None
    tw: Dict[str, str] = {}
    for m in soup.find_all("meta", attrs={"name": re.compile(r"^twitter:", re.I)}):
        k = m.get("name", "").lower()
        v = m.get("content")
        if k and v:
            tw[k] = v
    head["twitter"] = tw or None
    return head


def extract_dates(jsonld: List[Dict[str, Any]], soup: BeautifulSoup) -> Dict[str, Optional[str]]:
    pub = None
    mod = None
    for obj in jsonld:
        for k in ("datePublished", "dateCreated"):
            if not pub and isinstance(obj.get(k), str):
                pub = obj[k]
        for k in ("dateModified", "dateUpdated"):
            if not mod and isinstance(obj.get(k), str):
                mod = obj[k]
    if not pub:
        tag = soup.find("meta", attrs={"property": "article:published_time"})
        if tag and tag.get("content"):
            pub = tag["content"]
    if not mod:
        tag = soup.find("meta", attrs={"property": "article:modified_time"})
        if tag and tag.get("content"):
            mod = tag["content"]
    if not mod:
        tag = soup.find("meta", attrs={"property": "og:updated_time"})
        if tag and tag.get("content"):
            mod = tag["content"]
    return {"published": pub, "modified": mod}


def extract_breadcrumbs(jsonld: List[Dict[str, Any]], soup: BeautifulSoup, base_url: str) -> List[Dict[str, Optional[str]]]:
    for obj in jsonld:
        if obj.get("@type") == "BreadcrumbList" and isinstance(obj.get("itemListElement"), list):
            bcs: List[Dict[str, Optional[str]]] = []
            for item in obj["itemListElement"]:
                if not isinstance(item, dict):
                    continue
                entry = item.get("item") or {}
                if isinstance(entry, dict):
                    name = entry.get("name")
                    url = entry.get("@id")
                else:
                    name = None
                    url = None
                if name:
                    bcs.append({
                        "name": str(name),
                        "url": make_abs(base_url, url) if url else None,
                    })
            if bcs:
                return bcs
    trail: List[Dict[str, Optional[str]]] = []
    nav = soup.select_one("nav.breadcrumb, .breadcrumb, nav[aria-label*='breadcrumb' i]")
    if nav:
        for a in nav.find_all("a"):
            name = normalize_ws(a.get_text())
            href = a.get("href")
            if name:
                trail.append({
                    "name": name,
                    "url": make_abs(base_url, href) if href else None,
                })
    return trail

# ----------------------------
# Content container selection + cleanup
# ----------------------------

def pick_main(soup: BeautifulSoup):
    for sel in ["main[role='main']", "main", "article", "body"]:
        el = soup.select_one(sel)
        if el:
            return el
    return soup


def pick_article_container(soup: BeautifulSoup):
    selectors = [
        ".elementor-widget-theme-post-content",
        "article.post .entry-content",
        ".single-post .entry-content",
        ".entry-content",
    ]
    for sel in selectors:
        el = soup.select_one(sel)
        if el:
            return el
    return pick_main(soup)


def pick_content_container(soup: BeautifulSoup, page_type: str):
    if page_type in ("Article", "Product"):
        return pick_article_container(soup)
    return pick_main(soup)


def cleanup_content_container(container):
    selectors = [
        "#comments",
        ".comments-area",
        ".comment-form",
        "form",
        "nav",
        "header",
        "footer",
        ".site-footer",
        ".widget-area",
        ".entry-footer",
        ".wp-block-comments",
        ".pagination",
        "aside",
    ]
    for sel in selectors:
        for bad in container.select(sel):
            bad.decompose()

# ----------------------------
# Content extractors
# ----------------------------

def extract_tables(container) -> List[List[List[str]]]:
    tables: List[List[List[str]]] = []
    for t in container.find_all("table"):
        rows: List[List[str]] = []
        for tr in t.find_all("tr"):
            cells = [normalize_ws(td.get_text(" ")) for td in tr.find_all(["th", "td"])]
            if any(c.strip() for c in cells):
                rows.append(cells)
        if rows:
            tables.append(rows)
    return tables


def extract_headings(container) -> Tuple[Optional[str], List[Dict[str, str]]]:
    h1_el = container.find("h1")
    h1 = normalize_ws(h1_el.get_text()) if h1_el else None
    heads: List[Dict[str, str]] = []
    for tag in container.find_all(["h2", "h3"]):
        heads.append({"tag": tag.name, "text": normalize_ws(tag.get_text())})
    return h1, heads


def extract_images(container, base_url: str) -> List[Dict[str, str]]:
    imgs: List[Dict[str, str]] = []
    for im in container.find_all("img"):
        src = im.get("src") or im.get("data-src") or im.get("data-lazy-src")
        if not src:
            continue
        absu = make_abs(base_url, src)
        alt = normalize_ws(im.get("alt") or "")
        imgs.append({"src": absu, "alt": alt})
    return imgs


def extract_links(container, base_url: str) -> Tuple[List[str], List[str], List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    links_internal: List[str] = []
    links_external: List[str] = []
    links_internal_rich: List[Dict[str, str]] = []
    links_external_rich: List[Dict[str, str]] = []
    downloads: List[Dict[str, str]] = []
    for a in container.find_all("a", href=True):
        href = a["href"].strip()
        absu = make_abs(base_url, href)
        text = normalize_ws(a.get_text())
        lower = absu.lower()
        if any(lower.endswith(ext) for ext in DOWNLOAD_EXTS):
            downloads.append({"href": absu, "text": text})
            continue
        parsed = urlparse(absu)
        record = {"href": absu, "text": text}
        if parsed.netloc.lower() == ALLOWED_NETLOC.lower():
            links_internal.append(absu)
            links_internal_rich.append(record)
        else:
            links_external.append(absu)
            links_external_rich.append(record)
    links_internal = sorted(set(links_internal))
    links_external = sorted(set(links_external))
    return links_internal, links_external, links_internal_rich, links_external_rich, downloads


def extract_text_blocks(container) -> List[str]:
    blocks: List[str] = []
    for el in container.find_all(["p", "li", "blockquote"]):
        txt = normalize_ws(el.get_text(" "))
        if not txt:
            continue
        min_len = 20 if el.name == "li" else 40
        if len(txt) >= min_len:
            blocks.append(txt)
    out: List[str] = []
    seen: set[str] = set()
    for b in blocks:
        if b not in seen:
            out.append(b)
            seen.add(b)
    return out

# ----------------------------
# Product heading helpers
# ----------------------------

def normalize_heading_text(text: str) -> str:
    text = normalize_ws(text).lower()
    text = re.sub(r"[:?؛،\.\-]+$", "", text)
    return text.strip()


def classify_product_heading(text: str) -> Optional[str]:
    norm = normalize_heading_text(text)
    if not norm:
        return None
    for section, patterns in SECTION_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, norm):
                return section
    return None


def collect_section_content(start_heading: Tag) -> Tuple[List[str], List[List[List[str]]]]:
    texts: List[str] = []
    tables: List[List[List[str]]] = []

    for sib in start_heading.next_siblings:
        if isinstance(sib, Tag) and sib.name in ["h1", "h2", "h3", "h4"]:
            break

        if not isinstance(sib, Tag):
            continue

        if sib.name in ["p", "li", "blockquote"]:
            t = normalize_ws(sib.get_text(" "))
            if t:
                texts.append(t)

        if sib.name in ["ul", "ol"]:
            for li in sib.find_all("li"):
                t = normalize_ws(li.get_text(" "))
                if t:
                    texts.append(t)

        if sib.name == "table":
            rows: List[List[str]] = []
            for tr in sib.find_all("tr"):
                cells = [normalize_ws(td.get_text(" ")) for td in tr.find_all(["th", "td"])]
                if any(cells):
                    rows.append(cells)
            if rows:
                tables.append(rows)

    return texts, tables

# ----------------------------
# Classification
# ----------------------------

def classify_page(jsonld: List[Dict[str, Any]], soup: BeautifulSoup, url: str) -> str:
    # Force /products/ page as WebPage (collection of product links)
    if url.rstrip("/").lower() == f"{SITE_ORIGIN}/products".rstrip("/").lower():
        return "WebPage"

    types = {t.lower() for t in flatten_types(jsonld)}
    if "product" in types:
        return "Product"
    if any(t in types for t in ("article", "newsarticle", "blogposting")):
        return "Article"
    if any(t in types for t in ("collectionpage", "itemlist")):
        return "CollectionPage"
    if soup.select("nav.pagination, .pagination, a.next, a.prev"):
        cards = soup.select(
            "article, .post-card, .archive-item, .product-card, "
            ".entry, .card, .elementor-post, .elementor-grid-item"
        )
        if len(cards) >= 3:
            return "CollectionPage"
    return "WebPage"

# ----------------------------
# Envelope + entity builders
# ----------------------------

def compute_doc_id(url: str, title: Optional[str], text_blocks: List[str]) -> str:
    base = (url or "") + "|" + (title or "") + "|" + "\n".join(text_blocks or [])
    return "sha256:" + sha256_hex(base.encode("utf-8"))


def build_common_envelope(
    requested_url: str,
    final_url: str,
    status: int,
    head: Dict[str, Any],
    jsonld: List[Dict[str, Any]],
    soup: BeautifulSoup,
    page_type: str,
) -> Dict[str, Any]:
    canonical = head.get("canonical") or final_url
    dates = extract_dates(jsonld, soup)
    breadcrumbs = extract_breadcrumbs(jsonld, soup, final_url)
    container = pick_content_container(soup, page_type)
    cleanup_content_container(container)
    h1, headings = extract_headings(container)
    text_blocks = extract_text_blocks(container)
    if page_type == "Article" and not text_blocks:
        raw = normalize_ws(container.get_text(" "))
        if raw:
            text_blocks = [raw]
    tables = extract_tables(container)
    images = extract_images(container, final_url)
    links_internal, links_external, links_internal_rich, links_external_rich, downloads = extract_links(
        container, final_url
    )
    title_for_id = head.get("title") or h1 or ""
    doc_id = compute_doc_id(canonical, title_for_id, text_blocks)

    # Only attach Petro Naft fixed profile for the Petro Naft domain
    company_block: Optional[Dict[str, Any]]
    if urlparse(SITE_ORIGIN).netloc.lower() == "www.petronaftco.com" or \
       urlparse(SITE_ORIGIN).netloc.lower() == "petronaftco.com":
        company_block = COMPANY_FIXED
    else:
        company_block = None

    env: Dict[str, Any] = {
        "doc_id": doc_id,
        "url": final_url,
        "requested_url": requested_url,
        "http": {"status": status},
        "document_identifier": {
            "title": head.get("title") or h1,
            "canonical": canonical,
            "document_code": None,
            "page_type": page_type,
        },
        "page_meta": {
            "meta_description": head.get("meta_description"),
            "robots": head.get("robots"),
            "og": head.get("og"),
            "twitter": head.get("twitter"),
            "breadcrumbs": breadcrumbs,
            "dates": dates,
        },
        "jsonld": jsonld or [],
        "content": {
            "h1": h1,
            "headings": headings,
            "text_blocks": text_blocks,
            "tables": tables,
            "images": images,
            "downloads": downloads,
            "links_internal": links_internal,
            "links_external": links_external,
            "links_internal_rich": links_internal_rich,
            "links_external_rich": links_external_rich,
        },
        "company_fixed": company_block,
    }
    return env


def build_article(env: Dict[str, Any], soup: BeautifulSoup) -> Dict[str, Any]:
    headline = env["content"]["h1"] or env["document_identifier"]["title"]
    author: Optional[str] = None
    tags: set[str] = set()
    categories: set[str] = set()
    featured: Optional[Dict[str, Optional[str]]] = None

    for obj in env.get("jsonld", []):
        t = str(obj.get("@type", "")).lower()
        if t in ("article", "newsarticle", "blogposting"):
            auth = obj.get("author")
            if isinstance(auth, dict) and isinstance(auth.get("name"), str):
                if not author:
                    author = auth["name"]
            elif isinstance(auth, list):
                for a in auth:
                    if isinstance(a, dict) and isinstance(a.get("name"), str):
                        if not author:
                            author = a["name"]
                        break
            kw = obj.get("keywords")
            if isinstance(kw, str):
                for part in kw.split(","):
                    part = part.strip()
                    if part:
                        tags.add(part)
            elif isinstance(kw, list):
                for part in kw:
                    if isinstance(part, str):
                        p = part.strip()
                        if p:
                            tags.add(p)
            sec = obj.get("articleSection")
            if isinstance(sec, str):
                sec = sec.strip()
                if sec:
                    categories.add(sec)
            elif isinstance(sec, list):
                for s in sec:
                    if isinstance(s, str):
                        s = s.strip()
                        if s:
                            categories.add(s)
            img = obj.get("image")
            if not featured and isinstance(img, dict) and isinstance(img.get("url"), str):
                featured = {"src": img["url"], "alt": None}
            elif not featured and isinstance(img, str):
                featured = {"src": img, "alt": None}
            elif not featured and isinstance(img, list):
                for item in img:
                    if isinstance(item, dict) and isinstance(item.get("url"), str):
                        featured = {"src": item["url"], "alt": None}
                        break
                    if isinstance(item, str):
                        featured = {"src": item, "alt": None}
                        break

    for m in soup.find_all("meta", attrs={"property": "article:tag"}):
        val = (m.get("content") or "").strip()
        if val:
            tags.add(val)
    sec_meta = soup.find("meta", attrs={"property": "article:section"})
    if sec_meta and sec_meta.get("content"):
        categories.add(sec_meta["content"].strip())

    if not featured:
        og = env.get("page_meta", {}).get("og") or {}
        og_img = None
        for key in ("og:image", "og:image:url"):
            if key in og and og[key]:
                og_img = og[key]
                break
        if og_img:
            featured = {"src": og_img, "alt": None}

    if not featured and env["content"]["images"]:
        for im in env["content"]["images"]:
            src = im.get("src") or ""
            alt = im.get("alt") or ""
            if not src or src.startswith("data:"):
                continue
            alt_low = alt.lower()
            if "logo" in alt_low or "avatar" in alt_low:
                continue
            featured = {"src": src, "alt": alt or None}
            break

    env["article"] = {
        "headline": headline,
        "author": author,
        "categories": sorted(categories),
        "tags": sorted(tags),
        "featured_image": featured,
    }
    return env


def build_product(env: Dict[str, Any], soup: BeautifulSoup) -> Dict[str, Any]:
    name: Optional[str] = None
    for obj in env.get("jsonld", []):
        if str(obj.get("@type", "")).lower() == "product" and isinstance(obj.get("name"), str):
            name = obj["name"]
            break
    name = name or env["content"]["h1"] or env["document_identifier"]["title"]

    container = pick_content_container(soup, "Product")
    cleanup_content_container(container)

    section_texts: Dict[str, List[str]] = {
        "description": [],
        "what_is": [],
        "other_names": [],
        "specifications": [],
        "applications_uses": [],
        "packing": [],
    }
    specs_tables: List[List[List[str]]] = []

    for h in container.find_all(["h2", "h3", "h4"]):
        heading_text = normalize_ws(h.get_text(" "))
        if not heading_text:
            continue
        key = classify_product_heading(heading_text)
        if not key:
            continue
        if section_texts.get(key):
            continue

        texts, tables = collect_section_content(h)
        if texts:
            section_texts[key].extend(texts)
        if key == "specifications" and tables:
            specs_tables.extend(tables)

    description_text = "\n\n".join(section_texts["description"]) or None
    what_is_text = "\n\n".join(section_texts["what_is"]) or None
    other_names_text = "\n\n".join(section_texts["other_names"]) or None
    specs_text = "\n\n".join(section_texts["specifications"]) or None
    applications_text = "\n\n".join(section_texts["applications_uses"]) or None
    packing_text = "\n\n".join(section_texts["packing"]) or None

    grades_specs = specs_tables or None
    uses_applications = applications_text or None

    env["product"] = {
        "name": name,
        "sections": {
            "description_of_product": description_text,
            "what_is_product": what_is_text,
            "other_names": other_names_text,
            "specifications": {
                "text": specs_text,
                "tables": grades_specs,
            },
            "applications_and_uses": applications_text,
            "packing": packing_text,
            # Backward-compatible aliases:
            "uses_applications": uses_applications,
            "grades_specs": grades_specs,
            "faqs": [],
        },
    }
    return env


def build_collection(env: Dict[str, Any], soup: BeautifulSoup) -> Dict[str, Any]:
    container = pick_content_container(soup, "CollectionPage")
    name = env["content"]["h1"] or env["document_identifier"]["title"]
    items: List[Dict[str, Optional[str]]] = []
    for card in container.select(
        "article, .post-card, .archive-item, .product-card, .entry, .card, "
        ".elementor-post, .elementor-grid-item"
    ):
        a = card.find("a", href=True)
        if not a:
            continue
        href = make_abs(env["url"], a["href"])
        title = normalize_ws(a.get_text()) or normalize_ws(card.get_text())
        excerpt_el = card.find(class_=re.compile(r"(excerpt|summary|entry-summary)", re.I))
        excerpt = normalize_ws(excerpt_el.get_text()) if excerpt_el else None
        img = card.find("img")
        thumb = make_abs(env["url"], img["src"]) if img and img.get("src") else None
        items.append({
            "title": title,
            "url": href,
            "excerpt": excerpt,
            "thumb": thumb,
            "date": None,
        })
    env["collection"] = {
        "name": name,
        "description": None,
        "pagination": {"page": 1, "total_pages": None},
        "items": items,
    }
    return env

# ----------------------------
# Page worker
# ----------------------------

@dataclass
class PageResult:
    ok: bool
    url: str
    final_url: str
    status: int
    record: Optional[Dict[str, Any]]
    error: Optional[str] = None


def process_url(url: str) -> PageResult:
    try:
        status, final_url, html, _ = safe_get(url)
        if status != 200 or not html:
            return PageResult(False, url, final_url, status, None, f"HTTP {status}")
        soup = BeautifulSoup(html, "lxml")
        jsonld = parse_jsonld(soup)
        head = extract_head_meta(soup, final_url)
        page_type = classify_page(jsonld, soup, final_url)
        env = build_common_envelope(url, final_url, status, head, jsonld, soup, page_type)
        if page_type == "Product":
            env = build_product(env, soup)
        elif page_type == "Article":
            env = build_article(env, soup)
        elif page_type == "CollectionPage":
            env = build_collection(env, soup)
        return PageResult(True, url, final_url, status, env, None)
    except Exception as e:
        return PageResult(False, url, url, 0, None, f"{type(e).__name__}: {e}")

# ----------------------------
# File writers
# ----------------------------

def write_ndjson(path: Path, records: Iterable[Dict[str, Any]]) -> Tuple[int, str]:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    gz_path = path.with_suffix(path.suffix + ".gz")
    with path.open("rb") as src:
        gz_data = gzip_bytes(src.read())
    with gz_path.open("wb") as gzf:
        gzf.write(gz_data)
    sha = sha256_hex(gz_data)
    return count, sha


def write_csv_site_pages(path: Path, records: List[Dict[str, Any]]) -> int:
    fields = [
        "doc_id",
        "url",
        "canonical",
        "page_type",
        "title",
        "h1",
        "meta_description",
        "published",
        "modified",
        "breadcrumbs",
        "num_text_blocks",
        "num_internal_links",
        "num_external_links",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rec in records:
            meta = rec["page_meta"]
            content = rec["content"]
            row = {
                "doc_id": rec["doc_id"],
                "url": rec["url"],
                "canonical": rec["document_identifier"]["canonical"],
                "page_type": rec["document_identifier"]["page_type"],
                "title": rec["document_identifier"]["title"],
                "h1": content["h1"],
                "meta_description": meta["meta_description"],
                "published": (meta.get("dates") or {}).get("published"),
                "modified": (meta.get("dates") or {}).get("modified"),
                "breadcrumbs": " > ".join(
                    [b["name"] for b in meta.get("breadcrumbs") or [] if b.get("name")]
                ),
                "num_text_blocks": len(content.get("text_blocks") or []),
                "num_internal_links": len(content.get("links_internal") or []),
                "num_external_links": len(content.get("links_external") or []),
            }
            w.writerow(row)
    return len(records)


def write_csv_products(path: Path, records: List[Dict[str, Any]]) -> int:
    fields = [
        "doc_id",
        "url",
        "canonical",
        "title",
        "h1",
        "product_name",
        "uses_applications",
        "has_specs_tables",
        "num_text_blocks",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rec in records:
            prod = rec.get("product") or {}
            sections = prod.get("sections") or {}
            content = rec["content"]
            row = {
                "doc_id": rec["doc_id"],
                "url": rec["url"],
                "canonical": rec["document_identifier"]["canonical"],
                "title": rec["document_identifier"]["title"],
                "h1": content["h1"],
                "product_name": prod.get("name"),
                "uses_applications": sections.get("uses_applications"),
                "has_specs_tables": bool(sections.get("grades_specs")),
                "num_text_blocks": len(content.get("text_blocks") or []),
            }
            w.writerow(row)
    return len(records)


def write_csv_articles(path: Path, records: List[Dict[str, Any]]) -> int:
    fields = [
        "doc_id",
        "url",
        "canonical",
        "title",
        "h1",
        "headline",
        "author",
        "published",
        "modified",
        "num_text_blocks",
        "num_tags",
        "num_categories",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rec in records:
            meta = rec["page_meta"]
            art = rec.get("article") or {}
            content = rec["content"]
            row = {
                "doc_id": rec["doc_id"],
                "url": rec["url"],
                "canonical": rec["document_identifier"]["canonical"],
                "title": rec["document_identifier"]["title"],
                "h1": content["h1"],
                "headline": art.get("headline"),
                "author": art.get("author"),
                "published": (meta.get("dates") or {}).get("published"),
                "modified": (meta.get("dates") or {}).get("modified"),
                "num_text_blocks": len(content.get("text_blocks") or []),
                "num_tags": len(art.get("tags") or []),
                "num_categories": len(art.get("categories") or []),
            }
            w.writerow(row)
    return len(records)

# ----------------------------
# Deterministic ordering helper
# ----------------------------

def record_sort_key(rec: Dict[str, Any]) -> Tuple[str, str, str]:
    di = rec.get("document_identifier") or {}
    canon = di.get("canonical") or rec.get("url") or ""
    page_type = di.get("page_type") or ""
    doc_id = rec.get("doc_id") or ""
    return (canon, page_type, doc_id)

# ----------------------------
# QA metrics
# ----------------------------

def compute_quality_metrics(
    all_records: List[Dict[str, Any]],
    products: List[Dict[str, Any]],
    articles: List[Dict[str, Any]],
    collections: List[Dict[str, Any]],
) -> Dict[str, Any]:
    qm: Dict[str, Any] = {}

    art_empty_text = 0
    art_lt5_text = 0
    art_empty_tags = 0
    art_empty_cats = 0
    for rec in articles:
        tb = rec["content"].get("text_blocks") or []
        if not tb:
            art_empty_text += 1
        if len(tb) < 5:
            art_lt5_text += 1
        art = rec.get("article") or {}
        if not (art.get("tags") or []):
            art_empty_tags += 1
        if not (art.get("categories") or []):
            art_empty_cats += 1

    prod_no_specs = 0
    for rec in products:
        prod = rec.get("product") or {}
        sections = prod.get("sections") or {}
        if not sections.get("grades_specs"):
            prod_no_specs += 1

    coll_empty_items = 0
    for rec in collections:
        coll = rec.get("collection") or {}
        items = coll.get("items") or []
        if not items:
            coll_empty_items += 1

    qm["articles"] = {
        "total": len(articles),
        "num_empty_text_blocks": art_empty_text,
        "num_less_than_5_blocks": art_lt5_text,
        "num_empty_tags": art_empty_tags,
        "num_empty_categories": art_empty_cats,
    }
    qm["products"] = {
        "total": len(products),
        "num_without_specs_tables": prod_no_specs,
    }
    qm["collections"] = {
        "total": len(collections),
        "num_empty_items": coll_empty_items,
    }
    qm["site"] = {
        "total_records": len(all_records),
    }
    return qm

# ----------------------------
# Platform metadata writers
# ----------------------------

def write_internet_archive_metadata(path: Path, version: str) -> None:
    meta = {
        "title": f"Petro Naft AI Corpus {version}",
        "description": (
            "Authoritative, first-party dataset of Petro Naft website content "
            "for AI training, retrieval-augmented generation (RAG), and search. "
            f"Includes structured products, articles, pages, and collections "
            f"harvested exclusively from {SITE_ORIGIN}."
        ),
        "creator": "Petro Naft",
        "mediatype": "data",
        "language": "eng",
        "collection": "opensource",
        "subject": [
            "Petro Naft",
            "petroleum products",
            "bitumen",
            "gilsonite",
            "paraffin wax",
            "industrial chemicals",
            "AI dataset",
            "RAG",
        ],
        "source": SITE_ORIGIN,
        "rights": "CC BY 4.0 (Attribution required).",
        "external_identifier": f"petronaft-ai-corpus-{version}",
    }
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def write_zenodo_metadata(path: Path, version: str) -> None:
    meta = {
        "upload_type": "dataset",
        "publication_date": datetime.utcnow().date().isoformat(),
        "title": f"Petro Naft AI Corpus ({version})",
        "creators": [
            {"name": "Petro Naft", "affiliation": "PETRO NAFT PETROKIMYA MADEN SANAYI TICARET LTD. STI."}
        ],
        "description": (
            "AI-ready export of the Petro Naft website, including structured products, "
            "technical articles, and web pages. Designed for use in search, RAG, "
            "and domain-specific AI assistants. All content is harvested from the "
            f"official first-party source {SITE_ORIGIN}."
        ),
        "access_right": "open",
        "license": "CC-BY-4.0",
        "language": "eng",
        "keywords": [
            "Petro Naft",
            "petroleum",
            "bitumen",
            "gilsonite",
            "paraffin wax",
            "industrial chemicals",
            "AI",
            "dataset",
            "RAG",
        ],
        "related_identifiers": [
            {
                "identifier": SITE_ORIGIN,
                "relation": "isDerivedFrom",
                "resource_type": "dataset",
            }
        ],
    }
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def write_hf_dataset_card(path: Path, version: str) -> None:
    text = (
        "---\n"
        "pretty_name: Petro Naft AI Corpus\n"
        f"version: {version}\n"
        "license: cc-by-4.0\n"
        "language:\n"
        "  - en\n"
        "tags:\n"
        "  - petro-naft\n"
        "  - petroleum-products\n"
        "  - bitumen\n"
        "  - gilsonite\n"
        "  - paraffin-wax\n"
        "  - industrial-chemicals\n"
        "  - ai-training\n"
        "  - rag\n"
        "task_categories:\n"
        "  - question-answering\n"
        "  - information-retrieval\n"
        "size_categories:\n"
        "  - 1K<n<10K\n"
        "---\n"
        "\n"
        f"# Petro Naft AI Corpus ({version})\n"
        "\n"
        "This dataset is a structured export of the official Petro Naft website,\n"
        "designed specifically for AI training, retrieval-augmented generation (RAG),\n"
        "and domain-specific assistants.\n"
        "\n"
        "All content is first-party and harvested exclusively from:\n"
        f"<{SITE_ORIGIN}>.\n"
        "\n"
        "## Files\n"
        "\n"
        "The dataset typically includes:\n"
        "\n"
        "- `site_pages.jsonl.gz` – all crawled pages (products, articles, pages, collections)\n"
        "- `products.jsonl.gz` – records with `page_type = \"Product\"`\n"
        "- `articles.jsonl.gz` – records with `page_type = \"Article\"`\n"
        "- `pages.jsonl.gz` – general `WebPage` records\n"
        "- `collections.jsonl.gz` – `CollectionPage` records (archives, categories)\n"
        "- CSV mirrors for BI / SQL:\n"
        "  - `petronaft_site_pages.csv`\n"
        "  - `petronaft_products.csv`\n"
        "  - `petronaft_articles.csv`\n"
        "\n"
        "Each JSONL record has:\n"
        "\n"
        "- `doc_id`: stable SHA-256 identifier\n"
        "- `url`, `requested_url`, `http.status`\n"
        "- `document_identifier`: title, canonical, page_type\n"
        "- `page_meta`: meta description, robots, OpenGraph, Twitter, breadcrumbs, dates\n"
        "- `content`:\n"
        "  - `h1`, `headings`\n"
        "  - `text_blocks`\n"
        "  - `tables`\n"
        "  - `images`\n"
        "  - `downloads`\n"
        "  - `links_internal`, `links_external`\n"
        "  - `links_internal_rich`, `links_external_rich`\n"
        "- `company_fixed`: Petro Naft corporate facts (legal name, certificates, Incoterms, etc.)\n"
        "\n"
        "## Usage\n"
        "\n"
        "You can load the JSONL files with the `datasets` library:\n"
        "\n"
        "```python\n"
        "from datasets import load_dataset\n"
        "\n"
        "ds = load_dataset(\n"
        "    \"json\",\n"
        "    data_files={\n"
        "        \"site_pages\": \"site_pages.jsonl.gz\",\n"
        "        \"products\": \"products.jsonl.gz\",\n"
        "        \"articles\": \"articles.jsonl.gz\",\n"
        "        \"pages\": \"pages.jsonl.gz\",\n"
        "        \"collections\": \"collections.jsonl.gz\",\n"
        "    }\n"
        ")\n"
        "```\n"
        "\n"
        "Use the `content.text_blocks` field for embeddings / RAG, and keep\n"
        "`company_fixed` available as a high-trust reference for Petro Naft’s\n"
        "corporate identity and trade capabilities.\n"
        "\n"
        "## License\n"
        "\n"
        "The dataset is provided under **CC BY 4.0**. Proper attribution to **Petro Naft**\n"
        "is required when using this corpus in research, products, or AI systems.\n"
    )
    path.write_text(text, encoding="utf-8")


def write_kaggle_metadata(path: Path, version: str) -> None:
    meta = {
        "title": f"Petro Naft AI Corpus {version}",
        "id": f"username/petronaft-ai-corpus-{version}",  # replace "username"
        "licenses": [{"name": "CC-BY-4.0"}],
        "description": (
            "Structured, AI-ready corpus of the Petro Naft website, suitable for "
            "search, RAG, and industrial domain assistants. Includes products, "
            "articles, and general pages with rich metadata."
        ),
        "keywords": [
            "Petro Naft",
            "petroleum",
            "bitumen",
            "gilsonite",
            "paraffin wax",
            "industrial chemicals",
            "AI",
            "dataset",
        ],
        "resources": [
            {"path": "site_pages.jsonl.gz", "description": "All harvested pages"},
            {"path": "products.jsonl.gz", "description": "Product pages"},
            {"path": "articles.jsonl.gz", "description": "Articles and news"},
            {"path": "pages.jsonl.gz", "description": "General web pages"},
            {"path": "collections.jsonl.gz", "description": "Collections / archives"},
            {"path": "petronaft_site_pages.csv", "description": "CSV summary of all pages"},
            {"path": "petronaft_products.csv", "description": "CSV summary of products"},
            {"path": "petronaft_articles.csv", "description": "CSV summary of articles"},
        ],
    }
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def write_dataworld_readme(path: Path, version: str) -> None:
    text = (
        f"# Petro Naft AI Corpus ({version})\n"
        "\n"
        "This dataset is an AI-ready export of the official Petro Naft website\n"
        f"({SITE_ORIGIN}).\n"
        "\n"
        "It includes:\n"
        "\n"
        "- `site_pages.jsonl.gz` – all pages (products, articles, general pages, collections)\n"
        "- `products.jsonl.gz` – product pages\n"
        "- `articles.jsonl.gz` – informative articles and news\n"
        "- `pages.jsonl.gz` – general pages\n"
        "- `collections.jsonl.gz` – archive / category pages\n"
        "- CSV mirrors:\n"
        "  - `petronaft_site_pages.csv`\n"
        "  - `petronaft_products.csv`\n"
        "  - `petronaft_articles.csv`\n"
        "\n"
        "Each record includes headings, cleaned text blocks, tables, images, and links,\n"
        "alongside Petro Naft’s fixed corporate profile, making this suitable for:\n"
        "\n"
        "- Knowledge graph construction\n"
        "- RAG / semantic search\n"
        "- Analytics and reporting\n"
        "\n"
        f"**Source:** <{SITE_ORIGIN}>\n"
        "\n"
        "**License:** CC BY 4.0 (attribution required).\n"
    )
    path.write_text(text, encoding="utf-8")


def write_github_readme(path: Path, version: str) -> None:
    text = (
        f"# Petro Naft AI Corpus ({version})\n"
        "\n"
        "This repository contains an AI-ready export of the **Petro Naft** website,\n"
        "intended as the authoritative, first-party dataset for:\n"
        "\n"
        "- Search and retrieval-augmented generation (RAG)\n"
        "- Domain-specific assistants\n"
        "- Evaluation of LLMs on petroleum / industrial topics\n"
        "\n"
        f"All data is harvested exclusively from <{SITE_ORIGIN}>,\n"
        "and every record carries Petro Naft’s fixed corporate profile for consistent\n"
        "grounding across AI systems.\n"
        "\n"
        "## Contents\n"
        "\n"
        "- `site_pages.jsonl(.gz)` – all pages\n"
        "- `products.jsonl(.gz)` – product pages\n"
        "- `articles.jsonl(.gz)` – articles and news\n"
        "- `pages.jsonl(.gz)` – general pages\n"
        "- `collections.jsonl(.gz)` – collections / archives\n"
        "\n"
        "CSV mirrors:\n"
        "\n"
        "- `petronaft_site_pages.csv`\n"
        "- `petronaft_products.csv`\n"
        "- `petronaft_articles.csv`\n"
        "\n"
        "Platform metadata / templates:\n"
        "\n"
        "- `internet_archive_metadata.json`\n"
        "- `zenodo_metadata.json`\n"
        "- `huggingface_dataset_card.md`\n"
        "- `kaggle_dataset-metadata.json`\n"
        "- `dataworld_readme.md`\n"
        "- `wikidata_quickstatements.tsv`\n"
        "\n"
        "## Schema overview\n"
        "\n"
        "Each JSONL record includes:\n"
        "\n"
        "- `doc_id`\n"
        "- `url`, `requested_url`, `http.status`\n"
        "- `document_identifier` (title, canonical, page_type)\n"
        "- `page_meta` (meta description, robots, OG, Twitter, breadcrumbs, dates)\n"
        "- `content`:\n"
        "  - `h1`, `headings`\n"
        "  - `text_blocks`\n"
        "  - `tables`\n"
        "  - `images`\n"
        "  - `downloads`\n"
        "  - `links_internal`, `links_external`\n"
        "  - `links_internal_rich`, `links_external_rich`\n"
        "- `company_fixed` (legal info, certifications, Incoterms, etc.)\n"
        "\n"
        "## License\n"
        "\n"
        "Content is provided under **CC BY 4.0**. Please attribute **Petro Naft**\n"
        "when using this corpus in any downstream system.\n"
    )
    path.write_text(text, encoding="utf-8")


def write_wikidata_quickstatements(path: Path) -> None:
    text = (
        "CREATE\n"
        "LAST\tP31\tQ4830453\n"
        "LAST\tP17\tQ43\n"
        f"LAST\tP856\t\"{SITE_ORIGIN}\"\n"
        "LAST\tP571\t+2011-01-01T00:00:00Z/9\n"
        "LAST\tL.en\t\"Petro Naft\"\n"
        "LAST\tD.en\t\"Petro Naft is a manufacturer and global supplier of petroleum-based industrial and commercial products.\"\n"
    )
    path.write_text(text, encoding="utf-8")

# ----------------------------
# Versioning helpers
# ----------------------------

def find_latest_version_dir(base: Path) -> Tuple[Optional[int], Optional[Path]]:
    latest_n: Optional[int] = None
    latest_path: Optional[Path] = None
    if not base.exists():
        return None, None
    for child in base.iterdir():
        if not child.is_dir():
            continue
        m = re.match(r"petronaft_ai_ver(\d+)$", child.name)
        if not m:
            continue
        n = int(m.group(1))
        if latest_n is None or n > latest_n:
            latest_n = n
            latest_path = child
    return latest_n, latest_path


def load_previous_corpus_signature(prev_dir: Optional[Path]) -> Optional[str]:
    if not prev_dir:
        return None
    summary_path = prev_dir / "harvest_summary.json"
    if not summary_path.exists():
        return None
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    cd = data.get("change_detection") or {}
    return cd.get("corpus_signature")


def compute_corpus_signature(site_pages_path: Path) -> str:
    hasher = hashlib.sha256()
    records: List[Dict[str, Any]] = []
    with site_pages_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append(rec)

    def sort_key(rec: Dict[str, Any]) -> Tuple[str, str]:
        canonical = (rec.get("document_identifier") or {}).get("canonical") or rec.get("url") or ""
        doc_id = rec.get("doc_id") or ""
        return (canonical, doc_id)

    records.sort(key=sort_key)
    for rec in records:
        dumped = json.dumps(rec, sort_keys=True, separators=(",", ":")).encode("utf-8")
        hasher.update(dumped)
        hasher.update(b"\n")
    return hasher.hexdigest()


def compare_and_copy_changed_files(staging_dir: Path, prev_dir: Optional[Path], version_dir: Path) -> Tuple[List[str], List[str]]:
    changed: List[str] = []
    unchanged: List[str] = []
    version_dir.mkdir(parents=True, exist_ok=True)
    for root, _, files in os.walk(staging_dir):
        rel_root = Path(root).relative_to(staging_dir)
        for fname in files:
            src_path = Path(root) / fname
            rel_path = rel_root / fname
            dst_path = version_dir / rel_path
            if rel_root != Path("."):
                (version_dir / rel_root).mkdir(parents=True, exist_ok=True)
            prev_path = prev_dir / rel_path if prev_dir else None
            needs_copy = True
            if prev_path and prev_path.exists():
                with src_path.open("rb") as f1, prev_path.open("rb") as f2:
                    if f1.read() == f2.read():
                        needs_copy = False
            if needs_copy:
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
                changed.append(str(rel_path).replace("\\", "/"))
            else:
                unchanged.append(str(rel_path).replace("\\", "/"))
    return changed, unchanged

# ----------------------------
# Main harvest routine
# ----------------------------

def main() -> int:
    configure_harvester()
    print("Step 1: Ensuring local virtual environment (optional)...")
    ensure_venv()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"  Output root directory ready: {OUTPUT_ROOT}")
    print()

    prev_version_num, prev_version_dir = find_latest_version_dir(OUTPUT_ROOT)
    prev_signature = load_previous_corpus_signature(prev_version_dir)
    if prev_version_num is not None:
        print(f"Detected previous version: petronaft_ai_ver{prev_version_num}")
    else:
        print("No previous version found; this will be version v1.")
    print()

    tmp_dir_name = f"petronaft_ai_tmp_{rand_suffix()}_{rand_suffix()}"
    staging_dir = OUTPUT_ROOT / tmp_dir_name
    staging_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created staging directory: {staging_dir}")
    print()

    print(f"Step 2: Discovering URLs from sitemap index: {SITEMAP_INDEX_URL}")
    urls = discover_urls()
    if not urls:
        print("ERROR: No URLs discovered from sitemap index. Aborting.")
        shutil.rmtree(staging_dir, ignore_errors=True)
        return 2
    print(f"  Discovered {len(urls)} URLs to harvest.")
    print()

    print(f"Step 3: Fetching and parsing {len(urls)} pages (max_workers={MAX_WORKERS})...")
    results: List[PageResult] = []
    total_urls = len(urls)
    processed = 0

    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = []
        for u in urls:
            futs.append(ex.submit(process_url, u))
            time.sleep(LAUNCH_STAGGER)
        for fut in cf.as_completed(futs):
            res = fut.result()
            results.append(res)
            processed += 1
            if processed % 20 == 0 or processed == total_urls:
                print(f"  Processed {processed}/{total_urls} pages...")

    ok_records = [r.record for r in results if r.ok and r.record]
    errors = [r for r in results if not r.ok]
    print(f"  Harvested OK: {len(ok_records)} pages")
    print(f"  Harvest errors: {len(errors)} pages")
    print()

    # Error log
    if errors:
        err_path = staging_dir / "harvest_errors.log"
        with err_path.open("w", encoding="utf-8") as f:
            for e in errors:
                f.write(
                    f"{e.url}\tstatus={e.status}\tfinal={e.final_url}\t{e.error or ''}\n"
                )
        print(f"  Wrote error log: {err_path}")
        print()

    # Deterministic ordering
    print("Step 4: Sorting records deterministically for stable output...")
    ok_records.sort(key=record_sort_key)

    products = [rec for rec in ok_records if rec["document_identifier"]["page_type"] == "Product"]
    articles = [rec for rec in ok_records if rec["document_identifier"]["page_type"] == "Article"]
    collections = [rec for rec in ok_records if rec["document_identifier"]["page_type"] == "CollectionPage"]
    pages = [rec for rec in ok_records if rec["document_identifier"]["page_type"] == "WebPage"]

    products.sort(key=record_sort_key)
    articles.sort(key=record_sort_key)
    collections.sort(key=record_sort_key)
    pages.sort(key=record_sort_key)

    site_all_path = staging_dir / "site_pages.jsonl"
    products_path = staging_dir / "products.jsonl"
    articles_path = staging_dir / "articles.jsonl"
    pages_path = staging_dir / "pages.jsonl"
    coll_path = staging_dir / "collections.jsonl"

    print("Step 5: Writing JSONL (+gz) files...")
    total_all, sha_all = write_ndjson(site_all_path, ok_records)
    n_prod, sha_prod = write_ndjson(products_path, products)
    n_art, sha_art = write_ndjson(articles_path, articles)
    n_page, sha_page = write_ndjson(pages_path, pages)
    n_col, sha_col = write_ndjson(coll_path, collections)
    print(f"  site_pages.jsonl.gz   → {total_all} records")
    print(f"  products.jsonl.gz     → {n_prod} records")
    print(f"  articles.jsonl.gz     → {n_art} records")
    print(f"  pages.jsonl.gz        → {n_page} records")
    print(f"  collections.jsonl.gz  → {n_col} records")
    print()

    print("Step 6: Writing CSV summary files...")
    csv_site_path = staging_dir / "petronaft_site_pages.csv"
    csv_prod_path = staging_dir / "petronaft_products.csv"
    csv_art_path = staging_dir / "petronaft_articles.csv"
    csv_site_n = write_csv_site_pages(csv_site_path, ok_records)
    csv_prod_n = write_csv_products(csv_prod_path, products)
    csv_art_n = write_csv_articles(csv_art_path, articles)
    print(f"  {csv_site_path} → {csv_site_n} rows")
    print(f"  {csv_prod_path} → {csv_prod_n} rows")
    print(f"  {csv_art_path}  → {csv_art_n} rows")
    print()

    version_str = "v1" if prev_version_num is None else f"v{prev_version_num + 1}"

    print("Step 7: Writing platform metadata templates...")
    write_internet_archive_metadata(staging_dir / "internet_archive_metadata.json", version_str)
    write_zenodo_metadata(staging_dir / "zenodo_metadata.json", version_str)
    write_hf_dataset_card(staging_dir / "huggingface_dataset_card.md", version_str)
    write_kaggle_metadata(staging_dir / "kaggle_dataset-metadata.json", version_str)
    write_dataworld_readme(staging_dir / "dataworld_readme.md", version_str)
    write_github_readme(staging_dir / "github_README.md", version_str)
    write_wikidata_quickstatements(staging_dir / "wikidata_quickstatements.tsv")
    print("  Metadata files written for Internet Archive, Zenodo, HF, Kaggle, data.world, GitHub, Wikidata.")
    print()

    print("Step 8: Computing corpus signature and quality metrics...")
    corpus_signature = compute_corpus_signature(site_all_path)
    quality_metrics = compute_quality_metrics(ok_records, products, articles, collections)
    print(f"  Corpus signature: {corpus_signature}")
    print(f"  Quality metrics (high level): {quality_metrics}")
    print()

    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source": SITE_ORIGIN,
        "sitemap_index": SITEMAP_INDEX_URL,
        "counts": {
            "total_records": total_all,
            "products": n_prod,
            "articles": n_art,
            "pages": n_page,
            "collections": n_col,
            "errors": len(errors),
            "csv": {
                "petronaft_site_pages.csv": csv_site_n,
                "petronaft_products.csv": csv_prod_n,
                "petronaft_articles.csv": csv_art_n,
            },
        },
        "hashes_sha256_of_gz": {
            "site_pages.jsonl.gz": sha_all,
            "products.jsonl.gz": sha_prod,
            "articles.jsonl.gz": sha_art,
            "pages.jsonl.gz": sha_page,
            "collections.jsonl.gz": sha_col,
        },
        "quality": quality_metrics,
        "change_detection": {
            "previous_version": None if prev_version_num is None else f"v{prev_version_num}",
            "previous_corpus_signature": prev_signature,
            "corpus_signature": corpus_signature,
            "changed_files": [],
            "unchanged_files": [],
        },
        "output_root": str(OUTPUT_ROOT),
    }

    summary_path = staging_dir / "harvest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  Wrote summary: {summary_path}")
    print()

    # No corpus change ⇒ do not create a new version
    if prev_signature and prev_signature == corpus_signature:
        print("Step 9: Change detection")
        print("  No semantic corpus change compared to previous version.")
        print("  No new version directory created; existing petronaft_ai_ver"
              f"{prev_version_num} remains current.")
        print()
        print("Run complete. Outputs in staging directory (not versioned):")
        print(f"  {staging_dir}")
        print("You may inspect these files manually if needed.")
        shutil.rmtree(staging_dir, ignore_errors=True)
        return 0

    print("Step 9: Change detection – corpus has changed; creating new version directory...")
    if prev_version_num is None:
        new_version_num = 1
    else:
        new_version_num = prev_version_num + 1
    version_dir = OUTPUT_ROOT / f"petronaft_ai_ver{new_version_num}"

    changed_files, unchanged_files = compare_and_copy_changed_files(
        staging_dir, prev_version_dir, version_dir
    )
    summary["change_detection"]["changed_files"] = changed_files
    summary["change_detection"]["unchanged_files"] = unchanged_files
    summary["change_detection"]["previous_version"] = None if prev_version_num is None else f"v{prev_version_num}"
    summary["change_detection"]["previous_corpus_signature"] = prev_signature
    summary["change_detection"]["corpus_signature"] = corpus_signature
    (version_dir / "harvest_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"  New version directory created: {version_dir}")
    print(f"  Changed files:   {len(changed_files)}")
    print(f"  Unchanged files: {len(unchanged_files)}")
    print()

    bundle_path = version_dir / "petronaft_ai_bundle.zip"
    print(f"Step 10: Building bundle ZIP: {bundle_path}")
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Always include harvest_summary.json
        zf.write(version_dir / "harvest_summary.json", arcname="harvest_summary.json")
        for rel in changed_files:
            # Avoid duplicate name for harvest_summary.json inside the ZIP
            if rel == "harvest_summary.json":
                continue
            src = version_dir / rel
            if src.is_file():
                zf.write(src, arcname=rel)
    print("  Bundle ZIP created.")
    print()

    # Versioning summary: which new files replace which old ones
    print("=== Versioning summary ===")
    print(f"Current version: petronaft_ai_ver{new_version_num}")
    if prev_version_dir:
        print(f"Previous version: {prev_version_dir.name}")
    else:
        print("Previous version: None (this is the first version).")
    print()

    if prev_version_dir:
        print("File replacements (new → old):")
        for rel in changed_files:
            new_file = version_dir / rel
            prev_file = prev_version_dir / rel
            if prev_file.exists():
                print(f"  NEW:  {new_file}")
                print(f"  OLD:  {prev_file}")
                print("  → Replace OLD with NEW when publishing.")
            else:
                print(f"  NEW:  {new_file}")
                print("  OLD:  (no previous file; this is a new addition)")
            print()
    else:
        print("All files are new; no previous version to replace.")
        print()

    print("=== How to use these outputs ===")
    print(f"Version directory: {version_dir}")
    print()
    print("Core JSONL (AI ingestion):")
    print("  - site_pages.jsonl.gz     → full corpus of all pages; best for generic embeddings / RAG.")
    print("  - products.jsonl.gz       → only product pages; ideal for product search, specs, pricing assistants.")
    print("  - articles.jsonl.gz       → only articles/news; ideal for educational / technical Q&A.")
    print("  - pages.jsonl.gz          → general web pages (about, contact, etc.).")
    print("  - collections.jsonl.gz    → category/archive pages; good for navigation and site map building.")
    print()
    print("Tabular CSV (analytics, BI, QA):")
    print("  - petronaft_site_pages.csv   → 1 row per page, with meta & basic stats.")
    print("  - petronaft_products.csv     → 1 row per product, with uses/spec tables flags.")
    print("  - petronaft_articles.csv     → 1 row per article, tags/categories counts.")
    print()
    print("Platform metadata / templates:")
    print("  - internet_archive_metadata.json    → for Internet Archive item creation.")
    print("  - zenodo_metadata.json              → for Zenodo deposition metadata.")
    print("  - huggingface_dataset_card.md       → dataset card for Hugging Face Hub.")
    print("  - kaggle_dataset-metadata.json      → dataset-metadata.json for Kaggle.")
    print("  - dataworld_readme.md               → dataset README for data.world.")
    print("  - github_README.md                  → README for a GitHub repository.")
    print("  - wikidata_quickstatements.tsv      → QuickStatements template for Wikidata.")
    print()
    print("Bundle ZIP for easy distribution:")
    print(f"  - {bundle_path}")
    print("    Contains harvest_summary.json + all changed files for this version.")
    print()
    print("Run complete. The corpus is ready to feed into AI systems and to publish")
    print("on platforms like Internet Archive, Zenodo, Hugging Face, Kaggle, data.world, GitHub, and Wikidata.")
    print()

    shutil.rmtree(staging_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
