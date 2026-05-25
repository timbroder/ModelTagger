from __future__ import annotations

import os
import re
import time
import threading
import requests
import frontmatter
from bs4 import BeautifulSoup
from urllib.parse import urlparse, unquote, quote
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from tqdm import tqdm
from markdownify import markdownify as md_convert

from utils import slugify

_HEADERS = {
    "User-Agent": "ModelTagger/1.0 (wiki scraper; contact via github.com/timbroder/ModelTagger)"
}

# Shared backoff state for Wayback Machine requests (across threads)
_wayback_lock = threading.Lock()
_wayback_pause_until: float = 0.0


def _wayback_get(url: str, max_retries: int = 4, timeout: int = 10, **kwargs) -> requests.Response | None:
    """GET a Wayback Machine URL with rate-limit and connection-refused backoff.

    Accepts the same keyword args as requests.get (e.g. params=).
    On HTTP 429/503 or network error the pause time is extended exponentially
    (2min, 4min, 8min, capped at 10min) and shared across all threads so the
    whole scraper backs off together rather than piling on.

    timeout=10s: Wayback pages average 1.5s, high-end ~6s. Anything slower
    is being throttled and not worth waiting for.
    """
    global _wayback_pause_until
    for attempt in range(max_retries):
        with _wayback_lock:
            wait = _wayback_pause_until - time.time()
        if wait > 0:
            print(f"  [Wayback] rate limited — pausing {wait:.0f}s")
            time.sleep(wait)
        try:
            resp = requests.get(url, timeout=timeout, headers=_HEADERS, **kwargs)
            if resp.status_code in (429, 503):
                delay = min(600, 120 * 2 ** attempt)  # 2min, 4min, 8min, capped at 10min
                print(f"  [Wayback] {resp.status_code} rate limited — backing off {delay}s ({delay//60}min)")
                with _wayback_lock:
                    _wayback_pause_until = max(_wayback_pause_until, time.time() + delay)
                time.sleep(delay)
                continue
            return resp
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            delay = min(600, 120 * 2 ** attempt)  # 2min, 4min, 8min, capped at 10min
            print(f"  [Wayback] connection error/timeout — backing off {delay}s ({delay//60}min)")
            with _wayback_lock:
                _wayback_pause_until = max(_wayback_pause_until, time.time() + delay)
            time.sleep(delay)
    print(f"  [Wayback] giving up after {max_retries} attempts: {url}")
    return None


# Domains that block all automated access — skip immediately rather than
# burning time on API + HTML + Wayback fallbacks
_BLOCKED_DOMAINS: set[str] = set()

# Domains where the live site requires login or is rate-limited; go straight
# to Wayback CDX to find a pre-login-wall snapshot
_WAYBACK_ONLY_DOMAINS = {
    "wh40k.lexicanum.com",
}

# Elements inside rendered HTML that add noise without useful text
_CATEGORY_NOISE = {"Categories", "categories", "Category", "category", "Hidden categories"}

_STRIP_CLASSES = {
    "toc", "navbox", "ambox", "noprint",
    "mw-editsection", "reference", "reflist",
    "mw-references-wrap", "printfooter", "catlinks",
}

# Infobox selectors: fandom portable-infobox first, plain wikitable fallback
_INFOBOX_SELS = [
    "aside.portable-infobox",   # Fandom
    "table.infobox",            # Lexicanum / standard MediaWiki
]


def _url_to_filename(url: str) -> str:
    parsed = urlparse(url)
    segment = unquote(parsed.path.rstrip("/").split("/")[-1])
    slug = slugify(segment) if segment else "index"
    # Prefix with domain slug so files from different wikis don't collide
    domain_slug = slugify(parsed.netloc.split(".")[0])
    return f"{domain_slug}--{slug}.md"


def _wiki_api_base(url: str) -> str:
    """Return the MediaWiki api.php base URL for any wiki URL."""
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}/api.php"


def _page_title_from_url(url: str) -> str:
    """Extract the page title from a /wiki/Page_Title URL."""
    path = urlparse(url).path
    if "/wiki/" in path:
        return unquote(path.split("/wiki/", 1)[1])
    return unquote(path.lstrip("/"))


def _page_url(base_url: str, title: str) -> str:
    """Reconstruct a canonical wiki URL from a base URL and page title."""
    p = urlparse(base_url)
    return f"{p.scheme}://{p.netloc}/wiki/{quote(title.replace(' ', '_'), safe=':/')}"


def _fetch_api(api_base: str, page: str) -> dict | None:
    """Call MediaWiki parse API and return the parse dict, or None if unavailable."""
    params = {
        "action": "parse",
        "page": page,
        "prop": "text|categories|sections|links",
        "disableeditsection": "1",
        "format": "json",
    }
    try:
        resp = requests.get(api_base, params=params, timeout=15, headers=_HEADERS)
        if resp.status_code in (403, 404):
            return None
        resp.raise_for_status()
        data = resp.json()
        if "error" in data or "parse" not in data:
            return None
        return data["parse"]
    except Exception:
        return None


def _best_content_div(soup: BeautifulSoup):
    """Return the content div with the most text, preferring mw-parser-output.

    Some Lexicanum layouts put a login notice inside mw-parser-output but the
    real article text directly inside mw-content-text. Picking the larger div
    handles both cases.
    """
    candidates = [soup.select_one(sel) for sel in ("div.mw-parser-output", "div#mw-content-text")]
    candidates = [c for c in candidates if c is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda c: len(c.get_text(strip=True)))


def _is_login_wall(soup: BeautifulSoup) -> bool:
    """Return True if the page is a wiki login wall rather than real content.

    A real login wall has no article text — just a short notice. We detect it
    by page title or by the best content div being suspiciously short.
    The presence of a Special:UserLogin link is NOT used here because Lexicanum
    embeds a "Log in" notice inside the content div on every anonymous page.
    """
    title_tag = soup.find("title")
    if title_tag:
        t = title_tag.get_text().lower()
        if "log in" in t or "login required" in t:
            return True

    content = _best_content_div(soup)
    if content is None:
        return True
    if len(content.get_text(strip=True)) < 100:
        return True

    return False


def _fetch_html(url: str, resp: requests.Response | None = None) -> dict | None:
    """Scrape page HTML directly. Returns a partial parse-like dict or None.

    If ``resp`` is provided (e.g. already fetched via _wayback_get) it is used
    directly; otherwise a fresh GET is issued.
    """
    if resp is None:
        resp = requests.get(url, timeout=15, headers=_HEADERS)
    if not resp.ok:
        return None
    soup = BeautifulSoup(resp.text, "html.parser")

    # Detect Cloudflare challenge or wiki login wall
    title_tag = soup.find("title")
    if title_tag and "just a moment" in title_tag.get_text().lower():
        return None
    if _is_login_wall(soup):
        return None

    # Title
    title = None
    for sel in ("h1.page-header__title", "h1.firstHeading", "h1"):
        el = soup.select_one(sel)
        if el:
            title = el.get_text(strip=True)
            break
    if not title and title_tag:
        title = title_tag.get_text(strip=True).split("|")[0].strip()

    # Categories from HTML
    cats: list[str] = []
    seen: set[str] = set()
    for a in soup.select(".page-header__categories a, section.categories li a, #catlinks a, .categories a"):
        text = a.get_text(strip=True)
        if text and len(text) > 1 and text not in _CATEGORY_NOISE and not re.match(r"^\d+ more$", text, re.IGNORECASE) and text not in seen:
            seen.add(text)
            cats.append(text)

    # Headings from content div (pick whichever has more text)
    content = _best_content_div(soup)
    headings = [h.get_text(strip=True) for h in (content or soup).find_all(["h2", "h3"]) if h.get_text(strip=True)]

    # Links for crawling
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/wiki/") and "#" not in href and "?" not in href:
            links.append({"ns": 0, "*": unquote(href[len("/wiki/"):])})

    return {
        "title": title or "",
        "categories": [{"*": c} for c in cats],
        "sections": [{"line": h, "toclevel": 1} for h in headings],
        "links": links,
        "text": {"*": str(content) if content else ""},
        "_soup": soup,  # carry soup so infobox extraction can reuse it
    }


def _fetch_wayback(url: str) -> dict | None:
    """Fetch the most recent Wayback Machine snapshot for a URL."""
    try:
        resp = _wayback_get(f"https://web.archive.org/web/{url}")
        return _fetch_html(f"https://web.archive.org/web/{url}", resp=resp) if resp else None
    except Exception:
        return None


def _fetch_wayback_cdx(url: str) -> dict | None:
    """Use the Wayback CDX API to find a usable snapshot and fetch it.

    Fetches one representative snapshot per calendar month (collapse=timestamp:6),
    sorted newest-first, up to 72 months (~6 years). Stops as soon as a
    non-login-wall snapshot is found. _is_login_wall() in _fetch_html rejects
    login-wall pages automatically. All Wayback requests go through _wayback_get
    so 429s and connection refusals trigger shared exponential backoff.
    """
    try:
        cdx_resp = _wayback_get(
            "http://web.archive.org/cdx/search/cdx",
            timeout=15,  # CDX is a DB query — give it a bit more headroom
            params={
                "url": url,
                "output": "json",
                "fl": "timestamp",
                "filter": "statuscode:200",
                "from": "20190101",
                "collapse": "timestamp:6",
                "limit": 72,
                "sort": "reverse",
            },
        )
        if cdx_resp is None or not cdx_resp.ok:
            return None
        rows = cdx_resp.json()
        n = len(rows) - 1  # subtract header row
        if n <= 0:
            print(f"  [Wayback CDX] no snapshots found for {url}")
            return None
        print(f"  [Wayback CDX] {n} snapshots to try for {url}")
        for row in rows[1:]:  # skip header row
            time.sleep(1)  # Wayback rate limit is ~1 req/sec; stay under it
            wayback_url = f"https://web.archive.org/web/{row[0]}/{url}"
            page_resp = _wayback_get(wayback_url)
            if page_resp is None:
                print(f"  [Wayback CDX] {row[0]}: _wayback_get returned None")
                continue
            if not page_resp.ok:
                print(f"  [Wayback CDX] {row[0]}: HTTP {page_resp.status_code}, skipping")
                continue
            result = _fetch_html(wayback_url, resp=page_resp)
            if result and result.get("text", {}).get("*", "").strip():
                print(f"  [Wayback CDX] {row[0]}: found usable content")
                return result
            print(f"  [Wayback CDX] {row[0]}: login wall or empty content")
        return None
    except Exception as e:
        print(f"  [Wayback CDX] exception: {e}")
        return None


def _parse_infobox(soup: BeautifulSoup) -> dict[str, str]:
    infobox: dict[str, str] = {}
    for sel in _INFOBOX_SELS:
        el = soup.select_one(sel)
        if not el:
            continue
        # Fandom portable-infobox
        if el.name == "aside":
            for item in el.find_all("div", class_="pi-item"):
                label = item.find(class_="pi-data-label")
                value = item.find(class_="pi-data-value")
                if label and value:
                    k = label.get_text(strip=True)
                    v = value.get_text(separator=" ", strip=True)
                    if k and v:
                        infobox[k] = v
        # Standard wikitable infobox (Lexicanum / vanilla MediaWiki)
        else:
            for row in el.find_all("tr"):
                cells = row.find_all(["th", "td"])
                if len(cells) == 2:
                    k = cells[0].get_text(strip=True)
                    v = cells[1].get_text(separator=" ", strip=True)
                    if k and v:
                        infobox[k] = v
        el.decompose()
        break  # only grab the first infobox
    return infobox


def _html_to_markdown(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for unwanted in soup.find_all(
        ["div", "table", "span", "ul"],
        class_=lambda c: c and bool(set(c) & _STRIP_CLASSES),
    ):
        unwanted.decompose()
    # Remove images and their enclosing <a> wrappers to keep text clean
    for img in soup.find_all("img"):
        parent = img.parent
        img.decompose()
        if parent and parent.name == "a" and not parent.get_text(strip=True):
            parent.decompose()
    for figure in soup.find_all("figure"):
        figure.decompose()
    return md_convert(str(soup), heading_style="ATX", bullets="-")


def scrape_url(args: tuple) -> list[tuple[str, int]] | None:
    """Fetch a wiki page (API first, HTML fallback) and append a document record to results."""
    url, depth, max_depth, visited, results = args
    if url in visited or depth > max_depth:
        return None
    domain = urlparse(url).netloc
    if domain in _BLOCKED_DOMAINS:
        print(f"  Skipping {url} — domain is blocklisted")
        visited.add(url)
        return []
    visited.add(url)
    print(f"Scraping ({len(visited)}): {url}")

    new_links: list[tuple[str, int]] = []
    try:
        page = _page_title_from_url(url)
        if domain in _WAYBACK_ONLY_DOMAINS:
            parsed = _fetch_wayback_cdx(url)
            if parsed:
                print(f"  Using Wayback CDX snapshot for {url}")
            else:
                print(f"  Skipping {url} — Wayback CDX found no usable snapshot")
                return new_links
        else:
            parsed = _fetch_api(_wiki_api_base(url), page)
            if parsed is None:
                parsed = _fetch_html(url)
            if parsed is None:
                parsed = _fetch_wayback(url)
                if parsed:
                    print(f"  Using Wayback Machine snapshot for {url}")
        if parsed is None:
            print(f"  Skipping {url} — API, HTML, and Wayback all failed")
            return new_links

        title = parsed.get("title", page)

        categories = [
            c["*"].replace("_", " ")
            for c in parsed.get("categories", [])
            if not c.get("hidden") and len(c["*"]) > 1
        ]
        headings = [
            s["line"]
            for s in parsed.get("sections", [])
            if s.get("toclevel", 1) <= 2
        ]

        content_html = parsed.get("text", {}).get("*", "")
        full_page_soup = parsed.pop("_soup", None)

        # _soup is the full page (HTML fallback path) — use it for infobox
        # detection since the infobox may sit outside the content div.
        # For text conversion always use the content div HTML so navigation
        # and sidebars are excluded.
        infobox_soup = full_page_soup if full_page_soup is not None else BeautifulSoup(content_html, "html.parser")
        infobox = _parse_infobox(infobox_soup)
        text = _html_to_markdown(content_html)

        results.append({
            "url": url,
            "title": title,
            "text": text,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "headings": headings,
            "categories": categories,
            "infobox": infobox,
        })

        if depth < max_depth:
            for link in parsed.get("links", []):
                if link.get("ns", -1) != 0:
                    continue
                linked_url = _page_url(url, link["*"])
                if urlparse(linked_url).netloc == domain:
                    new_links.append((linked_url, depth + 1))

    except Exception as e:
        print(f"Error scraping {url}: {e}")

    return new_links


def save_progress(output_dir: str, results: list[dict]) -> None:
    """Write each result as a markdown file with YAML frontmatter."""
    os.makedirs(output_dir, exist_ok=True)
    visited_path = os.path.join(output_dir, "_visited.txt")
    try:
        with open(visited_path, "a") as vf:
            for doc in results:
                filename = _url_to_filename(doc["url"])
                filepath = os.path.join(output_dir, filename)
                if os.path.exists(filepath):
                    base = filename[:-3]
                    i = 1
                    while os.path.exists(filepath):
                        filepath = os.path.join(output_dir, f"{base}_{i}.md")
                        i += 1

                text = doc.pop("text", "")
                post = frontmatter.Post(text, **doc)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(frontmatter.dumps(post))
                doc["text"] = text

                vf.write(doc["url"] + "\n")
        print(f"Progress saved to {output_dir}")
    except Exception as e:
        print(f"Error saving progress: {e}")


def load_progress(output_dir: str) -> tuple[list[dict], set[str]]:
    """Load visited URLs from the index file, falling back to frontmatter scan."""
    visited_path = os.path.join(output_dir, "_visited.txt")
    if os.path.exists(visited_path):
        try:
            with open(visited_path) as f:
                visited = {line.strip() for line in f if line.strip()}
            print(f"Loaded progress from {output_dir}, {len(visited)} URLs already processed.")
            return [], visited
        except Exception as e:
            print(f"Error loading progress: {e}")

    # _visited.txt missing — rebuild from .md frontmatter so we don't re-scrape
    if os.path.isdir(output_dir):
        visited: set[str] = set()
        for fname in os.listdir(output_dir):
            if not fname.endswith(".md") or fname.startswith("_"):
                continue
            try:
                post = frontmatter.load(os.path.join(output_dir, fname))
                url = post.metadata.get("url")
                if url:
                    visited.add(url)
            except Exception:
                pass
        if visited:
            print(f"Rebuilt progress from .md files in {output_dir}, {len(visited)} URLs already processed.")
        return [], visited

    return [], set()


def run_scraping(
    seed_file: str,
    output_dir: str,
    max_pages: int = 100,
    max_depth: int = 2,
    max_threads: int = 5,
    save_interval: int = 10,
) -> None:
    """Scrape lore pages from wiki seeds via MediaWiki API and save as markdown files."""
    with open(seed_file) as f:
        seeds = [line.strip() for line in f if line.strip()]

    results, visited = load_progress(output_dir)
    queue = [(url, 0) for url in seeds if url not in visited]

    if not queue:
        print("All seed URLs have already been visited. Add more seeds to continue scraping.")

    with ThreadPoolExecutor(max_threads) as executor, tqdm(
        total=len(seeds), initial=len(visited), desc="Scraping"
    ) as pbar:
        future_to_url: dict = {}
        processed_count = 0
        while queue and len(visited) < max_pages:
            while queue and len(future_to_url) < max_threads:
                url, depth = queue.pop(0)
                args = (url, depth, max_depth, visited, results)
                future_to_url[executor.submit(scrape_url, args)] = url

            for future in as_completed(future_to_url):
                new_links = future.result()
                if new_links:
                    queue.extend(new_links)
                del future_to_url[future]

                processed_count += 1
                pbar.update(1)
                if processed_count % save_interval == 0:
                    save_progress(output_dir, results)
                    results.clear()

    save_progress(output_dir, results)
    results.clear()
