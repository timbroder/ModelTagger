from __future__ import annotations

import os
import re
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


def _fetch_html(url: str) -> dict | None:
    """Scrape page HTML directly. Returns a partial parse-like dict or None."""
    resp = requests.get(url, timeout=15, headers=_HEADERS)
    if not resp.ok:
        return None
    soup = BeautifulSoup(resp.text, "html.parser")

    # Detect Cloudflare challenge page
    title_tag = soup.find("title")
    if title_tag and "just a moment" in title_tag.get_text().lower():
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

    # Headings from content div
    content = None
    for sel in ("div.mw-parser-output", "div#mw-content-text"):
        content = soup.select_one(sel)
        if content:
            break
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
    visited.add(url)
    print(f"Scraping ({len(visited)}): {url}")

    new_links: list[tuple[str, int]] = []
    try:
        page = _page_title_from_url(url)
        parsed = _fetch_api(_wiki_api_base(url), page)
        via_html = False
        if parsed is None:
            parsed = _fetch_html(url)
            via_html = True
        if parsed is None:
            print(f"  Skipping {url} — both API and HTML requests failed")
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

        # Reuse soup from HTML fallback to avoid double-parsing
        soup = parsed.pop("_soup", None)
        if soup is None:
            html = parsed.get("text", {}).get("*", "")
            soup = BeautifulSoup(html, "html.parser")

        infobox = _parse_infobox(soup)
        text = _html_to_markdown(str(soup))

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
            domain = urlparse(url).netloc
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
    """Load visited URLs from the index file."""
    visited_path = os.path.join(output_dir, "_visited.txt")
    if os.path.exists(visited_path):
        try:
            with open(visited_path) as f:
                visited = {line.strip() for line in f if line.strip()}
            print(f"Loaded progress from {output_dir}, {len(visited)} URLs already processed.")
            return [], visited
        except Exception as e:
            print(f"Error loading progress: {e}")
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
