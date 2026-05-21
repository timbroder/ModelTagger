import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from tqdm import tqdm
from markdownify import markdownify as md_convert

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Fandom wiki CSS selectors for structured metadata
_INFOBOX_SEL = "aside.portable-infobox"
_CATEGORY_SEL = (
    ".page-header__categories a, "
    "section.categories li a, "
    "#catlinks a, "
    ".categories a"
)
_CONTENT_SEL = ("div.mw-parser-output", "div#mw-content-text")
# Elements inside content that add noise without useful text
_STRIP_CLASSES = {
    "toc", "navbox", "ambox", "noprint",
    "mw-editsection", "reference", "reflist",
    "mw-references-wrap", "printfooter", "catlinks",
}


def _extract_title(soup: BeautifulSoup, url: str) -> str:
    """Return the best available page title."""
    # Prefer the visible page heading
    for sel in ("h1.page-header__title", "h1.firstHeading", "h1"):
        el = soup.select_one(sel)
        if el:
            return el.get_text(strip=True)
    # Fall back to <title>, stripping the " | Site Name" suffix
    tag = soup.find("title")
    if tag:
        return tag.get_text(strip=True).split("|")[0].strip()
    # Last resort: derive from URL
    stem = urlparse(url).path.rstrip("/").split("/")[-1]
    return stem.replace("_", " ").replace("-", " ")


def _extract_headings(content: BeautifulSoup) -> list[str]:
    return [
        h.get_text(strip=True)
        for h in content.find_all(["h2", "h3"])
        if h.get_text(strip=True)
    ]


def _extract_categories(soup: BeautifulSoup) -> list[str]:
    seen: set[str] = set()
    cats: list[str] = []
    for a in soup.select(_CATEGORY_SEL):
        text = a.get_text(strip=True)
        if text and text not in seen:
            seen.add(text)
            cats.append(text)
    return cats


def _extract_infobox(soup: BeautifulSoup) -> dict[str, str]:
    infobox: dict[str, str] = {}
    aside = soup.select_one(_INFOBOX_SEL)
    if not aside:
        return infobox
    for item in aside.find_all("div", class_="pi-item"):
        label_el = item.find(class_="pi-data-label")
        value_el = item.find(class_="pi-data-value")
        if label_el and value_el:
            key = label_el.get_text(strip=True)
            val = value_el.get_text(separator=" ", strip=True)
            if key and val:
                infobox[key] = val
    # Remove infobox from the tree so it doesn't appear in markdown text
    aside.decompose()
    return infobox


def _extract_text(soup: BeautifulSoup) -> str:
    """Return main page content as markdown, falling back to plain paragraphs."""
    content = None
    for sel in _CONTENT_SEL:
        content = soup.select_one(sel)
        if content:
            break

    if content:
        for unwanted in content.find_all(
            ["div", "table", "span", "ul"],
            class_=lambda c: c and bool(set(c) & _STRIP_CLASSES),
        ):
            unwanted.decompose()
        return md_convert(str(content), heading_style="ATX", bullets="-")

    return "\n".join(p.get_text() for p in soup.find_all("p"))


def scrape_url(args: tuple) -> list[tuple[str, int]] | None:
    """Scrape a single URL and append a rich document record to results."""
    url, depth, domain, max_depth, webpage_extensions, image_extensions, visited, results = args
    if url in visited or depth > max_depth:
        return None
    visited.add(url)
    print(f"Scraping ({len(visited)}): {url}")

    new_links = []
    try:
        res = requests.get(url, timeout=10, headers=_HEADERS)
        soup = BeautifulSoup(res.text, "html.parser")

        title = _extract_title(soup, url)
        categories = _extract_categories(soup)
        infobox = _extract_infobox(soup)  # also removes the aside from soup

        # Find content element before headings extraction so we only get
        # article headings, not site-chrome headings.
        content_el = None
        for sel in _CONTENT_SEL:
            content_el = soup.select_one(sel)
            if content_el:
                break
        headings = _extract_headings(content_el if content_el else soup)

        text = _extract_text(soup)

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
            for a in soup.find_all("a", href=True):
                link = urljoin(url, a["href"])
                parsed = urlparse(link)
                ext = parsed.path.lower().split(".")[-1] if "." in parsed.path else ""
                ext = f".{ext}" if ext else ""
                if (
                    parsed.netloc == domain
                    and parsed.path.startswith("/wiki")
                    and ext not in image_extensions
                    and "#" not in link
                    and "?" not in link
                    and (ext in webpage_extensions or parsed.path.endswith("/"))
                ):
                    new_links.append((link, depth + 1))
    except Exception as e:
        print(f"Error scraping {url}: {e}")

    return new_links


def save_progress(output_path: str, results: list[dict]) -> None:
    """Append new results to the output file."""
    try:
        existing_results = []
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                existing_results = json.load(f)
        all_results = existing_results + results
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Progress saved to {output_path}")
    except Exception as e:
        print(f"Error saving progress: {e}")


def load_progress(output_path: str) -> tuple[list[dict], set[str]]:
    """Load progress from output file, returning (results, visited_urls)."""
    if os.path.exists(output_path):
        try:
            with open(output_path, "r") as f:
                results = json.load(f)
            visited = {entry["url"] for entry in results}
            print(f"Loaded progress from {output_path}, {len(visited)} URLs already processed.")
            return results, visited
        except Exception as e:
            print(f"Error loading progress: {e}")
    return [], set()


def run_scraping(
    seed_file: str,
    output_path: str,
    max_pages: int = 100,
    max_depth: int = 2,
    max_threads: int = 5,
    save_interval: int = 10,
) -> None:
    """Scrape lore pages from wiki seeds and save to output file."""
    with open(seed_file) as f:
        seeds = [line.strip() for line in f if line.strip()]

    results, visited = load_progress(output_path)
    queue = [(url, 0) for url in seeds if url not in visited]
    domain = urlparse(seeds[0]).netloc

    webpage_extensions = {"", ".html", ".htm", ".php", ".asp", ".aspx", ".jsp"}
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp"}

    if not queue and len(visited) < max_pages:
        print("All seed URLs have already been visited. Add more seeds to continue scraping.")

    with ThreadPoolExecutor(max_threads) as executor, tqdm(
        total=len(seeds), initial=len(visited), desc="Scraping"
    ) as pbar:
        future_to_url = {}
        processed_count = 0
        while queue and len(visited) < max_pages:
            while queue and len(future_to_url) < max_threads:
                url, depth = queue.pop(0)
                args = (url, depth, domain, max_depth, webpage_extensions, image_extensions, visited, results)
                future_to_url[executor.submit(scrape_url, args)] = url

            for future in as_completed(future_to_url):
                new_links = future.result()
                if new_links:
                    queue.extend(new_links)
                del future_to_url[future]

                processed_count += 1
                pbar.update(1)
                if processed_count % save_interval == 0:
                    save_progress(output_path, results)
                    results.clear()

    save_progress(output_path, results)
    results.clear()
