import os
import sys
from bs4 import BeautifulSoup

sys.path.append('src')

from scrape import (
    _url_to_filename,
    _page_title_from_url,
    _page_url,
    _is_login_wall,
    save_progress,
    load_progress,
)


def test_url_to_filename_prefixes_domain():
    assert _url_to_filename("https://warhammer40k.fandom.com/wiki/Space_Marines") == \
        "warhammer40k--space-marines.md"
    assert _url_to_filename("https://wh40k.lexicanum.com/wiki/Space_Marines") == \
        "wh40k--space-marines.md"


def test_url_to_filename_handles_root_and_encoding():
    assert _url_to_filename("https://example.com/") == "example--index.md"
    assert _url_to_filename("https://example.com/wiki/Adepta%20Sororitas") == \
        "example--adepta-sororitas.md"


def test_page_title_url_round_trip():
    url = "https://wh40k.lexicanum.com/wiki/Adepta_Sororitas"
    title = _page_title_from_url(url)
    assert title == "Adepta_Sororitas"
    assert _page_url(url, "Adepta Sororitas") == url


def test_is_login_wall_by_title():
    soup = BeautifulSoup(
        "<html><head><title>Log in required - Lexicanum</title></head>"
        "<body><div class='mw-parser-output'>" + "x" * 500 + "</div></body></html>",
        "html.parser",
    )
    assert _is_login_wall(soup) is True


def test_is_login_wall_by_short_content():
    soup = BeautifulSoup(
        "<html><head><title>Space Marines - Lexicanum</title></head>"
        "<body><div class='mw-parser-output'>Please log in.</div></body></html>",
        "html.parser",
    )
    assert _is_login_wall(soup) is True


def test_is_login_wall_missing_content_div():
    soup = BeautifulSoup("<html><body><p>nothing wiki-like</p></body></html>", "html.parser")
    assert _is_login_wall(soup) is True


def test_is_login_wall_real_article():
    soup = BeautifulSoup(
        "<html><head><title>Space Marines - Lexicanum</title></head>"
        "<body><div class='mw-parser-output'>"
        "The Space Marines or Adeptus Astartes are foremost amongst the defenders "
        "of humanity, the greatest of the Emperor's warriors. " * 3 +
        "</div></body></html>",
        "html.parser",
    )
    assert _is_login_wall(soup) is False


def _doc(url, title="Title", text="Some lore text."):
    return {
        "url": url,
        "title": title,
        "text": text,
        "headings": ["History"],
        "categories": ["Chapters"],
        "infobox": {"Founding": "First"},
    }


def test_save_and_load_progress_round_trip(tmp_path):
    out = str(tmp_path / "lore")
    docs = [
        _doc("https://warhammer40k.fandom.com/wiki/Ultramarines"),
        _doc("https://wh40k.lexicanum.com/wiki/Ultramarines"),
    ]
    save_progress(out, docs)

    assert os.path.exists(os.path.join(out, "warhammer40k--ultramarines.md"))
    assert os.path.exists(os.path.join(out, "wh40k--ultramarines.md"))
    # save_progress must not consume the text field of the docs it was given
    assert docs[0]["text"] == "Some lore text."

    results, visited = load_progress(out)
    assert results == []
    assert visited == {d["url"] for d in docs}


def test_save_progress_avoids_filename_collisions(tmp_path):
    out = str(tmp_path / "lore")
    save_progress(out, [_doc("https://example.com/wiki/Page", text="first")])
    save_progress(out, [_doc("https://example.com/wiki/Page?x", text="second")])

    files = sorted(f for f in os.listdir(out) if f.endswith(".md"))
    assert files == ["example--page.md", "example--page_1.md"]


def test_load_progress_rebuilds_from_frontmatter(tmp_path):
    out = str(tmp_path / "lore")
    url = "https://warhammer40k.fandom.com/wiki/Tyranids"
    save_progress(out, [_doc(url)])
    os.remove(os.path.join(out, "_visited.txt"))

    results, visited = load_progress(out)
    assert visited == {url}


def test_load_progress_empty_dir():
    results, visited = load_progress("/nonexistent/path/for/test")
    assert results == []
    assert visited == set()


def test_run_scraping_crawls_and_saves(tmp_path, monkeypatch):
    """End-to-end crawl with a fake page fetcher: follows links, respects
    max_pages, terminates, and saves every scraped page."""
    import scrape

    seeds = tmp_path / "seeds.txt"
    seeds.write_text(
        "https://example.com/wiki/Page0\n"
        "https://example.com/wiki/Page1\n"
    )
    out = str(tmp_path / "lore")

    def fake_fetch_api(api_base, page):
        n = int(page.replace("Page", ""))
        links = [{"ns": 0, "*": f"Page{n + 2}"}] if n < 6 else []
        return {
            "title": page,
            "categories": [],
            "sections": [],
            "links": links,
            "text": {"*": f"<p>Lore for {page}</p>"},
        }

    monkeypatch.setattr(scrape, "_fetch_api", fake_fetch_api)
    scrape.run_scraping(str(seeds), out, max_pages=5, max_depth=3, max_threads=3, save_interval=2)

    files = sorted(f for f in os.listdir(out) if f.endswith(".md"))
    assert 5 <= len(files) <= 7  # max_pages plus at most the in-flight overshoot
    _, visited = load_progress(out)
    assert "https://example.com/wiki/Page0" in visited
    assert "https://example.com/wiki/Page1" in visited
