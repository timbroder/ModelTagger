import os
import sys
from unittest.mock import MagicMock, patch

import pytest
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


@pytest.fixture(autouse=True)
def _reset_scrape_state():
    import scrape
    scrape._retry_counts.clear()
    scrape._snapshot_indexes.clear()
    yield
    scrape._retry_counts.clear()
    scrape._snapshot_indexes.clear()


def test_collapse_monthly():
    from scrape import _collapse_monthly
    ts = ["20250101120000", "20250115120000", "20250201120000", "20240301120000"]
    # newest first, one per calendar month
    assert _collapse_monthly(ts) == ["20250201120000", "20250115120000", "20240301120000"]
    assert _collapse_monthly(ts, limit=2) == ["20250201120000", "20250115120000"]


def _cdx_resp(rows):
    r = MagicMock()
    r.ok = True
    r.json.return_value = rows
    return r


def test_fetch_domain_snapshots_paginates_with_resume_key():
    import scrape
    page1 = _cdx_resp([
        ["original", "timestamp"],
        ["https://wh40k.lexicanum.com/wiki/Phoenix", "20250101000000"],
        ["https://wh40k.lexicanum.com/wiki/Phoenix", "20260201000000"],
        [],
        ["resume-key-1"],
    ])
    page2 = _cdx_resp([
        ["original", "timestamp"],
        ["https://wh40k.lexicanum.com/wiki/Asurmen", "20251515000000"],
    ])
    with patch("scrape._wayback_get", side_effect=[page1, page2]) as mock_get:
        index = scrape._fetch_domain_snapshots("wh40k.lexicanum.com")

    assert index["/wiki/Phoenix"] == ["20260201000000", "20250101000000"]
    assert index["/wiki/Asurmen"] == ["20251515000000"]
    assert mock_get.call_args_list[1].kwargs["params"]["resumeKey"] == "resume-key-1"


def test_ensure_snapshot_index_uses_cache(tmp_path):
    import json
    import scrape
    cache = tmp_path / "_snapshots_wh40k-lexicanum-com.json"
    cache.write_text(json.dumps({"/wiki/Phoenix": ["20260201000000"]}))
    with patch("scrape._fetch_domain_snapshots") as mock_fetch:
        scrape._ensure_snapshot_index("wh40k.lexicanum.com", str(tmp_path))
    mock_fetch.assert_not_called()
    assert scrape._snapshot_indexes["wh40k.lexicanum.com"]["/wiki/Phoenix"] == ["20260201000000"]


def test_fetch_wayback_cdx_with_index_no_snapshots_is_permanent_skip():
    import scrape
    scrape._snapshot_indexes["wh40k.lexicanum.com"] = {}
    assert scrape._fetch_wayback_cdx("https://wh40k.lexicanum.com/wiki/Nonexistent") is None


def test_fetch_wayback_cdx_with_index_fetches_snapshot():
    import scrape
    scrape._snapshot_indexes["wh40k.lexicanum.com"] = {"/wiki/Phoenix": ["20260201000000"]}
    ok_resp = MagicMock(ok=True)
    parsed = {"title": "Phoenix", "text": {"*": "<p>lore</p>"}}
    with patch("scrape._wayback_get", return_value=ok_resp) as mock_get, \
            patch("scrape._fetch_html", return_value=parsed):
        result = scrape._fetch_wayback_cdx("https://wh40k.lexicanum.com/wiki/Phoenix")
    assert result == parsed
    assert "20260201000000" in mock_get.call_args.args[0]


def test_fetch_wayback_cdx_all_network_failures_is_transient():
    import scrape
    from scrape import TransientScrapeError
    scrape._snapshot_indexes["wh40k.lexicanum.com"] = {
        "/wiki/Phoenix": ["20260201000000", "20250101000000"],
    }
    with patch("scrape._wayback_get", return_value=None):
        with pytest.raises(TransientScrapeError):
            scrape._fetch_wayback_cdx("https://wh40k.lexicanum.com/wiki/Phoenix")


def test_scrape_url_requeues_on_transient_failure():
    import scrape
    from scrape import scrape_url, TransientScrapeError
    url = "https://wh40k.lexicanum.com/wiki/Phoenix"
    visited: set = set()
    results: list = []

    with patch("scrape._fetch_wayback_cdx", side_effect=TransientScrapeError("boom")):
        out = scrape_url((url, 0, 2, visited, results))

    assert out == [(url, 0)]
    assert url not in visited  # eligible for the requeued attempt

    # After exhausting retries, the page is abandoned for this run
    scrape._retry_counts[url] = 3
    with patch("scrape._fetch_wayback_cdx", side_effect=TransientScrapeError("boom")):
        out = scrape_url((url, 0, 2, visited, results))
    assert out == []
    assert results == []


def test_scrape_url_filters_namespace_links():
    from scrape import scrape_url
    parsed = {
        "title": "Phoenix",
        "categories": [],
        "sections": [],
        "links": [
            {"ns": 0, "*": "Asurmen"},
            {"ns": 0, "*": "File:Phoenix.jpg"},
            {"ns": 0, "*": "Category:Aeldari"},
        ],
        "text": {"*": "<p>lore</p>"},
    }
    visited: set = set()
    results: list = []
    with patch("scrape._fetch_api", return_value=parsed):
        links = scrape_url(("https://example.com/wiki/Phoenix", 0, 2, visited, results))
    assert [u for u, _ in links] == ["https://example.com/wiki/Asurmen"]


def test_run_scraping_filters_namespace_seeds(tmp_path, monkeypatch):
    import scrape
    seeds = tmp_path / "seeds.txt"
    seeds.write_text(
        "https://example.com/wiki/Phoenix\n"
        "https://example.com/wiki/File:Phoenix.jpg\n"
        "https://example.com/wiki/Category:Aeldari\n"
    )
    out = str(tmp_path / "lore")
    scraped = []

    def fake_fetch_api(api_base, page):
        scraped.append(page)
        return {"title": page, "categories": [], "sections": [], "links": [],
                "text": {"*": "<p>lore</p>"}}

    monkeypatch.setattr(scrape, "_fetch_api", fake_fetch_api)
    scrape.run_scraping(str(seeds), out, max_pages=10, max_depth=0, max_threads=2)
    assert scraped == ["Phoenix"]


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
