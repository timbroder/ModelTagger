import json
import sys
from unittest.mock import MagicMock, patch

sys.path.append('src')

from embed import (
    clean_markdown,
    split_sections,
    repair_title,
    dedupe_documents,
    build_summary_chunk,
)


def test_clean_markdown_strips_footnote_refs():
    text = "The Codex[[25b]](#fn_25b) was written[[1]](#fn_1) by Guilliman."
    assert clean_markdown(text) == "The Codex was written by Guilliman."


def test_clean_markdown_strips_inline_links():
    text = "See [Ultramarines](/web/2025/https://wh40k.lexicanum.com/wiki/Ultramarines \"Ultramarines\") for more."
    assert clean_markdown(text) == "See Ultramarines for more."


def test_clean_markdown_drops_banner_tables():
    text = (
        "| | |\n"
        "| --- | --- |\n"
        "| | This article contains text passages with citation issues. |\n"
        "\n"
        "The **Codex Astartes** is the doctrine of the Space Marine Chapters.\n"
    )
    cleaned = clean_markdown(text)
    assert "citation issues" not in cleaned
    assert "Codex Astartes" in cleaned


def test_clean_markdown_keeps_data_tables():
    text = (
        "| Rank | Role |\n"
        "| --- | --- |\n"
        "| Captain | Company command |\n"
    )
    assert "Captain" in clean_markdown(text)


def test_split_sections_drops_boilerplate():
    text = (
        "Intro paragraph about the chapter.\n"
        "## Contents\n- 1 History\n- 2 Sources\n"
        "## History\nFounded during the Second Founding.\n"
        "## Sources\n- Codex: Space Marines\n"
        "## See also\n- Ultramarines\n"
    )
    sections = split_sections(text)
    titles = [t for t, _ in sections]
    assert titles == ["", "History"]
    assert "Second Founding" in sections[1][1]


def test_repair_title():
    url = "https://wh40k.lexicanum.com/wiki/Digganob"
    assert repair_title("Warhammer 40k - Lexicanumβ", url) == "Digganob"
    assert repair_title(None, url) == "Digganob"
    assert repair_title("Digganob", url) == "Digganob"
    assert repair_title("Codex Astartes", "https://x.com/wiki/10th_Company") == "Codex Astartes"


def test_dedupe_documents_keeps_longest():
    docs = [
        {"url": "https://x.com/wiki/10th_Company", "title": "Codex Astartes", "text": "short"},
        {"url": "https://x.com/wiki/Codex_Astartes", "title": "Codex Astartes", "text": "much longer text"},
        {"url": "https://x.com/wiki/Digganob", "title": "Digganob", "text": "diggas"},
    ]
    kept = dedupe_documents(docs)
    assert len(kept) == 2
    assert {d["url"] for d in kept} == {"https://x.com/wiki/Codex_Astartes", "https://x.com/wiki/Digganob"}


def test_dedupe_merges_generic_titles_with_real_ones():
    # A generic-site-title doc for the same article as a properly titled one
    docs = [
        {"url": "https://x.com/wiki/Digganob", "title": "Warhammer 40k - Lexicanumβ", "text": "longer digga text"},
        {"url": "https://x.com/wiki/Digganob", "title": "Digganob", "text": "short"},
    ]
    kept = dedupe_documents(docs)
    assert len(kept) == 1
    assert kept[0]["text"] == "longer digga text"


def test_build_summary_chunk():
    chunk = build_summary_chunk(
        "Codex Astartes",
        ["Imperial Texts", "Space Marines"],
        {"Founding": "First", "Author": "Roboute Guilliman"},
    )
    assert chunk == (
        "Codex Astartes. Categories: Imperial Texts, Space Marines. "
        "Founding: First; Author: Roboute Guilliman"
    )
    assert build_summary_chunk("Title", [], {}) is None


def _run_embedding_with_mock(tmp_path, docs, **kwargs):
    lore_path = tmp_path / "lore.json"
    lore_path.write_text(json.dumps(docs))
    mock_col = MagicMock()
    with patch("embed.chromadb.PersistentClient") as mock_pc:
        mock_pc.return_value.get_or_create_collection.return_value = mock_col
        from embed import run_embedding
        run_embedding(str(lore_path), str(tmp_path / "db"), **kwargs)
    return mock_col


def test_run_embedding_emits_summary_and_section_chunks(tmp_path):
    docs = [{
        "url": "https://wh40k.lexicanum.com/wiki/Digganob",
        "title": "Digganob",
        "categories": ["Orks", "Angelis"],
        "infobox": {"Allegiance": "None"},
        "text": "Diggas are tribes of primal humans.\n## History\nThey descend from a survey team.\n## Sources\n- Gorkamorka\n",
    }]
    mock_col = _run_embedding_with_mock(tmp_path, docs)

    call = mock_col.upsert.call_args
    chunks = call.kwargs["documents"]
    metas = call.kwargs["metadatas"]
    ids = call.kwargs["ids"]

    assert chunks[0] == "Digganob. Categories: Orks, Angelis. Allegiance: None"
    assert metas[0]["section"] == "_summary"
    assert metas[0]["categories"] == "Orks, Angelis"
    # Section chunks carry "Title — Section" prefixes; Sources dropped
    assert any(c.startswith("Digganob - Diggas are tribes") for c in chunks)
    assert any(c.startswith("Digganob — History - ") for c in chunks)
    assert not any("Gorkamorka" in c for c in chunks)
    assert ids[0] == "https://wh40k.lexicanum.com/wiki/Digganob#chunk0"


def test_run_embedding_skips_redirects_and_dedupes(tmp_path):
    docs = [
        {"url": "https://x.com/wiki/A", "title": "A", "text": "Redirect to: B"},
        {"url": "https://x.com/wiki/B", "title": "Same Article", "text": "Real text here. " * 5},
        {"url": "https://x.com/wiki/B_alias", "title": "Same Article", "text": "shorter"},
    ]
    mock_col = _run_embedding_with_mock(tmp_path, docs)
    ids = mock_col.upsert.call_args.kwargs["ids"]
    assert all(i.startswith("https://x.com/wiki/B#") for i in ids)


def test_run_embedding_resumes_skipping_embedded_sources(tmp_path):
    docs = [
        {"url": "https://x.com/wiki/A", "title": "A", "text": "Alpha lore text here. " * 4},
        {"url": "https://x.com/wiki/B", "title": "B", "text": "Bravo lore text here. " * 4},
    ]
    lore_path = tmp_path / "lore.json"
    lore_path.write_text(json.dumps(docs))

    mock_col = MagicMock()
    # A is already embedded; only B should be processed on this (resumed) run
    mock_col.get.return_value = {"metadatas": [{"source": "https://x.com/wiki/A"}]}
    with patch("embed.chromadb.PersistentClient") as mock_pc:
        mock_pc.return_value.get_or_create_collection.return_value = mock_col
        from embed import run_embedding
        run_embedding(str(lore_path), str(tmp_path / "db"))

    upserted_sources = set()
    for call in mock_col.upsert.call_args_list:
        upserted_sources.update(m["source"] for m in call.kwargs["metadatas"])
    assert upserted_sources == {"https://x.com/wiki/B"}


def test_embedding_never_splits_a_document_across_batches(tmp_path):
    # Several multi-chunk docs with a tiny batch size: each document's chunks
    # must land entirely within one upsert call so a present source is always
    # complete (the property resume relies on).
    docs = [
        {"url": f"https://x.com/wiki/{n}", "title": n, "categories": [n], "infobox": {"k": "v"},
         "text": f"{n} lore one. {n} lore two. {n} lore three. {n} lore four. {n} lore five."}
        for n in ("A", "B", "C", "D")
    ]
    lore_path = tmp_path / "lore.json"
    lore_path.write_text(json.dumps(docs))

    mock_col = MagicMock()
    mock_col.get.return_value = {"metadatas": []}
    with patch("embed.chromadb.PersistentClient") as mock_pc:
        mock_pc.return_value.get_or_create_collection.return_value = mock_col
        from embed import run_embedding
        run_embedding(str(lore_path), str(tmp_path / "db"),
                      min_chunk_tokens=5, max_chunk_tokens=15, batch_size=3)

    assert mock_col.upsert.call_count >= 2  # forced into multiple batches
    # No source appears in more than one upsert call
    source_to_calls = {}
    for i, call in enumerate(mock_col.upsert.call_args_list):
        for src in {m["source"] for m in call.kwargs["metadatas"]}:
            source_to_calls.setdefault(src, set()).add(i)
    split = {s: c for s, c in source_to_calls.items() if len(c) > 1}
    assert not split, f"these sources were split across batches: {split}"


def test_run_embedding_skips_docs_missing_url(tmp_path):
    # A scraped file with no url in frontmatter must be skipped, not crash
    # (regression: KeyError 'url' on the last document of a 25k-doc run).
    docs = [
        {"title": "No URL", "text": "Orphan lore with no url field. " * 4},
        {"url": "https://x.com/wiki/Good", "title": "Good", "text": "Real lore here. " * 4},
    ]
    mock_col = _run_embedding_with_mock(tmp_path, docs)
    sources = set()
    for call in mock_col.upsert.call_args_list:
        sources.update(m["source"] for m in call.kwargs["metadatas"])
    assert sources == {"https://x.com/wiki/Good"}


def test_embedded_sources_handles_missing_collection():
    from embed import embedded_sources
    bad = MagicMock()
    bad.get.side_effect = RuntimeError("no such collection")
    assert embedded_sources(bad) == set()


def test_chunk_size_defaults_follow_embedder(tmp_path):
    docs = [{"url": "https://x.com/wiki/A", "title": "A", "text": "Some lore text."}]
    captured = {}

    def fake_chunker(text, min_tokens, max_tokens, model="gpt-4o"):
        captured["limits"] = (min_tokens, max_tokens)
        return [text]

    with patch("embed.semantic_chunk_text", side_effect=fake_chunker), \
            patch("chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"):
        _run_embedding_with_mock(tmp_path, docs)
        assert captured["limits"] == (80, 200)
        _run_embedding_with_mock(tmp_path, docs, use_local=True)
        assert captured["limits"] == (300, 800)
        _run_embedding_with_mock(tmp_path, docs, max_chunk_tokens=500, min_chunk_tokens=100)
        assert captured["limits"] == (100, 500)


def test_chunker_bounds_punctuationless_text():
    import tiktoken
    from embed import semantic_chunk_text
    # A long bullet list with no sentence punctuation — one giant "sentence"
    text = "- " + "\n- ".join(f"Squad {i} Tactical Marines with boltguns" for i in range(200))
    chunks = semantic_chunk_text(text, min_tokens=80, max_tokens=200)
    enc = tiktoken.get_encoding("cl100k_base")
    assert chunks
    assert all(len(enc.encode(c)) <= 200 for c in chunks)


def test_real_lore_file_cleanup():
    """Run the cleanup against a real scraped file when the corpus is present."""
    import os
    import frontmatter
    path = "/Users/tim/workspace/ModelTagger/warhammer_lore/wh40k--10th-company.md"
    if not os.path.exists(path):
        import pytest
        pytest.skip("local lore corpus not available")
    post = frontmatter.load(path)
    cleaned = clean_markdown(post.content)
    assert "citation issues" not in cleaned
    assert "Codex Astartes" in cleaned
    sections = split_sections(cleaned)
    titles = [t.lower() for t, _ in sections]
    assert "sources" not in titles
    assert "contents" not in titles
    assert not any("](#fn_" in body for _, body in sections)
