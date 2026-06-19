import csv
import json
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.append('src')

WH_FIELDS = ["faction", "subfaction", "unit", "model_type", "role", "allegiance", "equipment"]


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")


def test_parse_structured_valid_json():
    from tagging import parse_structured
    raw = json.dumps({
        "faction": "Space Marines",
        "subfaction": "Wolfspear",
        "unit": "Techmarine",
        "model_type": "Infantry",
        "role": "Elites",
        "allegiance": "Imperium",
        "equipment": ["Servo-arm", "Power Armour"],
        "tags": ["Primaris", "Adeptus Mechanicus"],
    })
    parsed = parse_structured(raw, WH_FIELDS)
    assert parsed["faction"] == "Space Marines"
    assert parsed["equipment"] == "Servo-arm, Power Armour"
    assert parsed["tags"] == ["Primaris", "Adeptus Mechanicus"]


def test_parse_structured_unknowns_become_empty():
    from tagging import parse_structured
    raw = '{"faction": "unknown", "unit": "Techmarine", "tags": []}'
    parsed = parse_structured(raw, WH_FIELDS)
    assert parsed["faction"] == ""
    assert parsed["unit"] == "Techmarine"
    assert parsed["role"] == ""


def test_parse_structured_json_inside_prose():
    from tagging import parse_structured
    raw = 'Here you go:\n{"faction": "Orks", "tags": ["Waaagh"]}\nHope that helps!'
    parsed = parse_structured(raw, WH_FIELDS)
    assert parsed["faction"] == "Orks"
    assert parsed["tags"] == ["Waaagh"]


def test_parse_structured_falls_back_to_flat_tags():
    from tagging import parse_structured
    parsed = parse_structured("Orks, Nob, Choppa", WH_FIELDS)
    assert parsed["faction"] == ""
    assert parsed["tags"] == ["Orks", "Nob", "Choppa"]


def test_related_pages_block():
    from tagging import related_pages_block
    metas = [
        {"title": "Techmarine", "categories": "Space Marines, Troops (Space Marines)"},
        {"title": "Techmarine", "categories": "Space Marines, Troops (Space Marines)"},
        {"title": "Wolfspear", "categories": ""},
        None,
    ]
    block = related_pages_block(metas)
    assert block.startswith("Related wiki pages:\n")
    assert block.count("Techmarine") == 1
    assert "[Categories: Space Marines, Troops (Space Marines)]" in block
    assert "- Wolfspear\n" in block
    assert related_pages_block([None]) == ""


def _run_tagging(tmp_path, query_return, response_text, slug_hit=True):
    zips = tmp_path / "zips"
    zips.mkdir()
    (zips / "Wolfspear+Techmarine.stl").write_text("mesh")
    out_csv = tmp_path / "tags.csv"

    empty = {"documents": [[]], "distances": [[]], "metadatas": [[]]}

    def query_side_effect(*args, **kwargs):
        # The slug-filtered query passes where=; make it miss when slug_hit is
        # False so retrieval falls through to the unfiltered query.
        if "where" in kwargs and not slug_hit:
            return empty
        return query_return

    mock_col = MagicMock()
    mock_col.query.side_effect = query_side_effect
    mock_pc = MagicMock()
    mock_pc.get_or_create_collection.return_value = mock_col

    class DummyResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": response_text, "models": []}

    mock_post = MagicMock(return_value=DummyResp())
    with patch("tagging.PersistentClient", return_value=mock_pc), \
            patch("tagging.requests.get", return_value=DummyResp()), \
            patch("tagging.requests.post", mock_post), \
            patch("tagging.count_tokens", side_effect=lambda text, model=None: len(text) // 4):
        from tagging import run_tagging
        run_tagging(str(zips), str(out_csv), str(tmp_path / "db"), None, "warhammer", use_local=True)
    prompt = mock_post.call_args[1]["json"]["prompt"]
    return out_csv, prompt


def test_structured_csv_columns(tmp_path):
    response = json.dumps({
        "faction": "Space Marines", "subfaction": "Wolfspear", "unit": "Techmarine",
        "model_type": "Infantry", "role": "Elites", "allegiance": "Imperium",
        "equipment": ["Servo-arm"], "tags": ["Primaris"],
    })
    out_csv, _ = _run_tagging(tmp_path, {
        "documents": [["Techmarine - lore text"]],
        "distances": [[0.3]],
        "metadatas": [[{"source": "u", "title": "Techmarine", "categories": "Space Marines"}]],
    }, response)

    rows = list(csv.reader(open(out_csv)))
    assert rows[0] == ["filename"] + WH_FIELDS + ["tags", "tagged_at"]
    row = dict(zip(rows[0], rows[1]))
    assert row["faction"] == "Space Marines"
    assert row["unit"] == "Techmarine"
    assert row["allegiance"] == "Imperium"
    assert row["tags"] == "Primaris"


def _tagged_log_line(caplog):
    return next(r.getMessage() for r in caplog.records
               if r.getMessage().startswith("Tagged"))


def test_logs_retrieval_diagnostics_on_weak_unfiltered_match(tmp_path, caplog):
    # Unfiltered fallback (slug miss) with a distant best match: the log must
    # record the distance and that the weak-context note fired, so a sparse
    # result is diagnosable rather than inferred.
    import logging
    with caplog.at_level(logging.INFO):
        _run_tagging(tmp_path, {
            "documents": [["Some unrelated lore"]],
            "distances": [[0.9]],
            "metadatas": [[{"source": "u", "title": "Thing", "categories": "X"}]],
        }, '{"faction": "Orks", "tags": []}', slug_hit=False)
    msg = _tagged_log_line(caplog)
    assert "dist=0.900" in msg
    assert "slug_filtered=False" in msg
    assert "weak_context=True" in msg


def test_logs_retrieval_diagnostics_slug_hit_suppresses_weak_flag(tmp_path, caplog):
    # A slug hit is on-topic by construction, so even a high distance must log
    # weak_context=False (mirrors the prompt's no-weak-note behavior).
    import logging
    with caplog.at_level(logging.INFO):
        _run_tagging(tmp_path, {
            "documents": [["Wolfspear chapter lore"]],
            "distances": [[0.74]],
            "metadatas": [[{"source": "u", "title": "Wolfspear", "categories": "Space Wolves"}]],
        }, '{"faction": "Space Marines", "tags": []}', slug_hit=True)
    msg = _tagged_log_line(caplog)
    assert "dist=0.740" in msg
    assert "slug_filtered=True" in msg
    assert "weak_context=False" in msg


def test_prompt_includes_related_pages_and_no_weak_note_when_close(tmp_path):
    _, prompt = _run_tagging(tmp_path, {
        "documents": [["Techmarine - lore text"]],
        "distances": [[0.3]],
        "metadatas": [[{"source": "u", "title": "Techmarine", "categories": "Space Marines, Troops (Space Marines)"}]],
    }, '{"tags": []}')
    assert "Related wiki pages:" in prompt
    assert "[Categories: Space Marines, Troops (Space Marines)]" in prompt
    assert "weak match" not in prompt


def test_prompt_warns_on_weak_context_when_unfiltered(tmp_path):
    # Unfiltered retrieval (slug filter missed) with a distant best match:
    # the model is warned to ignore the probably-unrelated lore.
    _, prompt = _run_tagging(tmp_path, {
        "documents": [["Obliterator Virus - unrelated lore"]],
        "distances": [[0.9]],
        "metadatas": [[{"source": "u", "title": "Obliterator Virus", "categories": "Diseases"}]],
    }, '{"tags": []}', slug_hit=False)
    assert "weak match" in prompt


def test_no_weak_warning_when_slug_filter_fired(tmp_path):
    # A slug match means the page is on-topic by name, so even a high
    # embedding distance must NOT trigger the weak-context warning (regression:
    # Wolfspear at 0.74 was wrongly told to ignore its own chapter page).
    _, prompt = _run_tagging(tmp_path, {
        "documents": [["Wolfspear - chapter lore"]],
        "distances": [[0.74]],
        "metadatas": [[{"source": "u", "title": "Wolfspear", "categories": "Space Wolves Successor Chapters"}]],
    }, '{"tags": []}', slug_hit=True)
    assert "weak match" not in prompt
    assert "Related wiki pages:" in prompt


def test_weak_warning_threshold_is_minilm_scale(tmp_path):
    # A 0.66 unfiltered match is now considered usable (below the 0.8 default
    # tuned for MiniLM); only clearly-distant matches warn.
    _, prompt = _run_tagging(tmp_path, {
        "documents": [["Some lore"]],
        "distances": [[0.66]],
        "metadatas": [[{"source": "u", "title": "Thing", "categories": "X"}]],
    }, '{"tags": []}', slug_hit=False)
    assert "weak match" not in prompt


def test_header_mismatch_exits(tmp_path):
    zips = tmp_path / "zips"
    zips.mkdir()
    (zips / "m.stl").write_text("mesh")
    out_csv = tmp_path / "tags.csv"
    out_csv.write_text("filename,tags\nold.stl,foo\n")

    with patch("tagging.PersistentClient", return_value=MagicMock()):
        from tagging import run_tagging
        with pytest.raises(SystemExit, match="fresh"):
            run_tagging(str(zips), str(out_csv), str(tmp_path / "db"), None, "warhammer", use_local=True)


def test_is_article_url():
    from embed import is_article_url
    assert is_article_url("https://wh40k.lexicanum.com/wiki/Techmarine")
    assert is_article_url("https://wh40k.lexicanum.com/wiki/Codex:_Space_Marines")
    assert not is_article_url("https://wh40k.lexicanum.com/wiki/File:BStechmarine.jpg")
    assert not is_article_url("https://wh40k.lexicanum.com/wiki/Category:Characters_(Techmarines)")
    assert not is_article_url("https://wh40k.lexicanum.com/wiki/Template:Infobox")
    assert not is_article_url("https://wh40k.lexicanum.com/wiki/User_talk:Someone")
    assert not is_article_url("https://wh40k.lexicanum.com/wiki/Lexicanum:Citation")


def test_run_embedding_skips_namespace_pages(tmp_path):
    docs = [
        {"url": "https://x.com/wiki/Techmarine", "title": "Techmarine", "text": "Real lore."},
        {"url": "https://x.com/wiki/File:Pic.jpg", "title": "File:Pic.jpg", "text": "image page"},
        {"url": "https://x.com/wiki/Category:Orks", "title": "Category:Orks", "text": "listing"},
    ]
    lore_path = tmp_path / "lore.json"
    lore_path.write_text(json.dumps(docs))
    mock_col = MagicMock()
    with patch("embed.chromadb.PersistentClient") as mock_pc:
        mock_pc.return_value.get_or_create_collection.return_value = mock_col
        from embed import run_embedding
        run_embedding(str(lore_path), str(tmp_path / "db"))
    ids = mock_col.upsert.call_args.kwargs["ids"]
    assert all("Techmarine" in i for i in ids)
