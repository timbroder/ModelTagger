import csv
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.append('src')


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")


def test_filter_query_tokens():
    from tagging import filter_query_tokens
    words = "space mongol blade master 8016 1n0lirn presupported stl v2 of Blade".split()
    assert filter_query_tokens(words) == ["space", "mongol", "blade", "master"]


def test_candidate_slugs():
    from tagging import candidate_slugs
    assert candidate_slugs(["Wolfspear", "Techmarine"]) == [
        "wolfspear-techmarine", "wolfspear", "techmarine",
    ]
    assert candidate_slugs(["Sister", "Superior"]) == [
        "sister-superior", "sister", "superior",
    ]
    assert candidate_slugs([]) == []


def test_select_context_docs_delta_and_floor():
    from tagging import select_context_docs
    docs = ["a", "b", "c", "d", "e"]
    dists = [0.30, 0.32, 0.40, 0.60, 0.70]
    metas = [None] * 5
    # a,b within delta of best; c,d admitted by the min_docs floor? No —
    # floor is 3, so c is admitted (picked<3 when reached), d is past delta with 3 picked
    assert select_context_docs(docs, dists, metas) == ["a", "b", "c"]


def test_select_context_docs_per_page_cap():
    from tagging import select_context_docs
    docs = ["p1c1", "p1c2", "p1c3", "p2c1", "p3c1"]
    dists = [0.30, 0.31, 0.32, 0.33, 0.34]
    metas = [
        {"source": "p1"}, {"source": "p1"}, {"source": "p1"},
        {"source": "p2"}, {"source": "p3"},
    ]
    # Third chunk from page p1 is skipped in favour of other pages
    assert select_context_docs(docs, dists, metas) == ["p1c1", "p1c2", "p2c1", "p3c1"]


def test_select_context_docs_max_cap():
    from tagging import select_context_docs
    docs = [f"d{i}" for i in range(20)]
    dists = [0.3 + i * 0.001 for i in range(20)]
    metas = [{"source": f"p{i}"} for i in range(20)]
    assert len(select_context_docs(docs, dists, metas)) == 8


def _tag_one_file(tmp_path, mock_col, filename="Wolfspear+Techmarine.stl"):
    zips = tmp_path / "zips"
    zips.mkdir()
    (zips / filename).write_text("mesh")
    out_csv = tmp_path / "tags.csv"

    mock_pc = MagicMock()
    mock_pc.get_or_create_collection.return_value = mock_col

    class DummyResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "tag1, tag2", "models": []}

    with patch("tagging.PersistentClient", return_value=mock_pc), \
            patch("tagging.requests.get", return_value=DummyResp()), \
            patch("tagging.requests.post", return_value=DummyResp()), \
            patch("tagging.count_tokens", side_effect=lambda text, model=None: len(text)):
        from tagging import run_tagging
        run_tagging(str(zips), str(out_csv), str(tmp_path / "db"), None, "warhammer", use_local=True)
    return out_csv


def test_run_tagging_uses_ngram_slug_filter(tmp_path):
    mock_col = MagicMock()
    mock_col.query.return_value = {
        "documents": [["Wolfspear - lore"]],
        "distances": [[0.2]],
        "metadatas": [[{"source": "u1"}]],
    }
    _tag_one_file(tmp_path, mock_col)

    first_call = mock_col.query.call_args_list[0]
    assert first_call.kwargs["where"] == {
        "slug": {"$in": ["wolfspear-techmarine", "wolfspear", "techmarine"]}
    }
    # Filtered query returned docs, so no unfiltered fallback happened
    assert len(mock_col.query.call_args_list) == 1


def test_run_tagging_falls_back_to_unfiltered(tmp_path):
    mock_col = MagicMock()
    empty = {"documents": [[]], "distances": [[]], "metadatas": [[]]}
    full = {
        "documents": [["some lore"]],
        "distances": [[0.4]],
        "metadatas": [[{"source": "u1"}]],
    }
    mock_col.query.side_effect = [empty, full]
    out_csv = _tag_one_file(tmp_path, mock_col)

    calls = mock_col.query.call_args_list
    assert "where" in calls[0].kwargs
    assert "where" not in calls[1].kwargs
    rows = list(csv.reader(open(out_csv)))
    assert rows[1][0] == "Wolfspear+Techmarine.stl"
    assert rows[1][1] != ""
