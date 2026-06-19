import csv
import json
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.append('src')

WH_FIELDS = ["faction", "subfaction", "unit", "model_type", "role", "allegiance", "equipment"]


@pytest.fixture(autouse=True)
def _reset_client():
    import tagging
    tagging._anthropic_client = None
    yield
    tagging._anthropic_client = None


def _fake_response(record: dict, in_tokens=100, out_tokens=40):
    block = MagicMock()
    block.type = "text"
    block.text = json.dumps(record)
    resp = MagicMock()
    resp.content = [block]
    resp.usage.input_tokens = in_tokens
    resp.usage.output_tokens = out_tokens
    return resp


def test_build_schema_permissive():
    from tagging import build_schema
    s = build_schema(["a", "b"])
    assert s["properties"]["a"] == {"type": "string"}
    assert s["properties"]["tags"]["type"] == "array"
    assert s["required"] == ["a", "b", "tags"]
    assert s["additionalProperties"] is False


def test_normalize_record_lists_and_unknowns():
    from tagging import normalize_record
    rec = {
        "faction": "Space Marines",
        "role": "unknown",
        "equipment": ["Servo-arm", "Power Armour"],
        "tags": ["Primaris", " "],
    }
    out = normalize_record(rec, WH_FIELDS)
    assert out["faction"] == "Space Marines"
    assert out["role"] == ""               # "unknown" -> empty
    assert out["equipment"] == "Servo-arm, Power Armour"
    assert out["tags"] == ["Primaris"]


def test_ask_anthropic_returns_guaranteed_record():
    from tagging import ask_anthropic
    record = {"faction": "Orks", "tags": ["Waaagh"]}
    client = MagicMock()
    client.messages.create.return_value = _fake_response(record)
    with patch("tagging.get_anthropic_client", return_value=client):
        out, tokens = ask_anthropic("prompt", {"type": "object"}, model="claude-sonnet-4-6")
    assert out == record
    assert tokens == 140
    # The schema was passed via output_config (structured output enforcement)
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["output_config"]["format"]["type"] == "json_schema"
    assert kwargs["model"] == "claude-sonnet-4-6"


def test_ask_anthropic_returns_none_on_failure(monkeypatch):
    from tagging import ask_anthropic
    monkeypatch.setattr("tagging.time.sleep", lambda s: None)
    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("boom")
    with patch("tagging.get_anthropic_client", return_value=client):
        out, tokens = ask_anthropic("p", {}, retries=2)
    assert out is None and tokens == 0


def _run_tag_anthropic(tmp_path, record, filename="Wolfspear+Techmarine.stl",
                       query_return=None):
    zips = tmp_path / "zips"
    zips.mkdir()
    (zips / filename).write_text("mesh")
    out_csv = tmp_path / "tags.csv"

    if query_return is None:
        query_return = {
            "documents": [["Wolfspear - lore"]],
            "distances": [[0.3]],
            "metadatas": [[{"source": "u", "title": "Wolfspear", "categories": "Space Wolves"}]],
        }
    mock_col = MagicMock()
    mock_col.query.return_value = query_return
    mock_pc = MagicMock()
    mock_pc.get_or_create_collection.return_value = mock_col

    client = MagicMock()
    client.messages.create.return_value = _fake_response(record)

    with patch("tagging.PersistentClient", return_value=mock_pc), \
            patch("tagging.get_anthropic_client", return_value=client), \
            patch("tagging.count_tokens", side_effect=lambda text, model=None: len(text) // 4):
        from tagging import run_tagging
        run_tagging(str(zips), str(out_csv), str(tmp_path / "db"), None, "warhammer",
                    provider="anthropic")
    return out_csv, client


def test_run_tagging_anthropic_writes_schema_columns(tmp_path):
    record = {
        "faction": "Space Marines", "subfaction": "Wolfspear", "unit": "Techmarine",
        "model_type": "Infantry", "role": "Elites", "allegiance": "Imperium",
        "equipment": ["Servo-arm"], "tags": ["Primaris"],
    }
    out_csv, client = _run_tag_anthropic(tmp_path, record)

    rows = list(csv.reader(open(out_csv)))
    assert rows[0] == ["filename"] + WH_FIELDS + ["tags", "tagged_at"]
    row = dict(zip(rows[0], rows[1]))
    assert row["faction"] == "Space Marines"
    assert row["subfaction"] == "Wolfspear"
    assert row["equipment"] == "Servo-arm"
    assert row["tags"] == "Primaris"
    assert row["tagged_at"]  # timestamp recorded

    # The enum-constrained schema (faction enum) was sent to the API
    schema = client.messages.create.call_args.kwargs["output_config"]["format"]["schema"]
    assert "Adepta Sororitas" in schema["properties"]["faction"]["enum"]


def test_run_tagging_anthropic_failure_not_written_so_it_retries(tmp_path):
    # ask_anthropic failing -> the file is NOT written, so a re-run retries it
    zips = tmp_path / "zips"
    zips.mkdir()
    (zips / "m.stl").write_text("mesh")
    out_csv = tmp_path / "tags.csv"
    mock_col = MagicMock()
    mock_col.query.return_value = {
        "documents": [["lore"]], "distances": [[0.3]], "metadatas": [[{"source": "u"}]],
    }
    mock_pc = MagicMock()
    mock_pc.get_or_create_collection.return_value = mock_col

    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("boom")
    with patch("tagging.PersistentClient", return_value=mock_pc), \
            patch("tagging.get_anthropic_client", return_value=client), \
            patch("tagging.time.sleep", lambda s: None), \
            patch("tagging.count_tokens", side_effect=lambda text, model=None: len(text) // 4):
        from tagging import run_tagging
        run_tagging(str(zips), str(out_csv), str(tmp_path / "db"), None, "warhammer",
                    provider="anthropic")

    rows = list(csv.reader(open(out_csv)))
    assert len(rows) == 1  # header only; failed file omitted
    assert "m.stl" not in {r[0] for r in rows[1:]}


def test_failed_file_is_retried_and_tagged_on_rerun(tmp_path):
    zips = tmp_path / "zips"
    zips.mkdir()
    (zips / "m.stl").write_text("mesh")
    out_csv = tmp_path / "tags.csv"
    mock_col = MagicMock()
    mock_col.query.return_value = {
        "documents": [["lore"]], "distances": [[0.3]], "metadatas": [[{"source": "u"}]],
    }
    mock_pc = MagicMock()
    mock_pc.get_or_create_collection.return_value = mock_col

    record = {f: "" for f in WH_FIELDS}
    record["faction"] = "Orks"
    record["tags"] = ["Waaagh"]

    client = MagicMock()
    with patch("tagging.PersistentClient", return_value=mock_pc), \
            patch("tagging.get_anthropic_client", return_value=client), \
            patch("tagging.time.sleep", lambda s: None), \
            patch("tagging.count_tokens", side_effect=lambda text, model=None: len(text) // 4):
        from tagging import run_tagging
        # Run 1: API fails -> nothing written
        client.messages.create.side_effect = RuntimeError("boom")
        run_tagging(str(zips), str(out_csv), str(tmp_path / "db"), None, "warhammer",
                    provider="anthropic")
        assert len(list(csv.reader(open(out_csv)))) == 1  # header only

        # Run 2 (resume): API succeeds -> the previously-failed file is retried
        client.messages.create.side_effect = None
        client.messages.create.return_value = _fake_response(record)
        run_tagging(str(zips), str(out_csv), str(tmp_path / "db"), None, "warhammer",
                    provider="anthropic")

    rows = list(csv.reader(open(out_csv)))
    assert len(rows) == 2  # header + exactly one row, no duplicate/blank
    assert rows[1][0] == "m.stl"
    assert dict(zip(rows[0], rows[1]))["faction"] == "Orks"


def test_resume_cleans_blank_and_duplicate_rows(tmp_path):
    # A CSV from an older run with a blank row (failed) and a stray duplicate;
    # resume should drop the blank (retry that file) and dedup the rest.
    out_csv = tmp_path / "tags.csv"
    header = ["filename"] + WH_FIELDS + ["tags"]
    blank_good = {f: "" for f in WH_FIELDS}
    with open(out_csv, "w", newline="") as f:
        import csv as _csv
        w = _csv.writer(f)
        w.writerow(header)
        w.writerow(["good.stl"] + ["Orks"] + [""] * (len(WH_FIELDS) - 1) + ["Waaagh"])
        w.writerow(["blank.stl"] + [""] * len(WH_FIELDS) + [""])          # failed -> retry
        w.writerow(["good.stl"] + ["Orks"] + [""] * (len(WH_FIELDS) - 1) + ["Waaagh"])  # dup

    zips = tmp_path / "zips"
    zips.mkdir()
    (zips / "blank.stl").write_text("mesh")   # the one to retry
    (zips / "good.stl").write_text("mesh")    # already done -> skip

    mock_col = MagicMock()
    mock_col.query.return_value = {
        "documents": [["lore"]], "distances": [[0.3]], "metadatas": [[{"source": "u"}]],
    }
    mock_pc = MagicMock()
    mock_pc.get_or_create_collection.return_value = mock_col
    record = {f: "" for f in WH_FIELDS}
    record["faction"] = "Tyranids"
    record["tags"] = ["Hive"]
    client = MagicMock()
    client.messages.create.return_value = _fake_response(record)

    with patch("tagging.PersistentClient", return_value=mock_pc), \
            patch("tagging.get_anthropic_client", return_value=client), \
            patch("tagging.count_tokens", side_effect=lambda text, model=None: len(text) // 4):
        from tagging import run_tagging
        run_tagging(str(zips), str(out_csv), str(tmp_path / "db"), None, "warhammer",
                    provider="anthropic")

    rows = list(csv.reader(open(out_csv)))
    by_name = {}
    for r in rows[1:]:
        by_name.setdefault(r[0], []).append(r)
    assert set(by_name) == {"good.stl", "blank.stl"}
    assert len(by_name["good.stl"]) == 1          # deduped
    assert len(by_name["blank.stl"]) == 1          # retried, now has content
    assert dict(zip(rows[0], by_name["blank.stl"][0]))["faction"] == "Tyranids"
    # good.stl was not re-tagged (skipped), so only blank.stl hit the API
    assert client.messages.create.call_count == 1


def test_fields_derived_from_schema(tmp_path):
    # Even if a preset's "fields" drifted, schema property order drives columns
    record = {f: "x" for f in WH_FIELDS}
    record["equipment"] = []
    record["tags"] = []
    out_csv, _ = _run_tag_anthropic(tmp_path, record)
    header = next(csv.reader(open(out_csv)))
    assert header == ["filename"] + WH_FIELDS + ["tags", "tagged_at"]


def test_resume_migrates_pre_timestamp_csv(tmp_path):
    # A CSV written before tagged_at existed (header lacks the trailing column)
    # must be migrated in place — kept rows get a backfilled empty timestamp, a
    # newly-tagged file gets a real one — without forcing a fresh file.
    out_csv = tmp_path / "tags.csv"
    old_header = ["filename"] + WH_FIELDS + ["tags"]  # no tagged_at
    with open(out_csv, "w", newline="") as f:
        import csv as _csv
        w = _csv.writer(f)
        w.writerow(old_header)
        w.writerow(["done.stl"] + ["Orks"] + [""] * (len(WH_FIELDS) - 1) + ["Waaagh"])

    zips = tmp_path / "zips"
    zips.mkdir()
    (zips / "done.stl").write_text("mesh")  # already tagged -> skip
    (zips / "new.stl").write_text("mesh")   # not yet tagged -> tag now

    mock_col = MagicMock()
    mock_col.query.return_value = {
        "documents": [["lore"]], "distances": [[0.3]], "metadatas": [[{"source": "u"}]],
    }
    mock_pc = MagicMock()
    mock_pc.get_or_create_collection.return_value = mock_col
    record = {f: "" for f in WH_FIELDS}
    record["faction"] = "Tyranids"
    record["tags"] = ["Hive"]
    client = MagicMock()
    client.messages.create.return_value = _fake_response(record)

    with patch("tagging.PersistentClient", return_value=mock_pc), \
            patch("tagging.get_anthropic_client", return_value=client), \
            patch("tagging.count_tokens", side_effect=lambda text, model=None: len(text) // 4):
        from tagging import run_tagging
        run_tagging(str(zips), str(out_csv), str(tmp_path / "db"), None, "warhammer",
                    provider="anthropic")

    rows = list(csv.reader(open(out_csv)))
    assert rows[0] == ["filename"] + WH_FIELDS + ["tags", "tagged_at"]
    by_name = {r[0]: dict(zip(rows[0], r)) for r in rows[1:]}
    assert set(by_name) == {"done.stl", "new.stl"}
    assert by_name["done.stl"]["tagged_at"] == ""        # migrated, backfilled blank
    assert by_name["new.stl"]["tagged_at"]               # freshly tagged, real timestamp
    assert client.messages.create.call_count == 1        # only new.stl hit the API
