"""Tests for --faction (ModelTagger2-nwu): pin the faction column for a
single-faction library, keeping the model-determined fields."""

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


def _fake_response(record, in_tokens=100, out_tokens=40):
    block = MagicMock(); block.type = "text"; block.text = json.dumps(record)
    resp = MagicMock(); resp.content = [block]
    resp.usage.input_tokens = in_tokens; resp.usage.output_tokens = out_tokens
    return resp


def test_faction_override_forces_column_but_keeps_other_fields(tmp_path):
    zips = tmp_path / "zips"; zips.mkdir()
    (zips / "Barbgants.stl").write_text("mesh")
    out_csv = tmp_path / "tags.csv"

    # The model wrongly returns Orks — the override must win.
    record = {"faction": "Orks", "subfaction": "", "unit": "Barbgants",
              "model_type": "Monster", "role": "Troops", "allegiance": "Xenos",
              "equipment": [], "tags": ["Bio-titan"]}
    client = MagicMock()
    client.messages.create.return_value = _fake_response(record)

    with patch("tagging.PersistentClient") as pc, \
            patch("tagging.get_anthropic_client", return_value=client), \
            patch("tagging.count_tokens", side_effect=lambda text, model=None: len(text) // 4):
        from tagging import run_tagging
        # aos-style retrieval-disabled not needed; warhammer with a mocked DB:
        mock_col = MagicMock()
        mock_col.query.return_value = {"documents": [["lore"]], "distances": [[0.3]],
                                       "metadatas": [[{"source": "u"}]]}
        pc.return_value.get_or_create_collection.return_value = mock_col
        run_tagging(str(zips), str(out_csv), str(tmp_path / "db"), None, "warhammer",
                    provider="anthropic", faction="Tyranids")

    row = dict(zip(*[next(csv.reader(open(out_csv))), list(csv.reader(open(out_csv)))[1]]))
    assert row["faction"] == "Tyranids"        # forced, not Orks
    assert row["unit"] == "Barbgants"          # model field preserved
    assert row["tags"] == "Bio-titan"


def test_faction_override_fills_blank(tmp_path):
    zips = tmp_path / "zips"; zips.mkdir()
    (zips / "mystery.stl").write_text("mesh")
    out_csv = tmp_path / "tags.csv"

    record = {f: "" for f in WH_FIELDS}; record["tags"] = []   # model left everything blank
    client = MagicMock()
    client.messages.create.return_value = _fake_response(record)

    with patch("tagging.PersistentClient") as pc, \
            patch("tagging.get_anthropic_client", return_value=client), \
            patch("tagging.count_tokens", side_effect=lambda text, model=None: len(text) // 4):
        mock_col = MagicMock()
        mock_col.query.return_value = {"documents": [["lore"]], "distances": [[0.3]],
                                       "metadatas": [[{"source": "u"}]]}
        pc.return_value.get_or_create_collection.return_value = mock_col
        from tagging import run_tagging
        run_tagging(str(zips), str(out_csv), str(tmp_path / "db"), None, "warhammer",
                    provider="anthropic", faction="Tyranids")

    rows = list(csv.reader(open(out_csv)))
    assert dict(zip(rows[0], rows[1]))["faction"] == "Tyranids"


def test_faction_override_rejected_for_mode_without_faction_field(tmp_path):
    zips = tmp_path / "zips"; zips.mkdir()
    (zips / "x.stl").write_text("mesh")
    out_csv = tmp_path / "tags.csv"
    from tagging import run_tagging
    # terrain has no 'faction' field -> should abort clearly
    with pytest.raises(SystemExit, match="no faction field"):
        run_tagging(str(zips), str(out_csv), None, None, "terrain",
                    provider="anthropic", faction="Tyranids")
