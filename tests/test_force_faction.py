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


def _run_titan(tmp_path, stem, record, faction=None):
    """Tag one loose file via the anthropic provider with a mocked model record."""
    zips = tmp_path / "zips"; zips.mkdir()
    (zips / f"{stem}.stl").write_text("mesh")
    out_csv = tmp_path / "tags.csv"
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
                    provider="anthropic", faction=faction)
    rows = list(csv.reader(open(out_csv)))
    return dict(zip(rows[0], rows[1]))


def test_imperial_titan_pinned_to_adeptus_titanicus(tmp_path):
    # Model mis-picks a nearby Imperial faction for an Imperial Titan; the guard
    # pins Adeptus Titanicus while keeping the model's other fields.
    record = {"faction": "Adeptus Custodes", "subfaction": "Legio Ignatum",
              "unit": "Warlord Battle Titan", "model_type": "Titan",
              "role": "Lord of War", "allegiance": "Imperium",
              "equipment": [], "tags": ["God-Engine"]}
    row = _run_titan(tmp_path, "warlord", record)
    assert row["faction"] == "Adeptus Titanicus"     # pinned
    assert row["subfaction"] == "Legio Ignatum"      # model field preserved
    assert row["unit"] == "Warlord Battle Titan"


def test_xenos_titan_faction_not_overridden(tmp_path):
    # allegiance != Imperium -> the guard leaves the model's faction alone.
    record = {"faction": "Aeldari", "subfaction": "", "unit": "Phantom Titan",
              "model_type": "Titan", "role": "Lord of War", "allegiance": "Xenos",
              "equipment": [], "tags": []}
    row = _run_titan(tmp_path, "phantom", record)
    assert row["faction"] == "Aeldari"


def test_explicit_faction_flag_wins_over_titan_guard(tmp_path):
    # An explicit --faction pin takes precedence over the Titan guard.
    record = {"faction": "Adeptus Custodes", "subfaction": "", "unit": "Reaver Titan",
              "model_type": "Titan", "role": "Lord of War", "allegiance": "Imperium",
              "equipment": [], "tags": []}
    row = _run_titan(tmp_path, "reaver", record, faction="Adeptus Mechanicus")
    assert row["faction"] == "Adeptus Mechanicus"
