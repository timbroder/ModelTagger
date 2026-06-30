"""Tests for the Age of Sigmar / Old World mode (ModelTagger2-wua): a combined
fantasy faction preset that collects by faction and runs retrieval-disabled."""

import csv
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.append('src')

AOS_FIELDS = ["faction", "subfaction", "unit", "model_type", "role", "grand_alliance", "equipment"]
_CONFIG = Path(__file__).resolve().parent.parent / "config" / "tagging_presets.json"


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


# --- preset ---------------------------------------------------------------

def test_aos_preset_loads_combined_fantasy_vocab():
    presets = json.load(open(_CONFIG))
    assert "aos" in presets
    aos = presets["aos"]
    assert aos["fields"] == AOS_FIELDS
    assert aos["collection_field"] == "faction"   # collects by faction, like warhammer
    assert aos["retrieval"] is False              # no AoS corpus -> tag from names + knowledge
    factions = aos["schema"]["properties"]["faction"]["enum"]
    # Both AoS Grand Alliance names AND Old World army names are present (kept,
    # not canonicalized onto each other).
    assert "Soulblight Gravelords" in factions       # AoS
    assert "Vampire Counts" in factions              # Old World
    assert "Skaven" in factions
    assert "Tomb Kings" in factions
    grand = aos["schema"]["properties"]["grand_alliance"]["enum"]
    assert set(grand) == {"Order", "Chaos", "Death", "Destruction", "unknown"}


# --- upload: grand_alliance is owned, faction drives the collection -------

def test_grand_alliance_is_owned_namespace():
    from manyfold_ingest import OWNED_NAMESPACES
    assert "grand_alliance" in OWNED_NAMESPACES


def test_build_tags_namespaces_aos_fields():
    from manyfold_ingest import build_tags
    row = {
        "filename": "Prince Vhordrai.zip",
        "faction": "Soulblight Gravelords",
        "subfaction": "Legion of Blood",
        "unit": "Prince Vhordrai",
        "model_type": "Cavalry",
        "role": "Leader",
        "grand_alliance": "Death",
        "equipment": "Lance of Nagash",
        "tags": "Vampire, Zombie Dragon",
    }
    assert build_tags(row) == [
        "faction: Soulblight Gravelords",
        "subfaction: Legion of Blood",
        "unit: Prince Vhordrai",
        "model_type: Cavalry",
        "role: Leader",
        "equipment: Lance of Nagash",
        "grand_alliance: Death",
        "Vampire",
        "Zombie Dragon",
    ]


def _write_aos_csv(tmp_path, rows):
    header = ["filename"] + AOS_FIELDS + ["tags"]
    path = tmp_path / "tags-aos.csv"
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(r.get(h, "") for h in header))
    path.write_text("\n".join(lines) + "\n")
    return path


def _fake_client(models=None):
    client = MagicMock()
    client.list_models.return_value = models or []
    client.list_collections.return_value = []
    client.get_model.side_effect = lambda m: m
    client.create_collection.side_effect = lambda name: {"@id": "/collections/99", "name": name}
    client.trigger_scan.return_value = True
    return client


def test_run_upload_aos_collects_by_faction(tmp_path, monkeypatch):
    from manyfold_ingest import run_upload
    monkeypatch.setenv("MANYFOLD_API_URL", "https://mf.example")
    monkeypatch.setenv("MANYFOLD_API_TOKEN", "tok")
    monkeypatch.delenv("MANYFOLD_LIBRARY_PATH", raising=False)
    csv_path = _write_aos_csv(tmp_path, [
        {"filename": "Stormvermin.zip", "faction": "Skaven", "grand_alliance": "Chaos"},
    ])
    client = _fake_client(models=[{"id": 1, "name": "Stormvermin", "keywords": []}])

    with patch("manyfold_ingest.ManyfoldClient", return_value=client):
        run_upload(str(csv_path), collection_field="faction")

    attributes = client.update_model.call_args.args[1]
    assert "faction: Skaven" in attributes["keywords"]
    assert "grand_alliance: Chaos" in attributes["keywords"]
    client.create_collection.assert_called_once_with("Skaven")


# --- tagging: retrieval disabled ------------------------------------------

def test_run_tagging_aos_skips_retrieval(tmp_path):
    zips = tmp_path / "zips"
    zips.mkdir()
    (zips / "Nagash.stl").write_text("mesh")
    out_csv = tmp_path / "tags-aos.csv"

    record = {
        "faction": "Soulblight Gravelords", "subfaction": "", "unit": "Nagash",
        "model_type": "Monster", "role": "Leader", "grand_alliance": "Death",
        "equipment": [], "tags": ["Supreme Lord of Undeath"],
    }
    client = MagicMock()
    client.messages.create.return_value = _fake_response(record)

    with patch("tagging.PersistentClient") as pc, \
            patch("tagging.get_anthropic_client", return_value=client), \
            patch("tagging.count_tokens", side_effect=lambda text, model=None: len(text) // 4):
        from tagging import run_tagging
        run_tagging(str(zips), str(out_csv), None, None, "aos", provider="anthropic")

    pc.assert_not_called()  # never opened a vector DB
    rows = list(csv.reader(open(out_csv)))
    assert rows[0] == ["filename"] + AOS_FIELDS + ["tags", "tagged_at"]
    row = dict(zip(rows[0], rows[1]))
    assert row["faction"] == "Soulblight Gravelords"
    assert row["grand_alliance"] == "Death"
    assert row["unit"] == "Nagash"
