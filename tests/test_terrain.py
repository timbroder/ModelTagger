"""Tests for the terrain mode (ModelTagger2-78x): terrain taxonomy preset,
the per-mode collection field, and the skip-retrieval tagging path."""

import csv
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.append('src')

TERRAIN_FIELDS = ["terrain_type", "setting", "faction_theme", "function", "modular"]
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

def test_terrain_preset_loads_with_schema_and_per_mode_fields():
    presets = json.load(open(_CONFIG))
    assert "terrain" in presets
    terrain = presets["terrain"]
    # Schema drives the CSV columns; collection groups by terrain_type, not faction.
    assert terrain["fields"] == TERRAIN_FIELDS
    assert terrain["collection_field"] == "terrain_type"
    assert terrain["retrieval"] is False
    schema_fields = [k for k in terrain["schema"]["properties"] if k != "tags"]
    assert schema_fields == TERRAIN_FIELDS
    # A representative terrain_type enum value is present (not a 40K faction).
    assert "fortification" in terrain["schema"]["properties"]["terrain_type"]["enum"]


# --- upload: collection comes from the per-mode field ---------------------

def test_build_tags_namespaces_terrain_fields():
    from manyfold_ingest import build_tags
    row = {
        "filename": "bunker.zip",
        "terrain_type": "bunker",
        "setting": "gothic-imperial",
        "faction_theme": "Astra Militarum",
        "function": "cover/defensible",
        "modular": "no",
        "tags": "Sandbags, Damaged",
    }
    assert build_tags(row) == [
        "terrain_type: bunker",
        "setting: gothic-imperial",
        "faction_theme: Astra Militarum",
        "function: cover/defensible",
        "modular: no",
        "Sandbags",
        "Damaged",
    ]


def test_merge_tags_treats_terrain_fields_as_owned():
    from manyfold_ingest import merge_tags, OWNED_NAMESPACES
    for ns in TERRAIN_FIELDS:
        assert ns in OWNED_NAMESPACES
    # A stale terrain_type tag is replaced; a manual tag is kept.
    existing = ["terrain_type: ruin", "weathered"]
    new = ["terrain_type: bunker"]
    assert merge_tags(existing, new) == ["weathered", "terrain_type: bunker"]


def _write_terrain_csv(tmp_path, rows):
    header = ["filename"] + TERRAIN_FIELDS + ["tags"]
    path = tmp_path / "tags-terrain.csv"
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(r.get(h, "") for h in header))
    path.write_text("\n".join(lines) + "\n")
    return path


def _fake_client(models=None, collections=None):
    client = MagicMock()
    client.list_models.return_value = models or []
    client.list_collections.return_value = collections or []
    client.get_model.side_effect = lambda m: m
    client.create_collection.side_effect = lambda name: {"@id": "/collections/99", "name": name}
    client.trigger_scan.return_value = True
    return client


def test_run_upload_collects_by_terrain_type(tmp_path, monkeypatch):
    from manyfold_ingest import run_upload
    monkeypatch.setenv("MANYFOLD_API_URL", "https://mf.example")
    monkeypatch.setenv("MANYFOLD_API_TOKEN", "tok")
    monkeypatch.delenv("MANYFOLD_LIBRARY_PATH", raising=False)
    csv_path = _write_terrain_csv(tmp_path, [
        {"filename": "Imperial Bunker.zip", "terrain_type": "bunker",
         "setting": "gothic-imperial", "faction_theme": "none"},
    ])
    model = {"id": 1, "name": "Imperial Bunker", "keywords": []}
    client = _fake_client(models=[model])

    with patch("manyfold_ingest.ManyfoldClient", return_value=client):
        run_upload(str(csv_path), collection_field="terrain_type")

    attributes = client.update_model.call_args.args[1]
    assert "terrain_type: bunker" in attributes["keywords"]
    assert attributes["isPartOf"] == {"@id": "/collections/99", "@type": "Collection"}
    # Collection is named after the terrain_type, NOT a faction.
    client.create_collection.assert_called_once_with("bunker")


# --- tagging: retrieval is skipped entirely -------------------------------

def test_run_tagging_terrain_skips_retrieval(tmp_path):
    """Terrain mode must never open a vector DB or query a collection — it tags
    from file names + model knowledge alone."""
    zips = tmp_path / "zips"
    zips.mkdir()
    (zips / "Sanctum Imperialis.stl").write_text("mesh")
    out_csv = tmp_path / "tags-terrain.csv"

    record = {
        "terrain_type": "building", "setting": "gothic-imperial",
        "faction_theme": "none", "function": "los-blocker", "modular": "no",
        "tags": ["Cathedral", "Statues"],
    }
    client = MagicMock()
    client.messages.create.return_value = _fake_response(record)

    with patch("tagging.PersistentClient") as pc, \
            patch("tagging.get_anthropic_client", return_value=client), \
            patch("tagging.count_tokens", side_effect=lambda text, model=None: len(text) // 4):
        from tagging import run_tagging
        run_tagging(str(zips), str(out_csv), None, None, "terrain",
                    provider="anthropic")

    # No vector DB was ever constructed.
    pc.assert_not_called()

    rows = list(csv.reader(open(out_csv)))
    assert rows[0] == ["filename"] + TERRAIN_FIELDS + ["tags", "tagged_at"]
    row = dict(zip(rows[0], rows[1]))
    assert row["terrain_type"] == "building"
    assert row["setting"] == "gothic-imperial"
    assert row["tags"] == "Cathedral, Statues"

    # The prompt carried no "Lore context" section when retrieval is disabled.
    sent_prompt = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "Lore context follows" not in sent_prompt
