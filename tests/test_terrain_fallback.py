"""Tests for the terrain collection fallback (ModelTagger2-y17): when the
primary collection field (faction) is blank but the model is clearly terrain,
assign it to a 'Terrain' collection instead of leaving it Unassigned."""

import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.append('src')

from manyfold_ingest import (
    _looks_like_terrain, _row_is_terrain, _tags_are_terrain,
    TERRAIN_COLLECTION, run_upload, reconcile_model_collections,
)


# --- detection helpers ----------------------------------------------------

def test_looks_like_terrain_by_model_type():
    assert _looks_like_terrain("Terrain", "some random name")
    assert not _looks_like_terrain("Infantry", "some random name")


def test_looks_like_terrain_by_cue_word():
    assert _looks_like_terrain("", "Imperial Bunker")
    assert _looks_like_terrain("", "Gothic Ruins Set")
    assert _looks_like_terrain("unknown", "Shipping Container")


def test_looks_like_terrain_whole_token_only():
    # 'wall' must not match inside 'wallet' / 'firewall'
    assert not _looks_like_terrain("Character", "Wallet Merchant")
    assert _looks_like_terrain("", "Defence Wall")


def test_row_is_terrain_uses_model_type_and_tags():
    assert _row_is_terrain({"filename": "x.zip", "model_type": "Terrain", "tags": ""})
    assert _row_is_terrain({"filename": "Bunker.zip", "model_type": "", "tags": "sci-fi"})
    assert not _row_is_terrain({"filename": "Ork Boy.zip", "model_type": "Infantry", "tags": "melee"})


def test_tags_are_terrain_reads_model_type_tag():
    assert _tags_are_terrain(["model_type: Terrain", "faction:"])
    assert _tags_are_terrain(["painted", "ruined tower"])
    assert not _tags_are_terrain(["model_type: Infantry", "faction: Orks"])


# --- run_upload fallback --------------------------------------------------

def _write_csv(tmp_path, rows):
    import csv
    header = ["filename", "faction", "unit", "model_type", "tags"]
    path = tmp_path / "tags.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({h: r.get(h, "") for h in header})
    return path


def _fake_client(models):
    client = MagicMock()
    client.list_models.return_value = models
    client.list_collections.return_value = []
    client.get_model.side_effect = lambda m: m
    client.create_collection.side_effect = lambda name: {"@id": f"/collections/{name}", "name": name}
    client.trigger_scan.return_value = True
    return client


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("MANYFOLD_API_URL", "https://mf.example")
    monkeypatch.setenv("MANYFOLD_API_TOKEN", "tok")
    monkeypatch.delenv("MANYFOLD_LIBRARY_PATH", raising=False)


def test_run_upload_blank_faction_terrain_gets_terrain_collection(tmp_path):
    csv_path = _write_csv(tmp_path, [
        {"filename": "Imperial Bunker.zip", "faction": "", "model_type": "Terrain"},
    ])
    client = _fake_client([{"id": 1, "name": "Imperial Bunker", "keywords": []}])
    with patch("manyfold_ingest.ManyfoldClient", return_value=client):
        run_upload(str(csv_path), collection_field="faction")
    attributes = client.update_model.call_args.args[1]
    assert attributes["isPartOf"]["@id"] == f"/collections/{TERRAIN_COLLECTION}"
    client.create_collection.assert_called_once_with(TERRAIN_COLLECTION)


def test_run_upload_faction_present_wins_over_terrain(tmp_path):
    # A real faction takes precedence; the fallback only fires when it's blank.
    csv_path = _write_csv(tmp_path, [
        {"filename": "Ork Bunker.zip", "faction": "Orks", "model_type": "Terrain"},
    ])
    client = _fake_client([{"id": 1, "name": "Ork Bunker", "keywords": []}])
    with patch("manyfold_ingest.ManyfoldClient", return_value=client):
        run_upload(str(csv_path), collection_field="faction")
    attributes = client.update_model.call_args.args[1]
    assert attributes["isPartOf"]["@id"] == "/collections/Orks"


def test_run_upload_blank_faction_non_terrain_stays_unassigned(tmp_path):
    csv_path = _write_csv(tmp_path, [
        {"filename": "Mystery Bit.stl", "faction": "", "model_type": "Infantry"},
    ])
    client = _fake_client([{"id": 1, "name": "Mystery Bit", "keywords": []}])
    with patch("manyfold_ingest.ManyfoldClient", return_value=client):
        run_upload(str(csv_path), collection_field="faction")
    # No collection assigned, no collection created.
    client.create_collection.assert_not_called()
    if client.update_model.called:
        assert "isPartOf" not in client.update_model.call_args.args[1]


# --- reconcile fallback ---------------------------------------------------

def _reconcile_client(models):
    client = MagicMock()
    client.list_models.return_value = models
    client.list_collections.return_value = []
    client.get_model.side_effect = lambda m: m
    client.create_collection.side_effect = lambda name: {"@id": f"/collections/{name}", "name": name}
    return client


def test_reconcile_terrain_fallback_from_tags():
    models = [{"id": 1, "name": "Ruined Tower",
               "keywords": ["model_type: Terrain", "gothic"]}]  # no faction tag
    client = _reconcile_client(models)
    stats = reconcile_model_collections(client, "faction")
    assert stats["assigned"] == 1
    client.create_collection.assert_called_once_with(TERRAIN_COLLECTION)


def test_reconcile_non_terrain_blank_still_skipped():
    models = [{"id": 1, "name": "Bit", "keywords": ["painted"]}]
    client = _reconcile_client(models)
    stats = reconcile_model_collections(client, "faction")
    assert stats["no_tag"] == 1
    assert stats["assigned"] == 0
    client.create_collection.assert_not_called()
