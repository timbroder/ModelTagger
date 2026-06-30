"""Tests for the collection reconciliation pass (ModelTagger2-77e): assign
collections from each model's OWN namespaced tag, not CSV name-matching."""

import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.append('src')

from manyfold_ingest import _tag_value, reconcile_model_collections, run_upload


# --- _tag_value -----------------------------------------------------------

def test_tag_value_extracts_namespaced_value():
    tags = ["painted", "faction: Space Marines", "unit: Techmarine"]
    assert _tag_value(tags, "faction") == "Space Marines"
    assert _tag_value(tags, "unit") == "Techmarine"
    assert _tag_value(tags, "role") == ""


def test_tag_value_is_case_insensitive():
    assert _tag_value(["Faction: Orks"], "faction") == "Orks"


def test_tag_value_does_not_confuse_sibling_namespace():
    # 'faction' must NOT pick up 'faction_theme: ...'
    tags = ["faction_theme: Necrons"]
    assert _tag_value(tags, "faction") == ""
    assert _tag_value(tags, "faction_theme") == "Necrons"


# --- reconcile pass -------------------------------------------------------

def _client(models, collections=None):
    client = MagicMock()
    client.list_models.return_value = models
    client.list_collections.return_value = collections or []
    # In these tests the list item already carries keywords + isPartOf.
    client.get_model.side_effect = lambda m: m
    client.create_collection.side_effect = lambda name: {"@id": f"/collections/{name}", "name": name}
    return client


def test_reconcile_assigns_collection_from_tag():
    models = [{"id": 1, "name": "Ork Boy", "keywords": ["faction: Orks", "painted"]}]
    client = _client(models)

    stats = reconcile_model_collections(client, "faction")

    assert stats["assigned"] == 1
    client.create_collection.assert_called_once_with("Orks")
    args = client.update_model.call_args.args
    assert args[1]["isPartOf"] == {"@id": "/collections/Orks", "@type": "Collection"}


def test_reconcile_keeps_manual_collection():
    models = [{
        "id": 1, "name": "Ork Boy", "keywords": ["faction: Orks"],
        "isPartOf": {"@id": "/collections/7", "@type": "Collection"},
    }]
    client = _client(models)

    stats = reconcile_model_collections(client, "faction")

    assert stats["already_assigned"] == 1
    assert stats["assigned"] == 0
    client.update_model.assert_not_called()


def test_reconcile_skips_models_without_the_tag():
    models = [{"id": 1, "name": "Mystery", "keywords": ["painted"]}]
    client = _client(models)

    stats = reconcile_model_collections(client, "faction")

    assert stats["no_tag"] == 1
    assert stats["assigned"] == 0
    client.update_model.assert_not_called()


def test_reconcile_reuses_existing_collection_without_recreating():
    models = [
        {"id": 1, "name": "A", "keywords": ["faction: Orks"]},
        {"id": 2, "name": "B", "keywords": ["faction: Orks"]},
    ]
    client = _client(models, collections=[{"@id": "/collections/5", "name": "Orks"}])

    stats = reconcile_model_collections(client, "faction")

    assert stats["assigned"] == 2
    client.create_collection.assert_not_called()  # cached / pre-existing
    for call in client.update_model.call_args_list:
        assert call.args[1]["isPartOf"]["@id"] == "/collections/5"


def test_reconcile_uses_per_mode_collection_field():
    # Terrain groups by terrain_type, not faction.
    models = [{"id": 1, "name": "Bunker", "keywords": ["terrain_type: bunker"]}]
    client = _client(models)

    stats = reconcile_model_collections(client, "terrain_type")

    assert stats["assigned"] == 1
    client.create_collection.assert_called_once_with("bunker")


def test_reconcile_dry_run_writes_nothing():
    models = [{"id": 1, "name": "Ork Boy", "keywords": ["faction: Orks"]}]
    client = _client(models)

    reconcile_model_collections(client, "faction", dry_run=True)

    client.create_collection.assert_not_called()
    client.update_model.assert_not_called()


# --- run_upload --reconcile-collections wiring ----------------------------

def test_run_upload_reconcile_flag_skips_csv(tmp_path, monkeypatch):
    monkeypatch.setenv("MANYFOLD_API_URL", "https://mf.example")
    monkeypatch.setenv("MANYFOLD_API_TOKEN", "tok")
    models = [{"id": 1, "name": "Ork Boy", "keywords": ["faction: Orks"]}]
    client = _client(models)

    # No CSV file exists at this path — reconcile mode must not try to read it.
    missing_csv = str(tmp_path / "does-not-exist.csv")
    with patch("manyfold_ingest.ManyfoldClient", return_value=client):
        run_upload(missing_csv, collection_field="faction", reconcile_collections=True)

    client.update_model.assert_called_once()
    assert client.update_model.call_args.args[1]["isPartOf"]["@id"] == "/collections/Orks"
