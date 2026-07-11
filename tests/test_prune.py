"""Tests for the prune cleanup (ModelTagger2-01d): delete Manyfold models by an
exact-name list, safely."""

import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.append('src')

from manyfold import ManyfoldClient
from manyfold_ingest import run_prune


# --- client delete --------------------------------------------------------

def test_delete_model_issues_delete_to_resource_path():
    client = ManyfoldClient("https://mf.example", token="t")
    resp = MagicMock(); resp.ok = True
    with patch.object(client, "_request", return_value=resp) as req:
        client.delete_model({"@id": "/models/42"})
    method, path = req.call_args.args[:2]
    assert method == "DELETE"
    assert path == "/models/42"


def test_delete_model_raises_on_error():
    from manyfold import ManyfoldError
    client = ManyfoldClient("https://mf.example", token="t")
    resp = MagicMock(); resp.ok = False; resp.status_code = 500; resp.text = "boom"
    with patch.object(client, "_request", return_value=resp):
        with pytest.raises(ManyfoldError):
            client.delete_model({"@id": "/models/42"})


# --- run_prune ------------------------------------------------------------

def _names_file(tmp_path, names):
    p = tmp_path / "names.txt"
    p.write_text("\n".join(names) + "\n")
    return str(p)


def _client(models):
    client = MagicMock()
    client.list_models.return_value = models
    return client


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("MANYFOLD_API_URL", "https://mf.example")
    monkeypatch.setenv("MANYFOLD_API_TOKEN", "tok")


def test_prune_deletes_exact_matches(tmp_path):
    models = [
        {"@id": "/models/1", "title": "BA GRAVIS 1"},
        {"@id": "/models/2", "title": "Keeper Model"},
    ]
    client = _client(models)
    names = _names_file(tmp_path, ["BA GRAVIS 1"])
    with patch("manyfold_ingest.ManyfoldClient", return_value=client):
        stats = run_prune(names)
    assert stats["deleted"] == 1
    client.delete_model.assert_called_once_with(models[0])


def test_prune_case_insensitive_match(tmp_path):
    models = [{"@id": "/models/1", "title": "BA GRAVIS 1"}]
    client = _client(models)
    names = _names_file(tmp_path, ["ba gravis 1"])
    with patch("manyfold_ingest.ManyfoldClient", return_value=client):
        stats = run_prune(names)
    assert stats["deleted"] == 1


def test_prune_skips_not_found(tmp_path):
    client = _client([{"@id": "/models/1", "title": "Something Else"}])
    names = _names_file(tmp_path, ["Ghost Model"])
    with patch("manyfold_ingest.ManyfoldClient", return_value=client):
        stats = run_prune(names)
    assert stats == {"deleted": 0, "not_found": 1, "ambiguous": 0, "held_generic": 0, "errors": 0}
    client.delete_model.assert_not_called()


def test_prune_skips_ambiguous_duplicate_names(tmp_path):
    # Two models share a name -> never guess which to delete.
    models = [
        {"@id": "/models/1", "title": "Warhound Face"},
        {"@id": "/models/2", "title": "Warhound Face"},
    ]
    client = _client(models)
    names = _names_file(tmp_path, ["Warhound Face"])
    with patch("manyfold_ingest.ManyfoldClient", return_value=client):
        stats = run_prune(names)
    assert stats["ambiguous"] == 1
    assert stats["deleted"] == 0
    client.delete_model.assert_not_called()


def test_prune_holds_generic_names_by_default(tmp_path):
    models = [{"@id": "/models/1", "title": "axe"}, {"@id": "/models/2", "title": "Base"}]
    client = _client(models)
    names = _names_file(tmp_path, ["axe", "Base"])
    with patch("manyfold_ingest.ManyfoldClient", return_value=client):
        stats = run_prune(names)
    assert stats["held_generic"] == 2
    assert stats["deleted"] == 0
    client.delete_model.assert_not_called()


def test_prune_allow_generic_includes_them(tmp_path):
    models = [{"@id": "/models/1", "title": "axe"}]
    client = _client(models)
    names = _names_file(tmp_path, ["axe"])
    with patch("manyfold_ingest.ManyfoldClient", return_value=client):
        stats = run_prune(names, allow_generic=True)
    assert stats["deleted"] == 1
    client.delete_model.assert_called_once()


def test_prune_dry_run_deletes_nothing(tmp_path):
    models = [{"@id": "/models/1", "title": "BA GRAVIS 1"}]
    client = _client(models)
    names = _names_file(tmp_path, ["BA GRAVIS 1"])
    with patch("manyfold_ingest.ManyfoldClient", return_value=client):
        stats = run_prune(names, dry_run=True)
    assert stats["deleted"] == 1        # counted as "would delete"
    client.delete_model.assert_not_called()


def test_prune_limit_caps_deletions(tmp_path):
    models = [{"@id": f"/models/{i}", "title": f"Frag {i}"} for i in range(5)]
    client = _client(models)
    names = _names_file(tmp_path, [f"Frag {i}" for i in range(5)])
    with patch("manyfold_ingest.ManyfoldClient", return_value=client):
        stats = run_prune(names, limit=2)
    assert stats["deleted"] == 2
    assert client.delete_model.call_count == 2


def test_prune_matches_on_name_or_title(tmp_path):
    # Some models expose 'name' rather than 'title'.
    models = [{"@id": "/models/1", "name": "Orphan Render"}]
    client = _client(models)
    names = _names_file(tmp_path, ["Orphan Render"])
    with patch("manyfold_ingest.ManyfoldClient", return_value=client):
        stats = run_prune(names)
    assert stats["deleted"] == 1
