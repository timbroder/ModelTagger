import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.append('src')

from manyfold import ManyfoldClient, ManyfoldError, model_tags, _items, _next_link
from manyfold_ingest import (
    build_tags,
    merge_tags,
    normalize_name,
    match_model,
    _model_dir_name,
    stage_into_library,
    run_upload,
)


# --- pure helpers ---------------------------------------------------------

def test_build_tags_namespaces_fields_and_splits_lists():
    row = {
        "filename": "x.zip",
        "faction": "Space Marines",
        "subfaction": "Wolfspear",
        "unit": "Techmarine",
        "model_type": "",
        "role": "Elites",
        "allegiance": "Imperium",
        "equipment": "Servo-arm, Power Armour",
        "tags": "Primaris, Adeptus Mechanicus",
    }
    assert build_tags(row) == [
        "faction: Space Marines",
        "subfaction: Wolfspear",
        "unit: Techmarine",
        "role: Elites",
        "allegiance: Imperium",
        "equipment: Servo-arm",
        "equipment: Power Armour",
        "Primaris",
        "Adeptus Mechanicus",
    ]


def test_merge_tags_updates_owned_keeps_manual():
    existing = [
        "faction: Orks",            # owned, stale -> replaced
        "painted",                  # manual -> kept
        "scale: 32mm",              # not an owned namespace -> kept
        "Primaris",                 # plain, also in new -> no duplicate
    ]
    new = ["faction: Space Marines", "unit: Techmarine", "Primaris"]
    assert merge_tags(existing, new) == [
        "painted", "scale: 32mm", "Primaris",
        "faction: Space Marines", "unit: Techmarine",
    ]


def test_normalize_and_match():
    models = {
        normalize_name("Wolfspear Techmarine"): {"id": 1, "name": "Wolfspear Techmarine"},
        normalize_name("Ork Warboss"): {"id": 2, "name": "Ork Warboss"},
    }
    assert match_model("Wolfspear+Techmarine.zip", models)["id"] == 1
    assert match_model("wolfspear_techmarine.stl", models)["id"] == 1
    # Near-match within ratio
    assert match_model("Wolfspear Techmarines.zip", models)["id"] == 1
    # Clearly different name must NOT match
    assert match_model("Sister Superior.zip", models) is None


def test_model_dir_name_dedupes_doubled_words():
    # Vendor "Name_Name+variant" filenames must not double the title/folder
    assert _model_dir_name("Sister Superior_Sister+Superior.zip") == "Sister Superior"
    # Dedupe is case-insensitive, keeping the first occurrence's casing
    assert _model_dir_name("Khan_KHAN.stl") == "Khan"
    # Non-doubled names are unchanged
    assert _model_dir_name("Wolfspear+Techmarine.zip") == "Wolfspear Techmarine"
    assert _model_dir_name("Aveline.stl") == "Aveline"


def test_staged_dir_names_disambiguates_basename_collisions():
    from manyfold_ingest import _staged_dir_names
    names = _staged_dir_names([
        "General/Triarch Praetorians.zip",
        "Finished scans/Triarch Praetorians.zip",
        "Necrons/Lokhust.stl",  # unique basename -> stays clean
        "",                      # ignored
    ])
    # colliding basenames disambiguated by their relative parent
    assert names["General/Triarch Praetorians.zip"] == "Triarch Praetorians (General)"
    assert names["Finished scans/Triarch Praetorians.zip"] == "Triarch Praetorians (Finished scans)"
    # distinct sources -> distinct folders (neither silently dropped)
    assert (names["General/Triarch Praetorians.zip"]
            != names["Finished scans/Triarch Praetorians.zip"])
    # non-colliding name is untouched
    assert names["Necrons/Lokhust.stl"] == "Lokhust"


def test_model_dir_name_strips_multipart_suffix():
    assert _model_dir_name("Krieg.part01.rar") == "Krieg"
    assert _model_dir_name("Lesionaries (Supported).part2.rar") == "Lesionaries Supported"
    # split 7z set: the .7z.NNN collapses to the base name
    assert _model_dir_name("Grey Knights.7z.001") == "Grey Knights"
    # "Part 3" (space, single .rar) is a name, not a volume marker
    assert _model_dir_name("Tyranid Warriors Part 3.rar") == "Tyranid Warriors Part 3"


def test_match_dedupes_doubled_filename():
    # Vendor "Name_Name+variant" pattern must match Manyfold's shorter name
    models = {normalize_name("Sister Superior"): {"id": 9, "name": "Sister Superior"}}
    assert match_model("Sister Superior_Sister+Superior.zip", models)["id"] == 9


def test_model_tags_shapes():
    assert model_tags({"tags": ["a", "b"]}) == ["a", "b"]
    assert model_tags({"tags": [{"name": "a"}, {"name": "b"}]}) == ["a", "b"]
    assert model_tags({"tag_list": ["x"]}) == ["x"]
    assert model_tags({}) == []


def test_pagination_helpers():
    assert _items({"member": [1, 2]}) == [1, 2]
    assert _items({"@graph": [1]}) == [1]
    assert _items([3]) == [3]
    assert _next_link({"view": {"next": "/models?page=2"}}) == "/models?page=2"
    assert _next_link({"next": "/models?page=3"}) == "/models?page=3"
    assert _next_link({}) is None


# --- client ---------------------------------------------------------------

def _resp(status=200, body=None):
    r = MagicMock()
    r.status_code = status
    r.ok = 200 <= status < 300
    r.json.return_value = body if body is not None else {}
    r.text = json.dumps(body or {})
    return r


def test_base_url_strips_api_docs_suffix():
    # Pasting the docs URL (…/api) must still resolve resources + oauth at root
    assert ManyfoldClient("https://mf.example/api").base_url == "https://mf.example"
    assert ManyfoldClient("https://mf.example/api/v0").base_url == "https://mf.example"
    assert ManyfoldClient("https://mf.example/api/").base_url == "https://mf.example"
    assert ManyfoldClient("https://mf.example/").base_url == "https://mf.example"


def test_oauth_uses_configured_scope_and_root_token_url():
    client = ManyfoldClient("https://mf.example/api", client_id="cid",
                            client_secret="sec", scopes="read write", min_interval=0)
    with patch("manyfold.requests.post", return_value=_resp(200, {"access_token": "T"})) as mock_post:
        tok = client._ensure_token()
    assert tok == "T"
    assert mock_post.call_args.args[0] == "https://mf.example/oauth/token"  # /api stripped
    assert mock_post.call_args.kwargs["data"]["scope"] == "read write"


def test_oauth_default_scope_includes_public():
    # GET /models requires ["public","read"]; the default must include public
    client = ManyfoldClient("https://mf.example", client_id="c", client_secret="s", min_interval=0)
    with patch("manyfold.requests.post", return_value=_resp(200, {"access_token": "T"})) as mock_post:
        client._ensure_token()
    assert mock_post.call_args.kwargs["data"]["scope"] == "public read write"


def test_oauth_404_gives_actionable_error():
    client = ManyfoldClient("https://mf.example/api", client_id="c", client_secret="s", min_interval=0)
    with patch("manyfold.requests.post", return_value=_resp(404, {})):
        try:
            client._ensure_token()
            assert False, "expected ManyfoldError"
        except ManyfoldError as e:
            assert "root" in str(e).lower()  # hints to use the instance root URL


def test_oauth_invalid_scope_gives_actionable_error():
    bad = MagicMock(); bad.ok = False; bad.status_code = 400
    bad.text = '{"error":"invalid_scope","error_description":"..."}'
    client = ManyfoldClient("https://mf.example", client_id="c", client_secret="s",
                            scopes="read write delete", min_interval=0)
    with patch("manyfold.requests.post", return_value=bad):
        try:
            client._ensure_token()
            assert False, "expected ManyfoldError"
        except ManyfoldError as e:
            assert "scope" in str(e).lower()


def test_client_paginates_and_authenticates():
    client = ManyfoldClient("https://mf.example", token="tok", min_interval=0)
    pages = [
        _resp(200, {"member": [{"id": 1}], "next": "/models?page=2"}),
        _resp(200, {"member": [{"id": 2}]}),
    ]
    with patch("manyfold.requests.request", side_effect=pages) as mock_req:
        models = client.list_models()
    assert [m["id"] for m in models] == [1, 2]
    headers = mock_req.call_args_list[0].kwargs["headers"]
    assert headers["Authorization"] == "Bearer tok"


def test_client_refreshes_token_on_401(monkeypatch):
    # Long runs outlive the OAuth token TTL; a 401 must trigger a re-auth and
    # one retry with the fresh token rather than bubbling up as an error.
    monkeypatch.setattr("manyfold.time.sleep", lambda s: None)
    client = ManyfoldClient("https://mf.example", client_id="c", client_secret="s", min_interval=0)
    with patch("manyfold.requests.post",
               side_effect=[_resp(200, {"access_token": "T1"}),
                            _resp(200, {"access_token": "T2"})]) as mock_post, \
            patch("manyfold.requests.request",
                  side_effect=[_resp(401), _resp(200, {"member": []})]) as mock_req:
        assert client.list_models() == []
    assert mock_post.call_count == 2  # initial token + one refresh
    assert mock_req.call_count == 2   # 401, then retried
    # the retry carried the refreshed token
    assert mock_req.call_args_list[1].kwargs["headers"]["Authorization"] == "Bearer T2"


def test_client_does_not_loop_on_persistent_401(monkeypatch):
    # With no OAuth creds (static token) a 401 can't be refreshed; it returns
    # rather than retrying forever.
    monkeypatch.setattr("manyfold.time.sleep", lambda s: None)
    client = ManyfoldClient("https://mf.example", token="tok", min_interval=0)
    with patch("manyfold.requests.request", return_value=_resp(401, {})) as mock_req:
        try:
            client.list_models()
        except ManyfoldError:
            pass
    assert mock_req.call_count == 1  # no re-auth attempts


def test_client_retries_on_429(monkeypatch):
    monkeypatch.setattr("manyfold.time.sleep", lambda s: None)
    client = ManyfoldClient("https://mf.example", token="tok", min_interval=0)
    with patch("manyfold.requests.request", side_effect=[_resp(429), _resp(200, {"member": []})]):
        assert client.list_models() == []


def test_client_update_model_sends_flat_vendor_body():
    client = ManyfoldClient("https://mf.example", token="tok", min_interval=0)
    with patch("manyfold.requests.request", return_value=_resp(200)) as mock_req:
        client.update_model({"@id": "http://localhost:3214/models/5"},
                            {"keywords": ["a"], "isPartOf": {"@id": "/collections/1"}})
    c = mock_req.call_args.kwargs
    # flat JSON-LD body, vendor media type, and the localhost @id rewritten to our host
    assert json.loads(c["data"]) == {"keywords": ["a"], "isPartOf": {"@id": "/collections/1"}}
    assert c["headers"]["Content-Type"] == "application/vnd.manyfold.v0+json"
    assert mock_req.call_args.args[1] == "https://mf.example/models/5"


def test_request_rewrites_internal_host():
    client = ManyfoldClient("https://mf.example", token="tok", min_interval=0)
    with patch("manyfold.requests.request", return_value=_resp(200)) as mock_req:
        client._request("GET", "http://localhost:3214/models/abc?page=2")
    assert mock_req.call_args.args[1] == "https://mf.example/models/abc?page=2"


# --- staging --------------------------------------------------------------

def test_stage_into_library_loose_file(tmp_path):
    archive = tmp_path / "Wolfspear+Techmarine.stl"
    archive.write_text("mesh")
    library = tmp_path / "library"

    dest = stage_into_library(archive, library, ["faction: Space Marines"])

    assert dest == library / "Wolfspear Techmarine"
    assert (dest / "Wolfspear+Techmarine.stl").exists()
    pkg = json.loads((dest / "datapackage.json").read_text())
    assert pkg["title"] == "Wolfspear Techmarine"
    assert pkg["keywords"] == ["faction: Space Marines"]

    # Re-staging is a no-op (resume)
    assert stage_into_library(archive, library, []) == dest


def test_stage_into_library_loose_slicer_file(tmp_path):
    # A loose ChiTuBox project is a supported format now, so it stages like any
    # other loose model file.
    archive = tmp_path / "Marine.ctb"
    archive.write_text("sliced")
    library = tmp_path / "library"

    dest = stage_into_library(archive, library, ["faction: Space Marines"])

    assert dest == library / "Marine"
    assert (dest / "Marine.ctb").exists()
    pkg = json.loads((dest / "datapackage.json").read_text())
    assert pkg["keywords"] == ["faction: Space Marines"]


def test_stage_into_library_flattens_nested_archive(tmp_path, monkeypatch):
    # Manyfold makes one model per subfolder, so a staged archive must be
    # flattened into a single flat folder => one model per zip.
    archive = tmp_path / "Kit.zip"
    archive.write_text("zip")
    library = tmp_path / "library"

    def fake_extract(src, outdir):
        out = Path(outdir)
        (out / "Supported").mkdir(parents=True)
        (out / "Large Printers").mkdir(parents=True)
        (out / "Supported" / "body.stl").write_text("a")
        (out / "Large Printers" / "body.stl").write_text("b")  # same basename, diff folder
        (out / "readme.txt").write_text("c")                   # already at root

    monkeypatch.setattr("manyfold_ingest.patoolib.extract_archive", fake_extract)
    dest = stage_into_library(archive, library, ["faction: Orks"])

    assert dest == library / "Kit"
    # no subdirectories survive: Manyfold sees one model
    assert [p for p in dest.rglob("*") if p.is_dir()] == []
    files = {p.name for p in dest.iterdir() if p.is_file()}
    # nested parts flattened with collision-safe names; root file & datapackage kept
    assert files == {
        "Supported_body.stl", "Large Printers_body.stl", "readme.txt", "datapackage.json",
    }
    assert (dest / "Supported_body.stl").read_text() == "a"
    assert (dest / "Large Printers_body.stl").read_text() == "b"
    pkg = json.loads((dest / "datapackage.json").read_text())
    assert pkg["keywords"] == ["faction: Orks"]


def test_stage_into_library_caps_overlong_flattened_names(tmp_path, monkeypatch):
    # A deeply nested member would join into a >255-byte filename and raise
    # ENAMETOOLONG; the name must be capped (hashed) while staying staged.
    archive = tmp_path / "Kit.zip"
    archive.write_text("zip")
    library = tmp_path / "library"

    deep = Path("A" * 60, "B" * 60, "C" * 60, "D" * 60)  # join ~243 chars

    def fake_extract(src, outdir):
        target = Path(outdir) / deep
        target.mkdir(parents=True)
        (target / "promo.png").write_text("x")

    monkeypatch.setattr("manyfold_ingest.patoolib.extract_archive", fake_extract)
    dest = stage_into_library(archive, library, ["faction: Orks"])

    files = [p for p in dest.iterdir() if p.is_file() and p.name != "datapackage.json"]
    assert len(files) == 1
    f = files[0]
    assert len(f.name.encode("utf-8")) <= 200  # within the filesystem limit
    assert f.suffix == ".png"                   # extension preserved
    assert f.read_text() == "x"                  # content staged, no error


# --- run_upload flows -----------------------------------------------------

def _write_csv(tmp_path, rows):
    header = ["filename", "faction", "subfaction", "unit", "model_type",
              "role", "allegiance", "equipment", "tags"]
    path = tmp_path / "tags.csv"
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(r.get(h, "") for h in header))
    path.write_text("\n".join(lines) + "\n")
    return path


def _fake_client(models=None, collections=None):
    client = MagicMock()
    client.list_models.return_value = models or []
    client.list_collections.return_value = collections or []
    client.get_model.side_effect = lambda m: m  # list item == detail in tests
    client.create_collection.side_effect = lambda name: {"@id": "/collections/99", "name": name}
    client.trigger_scan.return_value = True
    return client


def test_run_upload_updates_existing_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MANYFOLD_API_URL", "https://mf.example")
    monkeypatch.setenv("MANYFOLD_API_TOKEN", "tok")
    monkeypatch.delenv("MANYFOLD_LIBRARY_PATH", raising=False)
    csv_path = _write_csv(tmp_path, [
        {"filename": "Wolfspear+Techmarine.zip", "faction": "Space Marines", "unit": "Techmarine"},
    ])
    model = {"id": 1, "name": "Wolfspear Techmarine", "keywords": ["painted"]}
    client = _fake_client(models=[model])

    with patch("manyfold_ingest.ManyfoldClient", return_value=client):
        run_upload(str(csv_path))

    attributes = client.update_model.call_args.args[1]
    assert "painted" in attributes["keywords"]                 # manual tag preserved
    assert "faction: Space Marines" in attributes["keywords"]
    assert attributes["isPartOf"] == {"@id": "/collections/99", "@type": "Collection"}
    client.create_collection.assert_called_once_with("Space Marines")


def test_run_upload_respects_existing_collection_assignment(tmp_path, monkeypatch):
    monkeypatch.setenv("MANYFOLD_API_URL", "https://mf.example")
    monkeypatch.setenv("MANYFOLD_API_TOKEN", "tok")
    csv_path = _write_csv(tmp_path, [
        {"filename": "Wolfspear+Techmarine.zip", "faction": "Space Marines"},
    ])
    model = {"id": 1, "name": "Wolfspear Techmarine", "keywords": [],
             "isPartOf": {"@id": "/collections/7", "@type": "Collection"}}
    client = _fake_client(models=[model])

    with patch("manyfold_ingest.ManyfoldClient", return_value=client):
        run_upload(str(csv_path))

    attributes = client.update_model.call_args.args[1]
    assert "isPartOf" not in attributes  # manual collection placement preserved


def test_run_upload_stages_missing_models_and_scans(tmp_path, monkeypatch):
    monkeypatch.setenv("MANYFOLD_API_URL", "https://mf.example")
    monkeypatch.setenv("MANYFOLD_API_TOKEN", "tok")
    zips = tmp_path / "zips"
    zips.mkdir()
    (zips / "Sister Superior.stl").write_text("mesh")
    library = tmp_path / "library"
    csv_path = _write_csv(tmp_path, [
        {"filename": "Sister Superior.stl", "faction": "Adepta Sororitas"},
    ])
    client = _fake_client()

    with patch("manyfold_ingest.ManyfoldClient", return_value=client):
        run_upload(str(csv_path), zips_dir=str(zips), library_path=str(library))

    assert (library / "Sister Superior" / "datapackage.json").exists()
    client.trigger_scan.assert_called_once()
    client.update_model.assert_not_called()


def test_run_upload_resolves_nested_relative_path_source(tmp_path, monkeypatch):
    # tag writes the CSV filename as a path relative to --zips; upload must
    # join it back onto --zips to find a source in a subfolder.
    monkeypatch.setenv("MANYFOLD_API_URL", "https://mf.example")
    monkeypatch.setenv("MANYFOLD_API_TOKEN", "tok")
    zips = tmp_path / "zips"
    (zips / "sub").mkdir(parents=True)
    (zips / "sub" / "Sister Superior.stl").write_text("mesh")
    library = tmp_path / "library"
    csv_path = _write_csv(tmp_path, [
        {"filename": "sub/Sister Superior.stl", "faction": "Adepta Sororitas"},
    ])
    client = _fake_client()

    with patch("manyfold_ingest.ManyfoldClient", return_value=client):
        run_upload(str(csv_path), zips_dir=str(zips), library_path=str(library))

    # staged under the final-component name (subfolder stripped), tags carried
    assert (library / "Sister Superior" / "datapackage.json").exists()
    client.trigger_scan.assert_called_once()


def test_run_upload_skips_continuation_rar_volumes(tmp_path, monkeypatch):
    # A multi-part set stages once via its first volume; continuation-volume
    # rows (incl. stale ones from a pre-fix CSV) are skipped.
    monkeypatch.setenv("MANYFOLD_API_URL", "https://mf.example")
    monkeypatch.setenv("MANYFOLD_API_TOKEN", "tok")
    zips = tmp_path / "zips"; zips.mkdir()
    (zips / "Krieg.part1.rar").write_text("a")
    (zips / "Krieg.part2.rar").write_text("b")
    library = tmp_path / "library"
    csv_path = _write_csv(tmp_path, [
        {"filename": "Krieg.part1.rar", "faction": "Astra Militarum"},
        {"filename": "Krieg.part2.rar", "faction": "Astra Militarum"},  # continuation -> skip
    ])
    client = _fake_client()

    def fake_extract(src, outdir):
        Path(outdir, "m.stl").write_text("mesh")

    monkeypatch.setattr("manyfold_ingest.patoolib.extract_archive", fake_extract)
    with patch("manyfold_ingest.ManyfoldClient", return_value=client):
        run_upload(str(csv_path), zips_dir=str(zips), library_path=str(library))

    # exactly one model, named after the set (no ".part01"), no second volume
    assert [p.name for p in library.iterdir()] == ["Krieg"]
    assert (library / "Krieg" / "datapackage.json").exists()


def test_run_upload_stages_split_7z_first_volume(tmp_path, monkeypatch):
    # A split 7z set stages once via .001 (extracted via patoolib by content),
    # named after the base; the .002 continuation row is skipped.
    monkeypatch.setenv("MANYFOLD_API_URL", "https://mf.example")
    monkeypatch.setenv("MANYFOLD_API_TOKEN", "tok")
    zips = tmp_path / "zips"; zips.mkdir()
    (zips / "Grey Knights.7z.001").write_text("a")
    (zips / "Grey Knights.7z.002").write_text("b")
    library = tmp_path / "library"
    csv_path = _write_csv(tmp_path, [
        {"filename": "Grey Knights.7z.001", "faction": "Grey Knights"},
        {"filename": "Grey Knights.7z.002", "faction": "Grey Knights"},  # continuation -> skip
    ])
    client = _fake_client()

    extracted = []
    def fake_extract(src, outdir):
        extracted.append(Path(src).name)
        Path(outdir, "m.stl").write_text("mesh")

    monkeypatch.setattr("manyfold_ingest.patoolib.extract_archive", fake_extract)
    with patch("manyfold_ingest.ManyfoldClient", return_value=client):
        run_upload(str(csv_path), zips_dir=str(zips), library_path=str(library))

    assert extracted == ["Grey Knights.7z.001"]  # only the first volume
    assert [p.name for p in library.iterdir()] == ["Grey Knights"]


def test_run_upload_stages_same_basename_sources_to_distinct_folders(tmp_path, monkeypatch):
    # Two different sources sharing a basename across --zips subfolders must
    # each stage into their own folder, not silently collide on one.
    monkeypatch.setenv("MANYFOLD_API_URL", "https://mf.example")
    monkeypatch.setenv("MANYFOLD_API_TOKEN", "tok")
    zips = tmp_path / "zips"
    (zips / "General").mkdir(parents=True)
    (zips / "Finished scans").mkdir(parents=True)
    (zips / "General" / "Execrator.stl").write_text("a")
    (zips / "Finished scans" / "Execrator.stl").write_text("b")
    library = tmp_path / "library"
    csv_path = _write_csv(tmp_path, [
        {"filename": "General/Execrator.stl", "faction": "Necrons"},
        {"filename": "Finished scans/Execrator.stl", "faction": "Necrons"},
    ])
    client = _fake_client()

    with patch("manyfold_ingest.ManyfoldClient", return_value=client):
        run_upload(str(csv_path), zips_dir=str(zips), library_path=str(library))

    # both staged into distinct folders — neither dropped
    assert (library / "Execrator (General)" / "datapackage.json").exists()
    assert (library / "Execrator (Finished scans)" / "datapackage.json").exists()
    assert (library / "Execrator (General)" / "Execrator.stl").read_text() == "a"
    assert (library / "Execrator (Finished scans)" / "Execrator.stl").read_text() == "b"


def test_run_upload_delete_source_removes_archive_after_staging(tmp_path, monkeypatch):
    monkeypatch.setenv("MANYFOLD_API_URL", "https://mf.example")
    monkeypatch.setenv("MANYFOLD_API_TOKEN", "tok")
    zips = tmp_path / "zips"
    zips.mkdir()
    src = zips / "Sister Superior.stl"
    src.write_text("mesh")
    library = tmp_path / "library"
    csv_path = _write_csv(tmp_path, [
        {"filename": "Sister Superior.stl", "faction": "Adepta Sororitas"},
    ])
    client = _fake_client()

    with patch("manyfold_ingest.ManyfoldClient", return_value=client):
        run_upload(str(csv_path), zips_dir=str(zips), library_path=str(library),
                   delete_source=True)

    assert (library / "Sister Superior" / "datapackage.json").exists()  # staged into B
    assert not src.exists()                                             # source removed from A


def test_run_upload_dry_run_keeps_source_even_with_delete_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("MANYFOLD_API_URL", "https://mf.example")
    monkeypatch.setenv("MANYFOLD_API_TOKEN", "tok")
    zips = tmp_path / "zips"
    zips.mkdir()
    src = zips / "Sister Superior.stl"
    src.write_text("mesh")
    library = tmp_path / "library"
    csv_path = _write_csv(tmp_path, [{"filename": "Sister Superior.stl", "faction": "Adepta Sororitas"}])
    client = _fake_client()

    with patch("manyfold_ingest.ManyfoldClient", return_value=client):
        run_upload(str(csv_path), zips_dir=str(zips), library_path=str(library),
                   delete_source=True, dry_run=True)

    assert src.exists()              # dry run never deletes
    assert not library.exists()      # and never stages


def test_run_upload_dry_run_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("MANYFOLD_API_URL", "https://mf.example")
    monkeypatch.setenv("MANYFOLD_API_TOKEN", "tok")
    zips = tmp_path / "zips"
    zips.mkdir()
    (zips / "Sister Superior.stl").write_text("mesh")
    library = tmp_path / "library"
    csv_path = _write_csv(tmp_path, [
        {"filename": "Wolfspear+Techmarine.zip", "faction": "Space Marines"},
        {"filename": "Sister Superior.stl", "faction": "Adepta Sororitas"},
    ])
    model = {"id": 1, "name": "Wolfspear Techmarine", "tags": []}
    client = _fake_client(models=[model])

    with patch("manyfold_ingest.ManyfoldClient", return_value=client):
        run_upload(str(csv_path), zips_dir=str(zips), library_path=str(library), dry_run=True)

    client.update_model.assert_not_called()
    client.create_collection.assert_not_called()
    client.trigger_scan.assert_not_called()
    assert not library.exists()


def test_run_upload_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("MANYFOLD_API_URL", "https://mf.example")
    monkeypatch.setenv("MANYFOLD_API_TOKEN", "tok")
    csv_path = _write_csv(tmp_path, [
        {"filename": "A.zip", "faction": "Orks"},
        {"filename": "B.zip", "faction": "Orks"},
    ])
    models = [
        {"id": 1, "name": "A", "tags": []},
        {"id": 2, "name": "B", "tags": []},
    ]
    client = _fake_client(models=models)

    with patch("manyfold_ingest.ManyfoldClient", return_value=client):
        run_upload(str(csv_path), limit=1)

    assert client.update_model.call_count == 1


def test_run_upload_check_probes_capabilities(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MANYFOLD_API_URL", "https://mf.example")
    monkeypatch.setenv("MANYFOLD_API_TOKEN", "tok")
    client = _fake_client()
    client.capabilities.return_value = {"spec_found": True, "model_update": True}

    with patch("manyfold_ingest.ManyfoldClient", return_value=client):
        run_upload(str(tmp_path / "nonexistent.csv"), check=True)

    out = capsys.readouterr().out
    assert "model_update: True" in out
    client.list_models.assert_not_called()
