"""Boundary-detection tests for loose-file folders (ModelTagger2-17z)."""

import csv
import json
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.append('src')

from loose_folders import resolve_model_units, MAX_SPLIT


def _mk(root, *rel_files):
    """Create each 'a/b/c.stl' relative path under root (parents included)."""
    for rel in rel_files:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("mesh")


def _units(root):
    units, warnings = resolve_model_units(root)
    return sorted(u.rel_path for u in units), warnings


# --- the worked example from ./files/ (5 models) --------------------------

def test_files_example_resolves_to_five_models(tmp_path):
    _mk(
        tmp_path,
        # Barbgants: wrapper chain collapses; Supported folds in -> 1 model
        "Barbgants/KaiTech_Design/space-bug/model.stl",
        "Barbgants/KaiTech_Design/space-bug/Supported/model_sup.stl",
        # Biovore: 4 model-bearing children -> split
        "Biovore/Bios Voreus/body.stl",                  # direct files win
        "Biovore/Bios Voreus/Spore_mine_with_base/mine.stl",
        "Biovore/Biovore/biovore.stl",
        "Biovore/Biovore 2/b2.stl",                      # direct files win (folds bits)
        "Biovore/Biovore 2/bits/bit.stl",
        "Biovore/Biovore 3/files/b3.stl",                # variant-only -> Biovore 3
    )
    paths, warnings = _units(tmp_path)
    assert paths == [
        "Barbgants",
        "Biovore/Bios Voreus",
        "Biovore/Biovore",
        "Biovore/Biovore 2",
        "Biovore/Biovore 3",
    ]
    assert warnings == []


def test_unit_name_is_final_path_component(tmp_path):
    _mk(tmp_path, "Biovore/Biovore 2/b2.stl", "Biovore/Biovore 3/files/b3.stl")
    units, _ = resolve_model_units(tmp_path)
    by_path = {u.rel_path: u.name for u in units}
    assert by_path["Biovore/Biovore 2"] == "Biovore 2"
    assert by_path["Biovore/Biovore 3"] == "Biovore 3"


# --- individual rules -----------------------------------------------------

def test_direct_files_win_folds_in_subfolders(tmp_path):
    # A folder with its own STLs is one model even with model-bearing subfolders.
    _mk(tmp_path, "Hero/hero.stl", "Hero/Weapons/sword.stl", "Hero/Supported/hero_sup.stl")
    paths, _ = _units(tmp_path)
    assert paths == ["Hero"]


def test_wrapper_collapse_single_child(tmp_path):
    _mk(tmp_path, "Squad/Wrapper/Inner/guy.stl")
    paths, _ = _units(tmp_path)
    assert paths == ["Squad"]  # name stays the meaningful chain top


def test_variant_only_folder_is_one_model(tmp_path):
    _mk(tmp_path, "Tank/OBJ/tank.obj", "Tank/STL/tank.stl")
    paths, _ = _units(tmp_path)
    assert paths == ["Tank"]


def test_variant_folder_detection_covers_abbreviations():
    from loose_folders import _is_variant_folder
    for v in ["stl supp", "lys supp", "stl fin", "STL Fixed", "stl fin 01",
              "obj", "Supported", "presupported"]:
        assert _is_variant_folder(v), v
    for real in ["Support Weapon", "Barbgants", "Genestealer 1", "Bios Voreus",
                 "Neurogants"]:
        assert not _is_variant_folder(real), real


def test_abbreviated_variant_children_fold_into_parent(tmp_path):
    # 'stl supp' / 'lys supp' / 'stl fin' are format variants, not separate
    # models — the parent must stay ONE model, keeping its name (ModelTagger2-keh).
    _mk(tmp_path,
        "Neurogants/stl supp/m.stl",
        "Neurogants/lys supp/m.lys",
        "Neurogants/stl fin/m.stl")
    paths, _ = _units(tmp_path)
    assert paths == ["Neurogants"]


def test_parts_kitbash_is_one_model(tmp_path):
    # >=3 children, >=60% part-named -> one kitbash model, not a split.
    _mk(
        tmp_path,
        "Hive Tyrant/base/base.stl",
        "Hive Tyrant/wings/wings.stl",
        "Hive Tyrant/fuhrer_sword/sword.stl",
        "Hive Tyrant/rending_claws_1/claw.stl",
        "Hive Tyrant/head/head.stl",
    )
    paths, _ = _units(tmp_path)
    assert paths == ["Hive Tyrant"]


def test_grouping_split_into_real_children(tmp_path):
    _mk(
        tmp_path,
        "Squad/Sergeant/sgt.stl",
        "Squad/Gunner/gun_guy.stl",
        "Squad/Champion/champ.stl",
    )
    paths, _ = _units(tmp_path)
    assert paths == ["Squad/Champion", "Squad/Gunner", "Squad/Sergeant"]


def test_generic_top_falls_through_to_meaningful_folder(tmp_path):
    # A generic wrapper (Patreon) yields the meaningful folder as root + name.
    _mk(tmp_path, "Patreon/Barbgants/model.stl")
    units, _ = resolve_model_units(tmp_path)
    assert [(u.rel_path, u.name) for u in units] == [("Patreon/Barbgants", "Barbgants")]


def test_root_level_loose_file_is_its_own_model(tmp_path):
    _mk(tmp_path, "lonely.stl", "Kit/a/x.stl")
    paths, _ = _units(tmp_path)
    assert "lonely.stl" in paths
    assert "Kit" in paths


def test_archive_only_folder_yields_no_unit(tmp_path):
    # Archives aren't loose model files -> handled elsewhere, no loose unit here.
    (tmp_path / "Bundle").mkdir()
    (tmp_path / "Bundle" / "pack.zip").write_text("zip")
    paths, _ = _units(tmp_path)
    assert paths == []


def test_max_split_emits_warning_but_still_produces_units(tmp_path):
    files = [f"Box/Sculpt{i}/m{i}.stl" for i in range(MAX_SPLIT + 1)]
    _mk(tmp_path, *files)
    units, warnings = resolve_model_units(tmp_path)
    assert len(units) == MAX_SPLIT + 1
    assert len(warnings) == 1
    assert "review before upload" in warnings[0]


# --- archive-dominated folders are not merged (ModelTagger2-89r) -----------

def _mk_files(root, *rels):
    for rel in rels:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")


def test_archive_collection_folder_is_skipped_not_merged(tmp_path):
    # A folder of many archives with a few stray loose files must NOT collapse
    # into one merged model — the archives are separate models.
    zips = tmp_path / "zips"
    archives = [f"KUH/kit{i}.zip" for i in range(10)]
    _mk_files(zips, *archives, "KUH/warhound face.stl", "KUH/banner.stl")
    units, warnings = resolve_model_units(zips)
    assert units == []                                   # nothing merged
    assert len(warnings) == 1
    assert "archive-dominated" in warnings[0]
    assert "KUH" in warnings[0]


def test_pure_loose_kit_still_groups(tmp_path):
    # No archives -> unaffected by the guard.
    zips = tmp_path / "zips"
    _mk_files(zips, "Barbgants/sculpt/model.stl")
    paths, warnings = _units(zips)
    assert paths == ["Barbgants"]
    assert warnings == []


def test_loose_kit_with_one_stray_archive_still_groups(tmp_path):
    # Loose files dominate (5 loose vs 1 archive) -> still grouped as a kit.
    zips = tmp_path / "zips"
    _mk_files(zips, "Kit/a.stl", "Kit/b.stl", "Kit/c.stl", "Kit/d.stl",
              "Kit/e.stl", "Kit/bonus.zip")
    paths, warnings = _units(zips)
    assert paths == ["Kit"]
    assert warnings == []


def test_stage_folder_unit_excludes_stray_archive(tmp_path):
    # The staged model must contain the loose files but NOT the stray archive
    # (that's staged separately).
    sys.path.append('src')
    from manyfold_ingest import stage_into_library
    folder = tmp_path / "Kit"
    _mk_files(folder, "a.stl", "b.stl", "bonus.zip")
    library = tmp_path / "library"
    dest = stage_into_library(folder, library, ["faction: Orks"])
    assert (dest / "a.stl").exists()
    assert (dest / "b.stl").exists()
    assert not (dest / "bonus.zip").exists()   # archive dropped from the unit


# --- tagging discovery integration ----------------------------------------

def _fake_response(record, in_tokens=100, out_tokens=40):
    block = MagicMock()
    block.type = "text"
    block.text = json.dumps(record)
    resp = MagicMock()
    resp.content = [block]
    resp.usage.input_tokens = in_tokens
    resp.usage.output_tokens = out_tokens
    return resp


@pytest.fixture
def _reset_anthropic():
    import tagging
    tagging._anthropic_client = None
    yield
    tagging._anthropic_client = None


def test_run_tagging_loose_as_folders_one_row_per_unit(tmp_path, _reset_anthropic):
    # A kit's loose files in nested subfolders become ONE model row keyed by the
    # folder, not one row per .stl. Uses aos mode (retrieval disabled) so no
    # vector DB is needed.
    zips = tmp_path / "zips"
    _mk(zips, "Barbgants/sculpt/model.stl", "Barbgants/sculpt/Supported/model_sup.stl")
    out_csv = tmp_path / "tags.csv"

    record = {"faction": "Tyranids", "subfaction": "", "unit": "Barbgants",
              "model_type": "Monster", "role": "Battleline", "grand_alliance": "unknown",
              "equipment": [], "tags": ["Bio-titan"]}
    client = MagicMock()
    client.messages.create.return_value = _fake_response(record)

    with patch("tagging.get_anthropic_client", return_value=client), \
            patch("tagging.count_tokens", side_effect=lambda text, model=None: len(text) // 4):
        from tagging import run_tagging
        run_tagging(str(zips), str(out_csv), None, None, "aos",
                    provider="anthropic", loose_as_folders=True)

    rows = list(csv.reader(open(out_csv)))
    data = rows[1:]
    assert len(data) == 1                         # one model, not two files
    assert data[0][0] == "Barbgants"              # keyed by the folder unit
    # Exactly one LLM call for the whole kit.
    assert client.messages.create.call_count == 1


def test_run_tagging_without_flag_tags_each_loose_file(tmp_path, _reset_anthropic):
    # Same tree, no flag: each loose .stl is its own row (the old behavior).
    zips = tmp_path / "zips"
    _mk(zips, "Barbgants/sculpt/model.stl", "Barbgants/sculpt/Supported/model_sup.stl")
    out_csv = tmp_path / "tags.csv"

    record = {"faction": "Tyranids", "tags": []}
    client = MagicMock()
    client.messages.create.return_value = _fake_response(record)

    with patch("tagging.get_anthropic_client", return_value=client), \
            patch("tagging.count_tokens", side_effect=lambda text, model=None: len(text) // 4):
        from tagging import run_tagging
        run_tagging(str(zips), str(out_csv), None, None, "aos", provider="anthropic")

    rows = list(csv.reader(open(out_csv)))
    assert len(rows[1:]) == 2  # two loose files -> two rows


# --- staging integration --------------------------------------------------

def test_stage_into_library_flattens_folder_unit(tmp_path):
    sys.path.append('src')
    from manyfold_ingest import stage_into_library
    folder = tmp_path / "Biovore 2"
    _mk(folder, "body.stl", "bits/arm.stl", "Supported/body_sup.stl")
    library = tmp_path / "library"

    dest = stage_into_library(folder, library, ["faction: Tyranids"])

    assert dest == library / "Biovore 2"
    # Every model file is flattened to the model root (no subfolders remain).
    assert (dest / "body.stl").exists()
    assert (dest / "bits_arm.stl").exists()
    assert (dest / "Supported_body_sup.stl").exists()
    assert not (dest / "bits").exists()
    pkg = json.loads((dest / "datapackage.json").read_text())
    assert pkg["title"] == "Biovore 2"
    assert pkg["keywords"] == ["faction: Tyranids"]


def test_run_upload_stages_folder_unit(tmp_path, monkeypatch):
    sys.path.append('src')
    from manyfold_ingest import run_upload
    monkeypatch.setenv("MANYFOLD_API_URL", "https://mf.example")
    monkeypatch.setenv("MANYFOLD_API_TOKEN", "tok")
    zips = tmp_path / "zips"
    _mk(zips, "Biovore/Biovore 2/b2.stl", "Biovore/Biovore 2/bits/bit.stl")
    library = tmp_path / "library"
    csv_path = tmp_path / "tags.csv"
    csv_path.write_text("filename,faction,tags\nBiovore/Biovore 2,Tyranids,\n")

    client = MagicMock()
    client.list_models.return_value = []
    client.list_collections.return_value = []
    client.trigger_scan.return_value = True

    with patch("manyfold_ingest.ManyfoldClient", return_value=client):
        run_upload(str(csv_path), zips_dir=str(zips), library_path=str(library))

    dest = library / "Biovore 2"
    assert (dest / "datapackage.json").exists()
    assert (dest / "b2.stl").exists()
    assert (dest / "bits_bit.stl").exists()  # flattened
    client.trigger_scan.assert_called_once()
