"""Tests for ModelTagger2-2pu: a lone image file is a render, not a model, so
it is neither discovered/tagged nor staged as its own Manyfold model. Images
still ride along inside an archive/folder."""

import csv
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.append('src')


@pytest.fixture(autouse=True)
def _reset_client():
    import tagging
    tagging._anthropic_client = None
    yield
    tagging._anthropic_client = None


def _fake_response(record, in_tokens=100, out_tokens=40):
    import json
    block = MagicMock(); block.type = "text"; block.text = json.dumps(record)
    resp = MagicMock(); resp.content = [block]
    resp.usage.input_tokens = in_tokens; resp.usage.output_tokens = out_tokens
    return resp


# --- ext-set invariants ---------------------------------------------------

def test_discoverable_exts_excludes_images_includes_models_and_archives():
    from utils import DISCOVERABLE_EXTS, IMAGE_EXTS, MODEL_EXTS, ARCHIVE_EXTS
    assert ".jpg" not in DISCOVERABLE_EXTS
    assert ".png" not in DISCOVERABLE_EXTS
    assert not (IMAGE_EXTS & DISCOVERABLE_EXTS)   # no image ext leaks in
    assert MODEL_EXTS <= DISCOVERABLE_EXTS
    assert ARCHIVE_EXTS <= DISCOVERABLE_EXTS


# --- tagging discovery ----------------------------------------------------

def test_lone_image_not_discovered_but_model_is(tmp_path):
    zips = tmp_path / "zips"
    zips.mkdir()
    (zips / "render.png").write_text("img")     # lone render -> must be skipped
    (zips / "mini.stl").write_text("mesh")       # real model -> tagged
    out_csv = tmp_path / "tags.csv"

    record = {"faction": "Orks", "tags": []}
    client = MagicMock()
    client.messages.create.return_value = _fake_response(record)

    with patch("tagging.get_anthropic_client", return_value=client), \
            patch("tagging.count_tokens", side_effect=lambda text, model=None: len(text) // 4):
        from tagging import run_tagging
        # aos mode = retrieval disabled, so no vector DB needed
        run_tagging(str(zips), str(out_csv), None, None, "aos", provider="anthropic")

    rows = list(csv.reader(open(out_csv)))
    names = {r[0] for r in rows[1:]}
    assert "mini.stl" in names
    assert "render.png" not in names          # the lone image was never tagged
    assert client.messages.create.call_count == 1


# --- staging --------------------------------------------------------------

def test_stage_into_library_rejects_lone_image(tmp_path):
    from manyfold_ingest import stage_into_library
    from manyfold import ManyfoldError
    img = tmp_path / "render.png"
    img.write_text("img")
    library = tmp_path / "library"
    with pytest.raises(ManyfoldError, match="Unsupported file type"):
        stage_into_library(img, library, [])


def test_stage_into_library_still_stages_loose_model(tmp_path):
    from manyfold_ingest import stage_into_library
    stl = tmp_path / "Aveline.stl"
    stl.write_text("mesh")
    library = tmp_path / "library"
    dest = stage_into_library(stl, library, ["faction: Orks"])
    assert (dest / "Aveline.stl").exists()


def test_archive_with_render_still_stages_the_image(tmp_path, monkeypatch):
    # Images inside an archive ride along with the model — unchanged.
    from manyfold_ingest import stage_into_library
    import manyfold_ingest as mi

    def fake_extract(archive, outdir):
        p = __import__("pathlib").Path(outdir)
        (p / "body.stl").write_text("mesh")
        (p / "render.png").write_text("img")
    monkeypatch.setattr(mi.patoolib, "extract_archive", lambda archive, outdir: fake_extract(archive, outdir))

    arc = tmp_path / "Kit.zip"
    arc.write_text("zip")
    library = tmp_path / "library"
    dest = stage_into_library(arc, library, [])
    assert (dest / "body.stl").exists()
    assert (dest / "render.png").exists()   # image rode along inside the archive
