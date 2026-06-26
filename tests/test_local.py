import json
import csv
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys
sys.path.append('src')


def test_supported_formats_mirror_manyfold():
    import utils
    # slicer/print-project + extra mesh formats Manyfold supports
    for ext in (".chitubox", ".ctb", ".lys", ".lyt", ".voxl", ".gcode"):
        assert ext in utils.SLICER_EXTS and ext in utils.LOOSE_EXTS
    for ext in (".3mf", ".ply", ".step", ".stp", ".gltf", ".glb", ".fbx", ".obj", ".stl"):
        assert ext in utils.MESH_EXTS and ext in utils.LOOSE_EXTS
    for ext in (".jpg", ".jpeg", ".webp", ".tiff", ".png"):
        assert ext in utils.IMAGE_EXTS and ext in utils.LOOSE_EXTS
    for ext in (".zip", ".rar", ".7z", ".gz", ".bz2"):
        assert ext in utils.ARCHIVE_EXTS and ext in utils.TAGGABLE_EXTS


def test_is_valid_archive_content_accepts_slicer_and_meshes(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    from tagging import is_valid_archive_content

    chitubox = tmp_path / "chitubox"; chitubox.mkdir()
    (chitubox / "model.chitubox").write_text("x")
    assert is_valid_archive_content(chitubox)  # was "no valid content" before

    mesh = tmp_path / "mesh"; mesh.mkdir()
    (mesh / "kit.3mf").write_text("x")
    assert is_valid_archive_content(mesh)

    danger = tmp_path / "danger"; danger.mkdir()
    (danger / "kit.stl").write_text("x"); (danger / "virus.exe").write_text("x")
    assert not is_valid_archive_content(danger)  # executables reject the archive

    docs_only = tmp_path / "docs"; docs_only.mkdir()
    (docs_only / "readme.md").write_text("x")
    assert not is_valid_archive_content(docs_only)  # no model content


def test_multipart_helpers():
    import utils
    assert utils.multipart_volume_number("Krieg.part01.rar") == 1
    assert utils.multipart_volume_number("Krieg.part7.rar") == 7
    assert utils.multipart_volume_number("Foo.rar") is None
    # "Part 3" (space) is a name, not a volume marker
    assert utils.multipart_volume_number("Tyranid Warriors Part 3.rar") is None
    # split-archive sets: name.<ext>.NNN
    assert utils.multipart_volume_number("Grey Knights.7z.001") == 1
    assert utils.multipart_volume_number("Grey Knights.7z.002") == 2
    assert utils.multipart_volume_number("Pirate Orc Boy Pack_2.7z.001") == 1
    assert utils.multipart_volume_number("Foo.7z") is None  # single 7z, not a split
    assert utils.strip_multipart_suffix("Krieg.part01") == "Krieg"
    assert utils.strip_multipart_suffix("Lesionaries (Supported).part2") == "Lesionaries (Supported)"
    assert utils.strip_multipart_suffix("Grey Knights.7z") == "Grey Knights"  # split stem
    assert utils.strip_multipart_suffix("no parts here") == "no parts here"
    assert utils.clean_file_name("Krieg.part01") == "Krieg"
    assert utils.clean_file_name("Grey Knights.7z") == "Grey Knights"


def test_multipart_volume_siblings(tmp_path):
    from utils import multipart_volume_siblings
    for n in (1, 2, 3):
        (tmp_path / f"Krieg.part{n}.rar").write_text("x")
    for n in ("001", "002"):
        (tmp_path / f"GK.7z.{n}").write_text("x")
    (tmp_path / "single.rar").write_text("x")

    assert {p.name for p in multipart_volume_siblings(tmp_path / "Krieg.part1.rar")} == \
        {"Krieg.part1.rar", "Krieg.part2.rar", "Krieg.part3.rar"}
    assert {p.name for p in multipart_volume_siblings(tmp_path / "GK.7z.001")} == \
        {"GK.7z.001", "GK.7z.002"}
    # non-multipart -> just itself
    assert multipart_volume_siblings(tmp_path / "single.rar") == [tmp_path / "single.rar"]


def test_tagging_distinguishes_extract_fail_from_no_content(tmp_path, monkeypatch, caplog):
    import logging
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    zips = tmp_path / "zips"; zips.mkdir()
    (zips / "corrupt.zip").write_text("z")
    (zips / "empty.zip").write_text("z")
    out_csv = tmp_path / "tags.csv"; vp = tmp_path / "db"

    mock_pc = MagicMock()
    mock_pc.get_or_create_collection.return_value = MagicMock()

    def fake_extract(p):
        if Path(p).name == "corrupt.zip":
            return None                                  # extraction failed
        d = tmp_path / ("ex_" + Path(p).name); d.mkdir()
        (d / "readme.txt").write_text("x")               # extracted, but no model file
        return d

    with patch("tagging.PersistentClient", return_value=mock_pc), \
            patch("tagging.extract_to_temp", side_effect=fake_extract), \
            patch("tagging.requests.get", return_value=MagicMock()), \
            patch("tagging.requests.post", return_value=MagicMock()), \
            caplog.at_level(logging.WARNING):
        from tagging import run_tagging
        run_tagging(str(zips), str(out_csv), str(vp), None, "warhammer", use_local=True)

    assert "Extraction failed for corrupt.zip" in caplog.text      # corrupt-vs-empty split
    assert "No model content for empty.zip" in caplog.text


def test_tagging_processes_only_first_rar_volume(tmp_path, monkeypatch):
    # A multi-part RAR set is one model: only .part1 is tagged; continuation
    # volumes are skipped (the first volume pulls in the rest at extract time).
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    zips = tmp_path / "zips"; zips.mkdir()
    for n in (1, 2, 3):
        (zips / f"Krieg.part{n}.rar").write_text("rar")
    out_csv = tmp_path / "tags.csv"; vector_path = tmp_path / "db"

    mock_col = MagicMock()
    mock_col.query.return_value = {"documents": [["lore"]], "distances": [[0.05]]}
    mock_pc = MagicMock(); mock_pc.get_or_create_collection.return_value = mock_col

    extracted = []
    def fake_extract(p):
        extracted.append(Path(p).name)
        d = tmp_path / ("ex_" + Path(p).name); d.mkdir()
        (d / "m.stl").write_text("mesh")
        return d

    class DummyGen:
        def raise_for_status(self): pass
        def json(self): return {"response": "tag1", "models": []}

    with patch("tagging.PersistentClient", return_value=mock_pc), \
            patch("tagging.extract_to_temp", side_effect=fake_extract), \
            patch("tagging.requests.get", return_value=DummyGen()), \
            patch("tagging.requests.post", return_value=DummyGen()), \
            patch("tagging.count_tokens", side_effect=lambda t, model=None: len(t)):
        from tagging import run_tagging
        run_tagging(str(zips), str(out_csv), str(vector_path), None, "warhammer", use_local=True)

    # only the first volume was even extracted/tagged
    assert extracted == ["Krieg.part1.rar"]
    rows = list(csv.reader(open(out_csv)))
    assert {r[0] for r in rows[1:]} == {"Krieg.part1.rar"}


def test_tagging_processes_only_first_7z_volume(tmp_path, monkeypatch):
    # A split 7z set (.7z.001 ... .NNN) is one model: only .001 is processed,
    # even though its extension (.001) isn't a recognized archive ext.
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    zips = tmp_path / "zips"; zips.mkdir()
    for n in ("001", "002", "003"):
        (zips / f"Grey Knights.7z.{n}").write_text("7z")
    out_csv = tmp_path / "tags.csv"; vector_path = tmp_path / "db"

    mock_col = MagicMock()
    mock_col.query.return_value = {"documents": [["lore"]], "distances": [[0.05]]}
    mock_pc = MagicMock(); mock_pc.get_or_create_collection.return_value = mock_col

    extracted = []
    def fake_extract(p):
        extracted.append(Path(p).name)
        d = tmp_path / ("ex_" + Path(p).name); d.mkdir()
        (d / "m.stl").write_text("mesh")
        return d

    class DummyGen:
        def raise_for_status(self): pass
        def json(self): return {"response": "tag1", "models": []}

    with patch("tagging.PersistentClient", return_value=mock_pc), \
            patch("tagging.extract_to_temp", side_effect=fake_extract), \
            patch("tagging.requests.get", return_value=DummyGen()), \
            patch("tagging.requests.post", return_value=DummyGen()), \
            patch("tagging.count_tokens", side_effect=lambda t, model=None: len(t)):
        from tagging import run_tagging
        run_tagging(str(zips), str(out_csv), str(vector_path), None, "warhammer", use_local=True)

    assert extracted == ["Grey Knights.7z.001"]
    rows = list(csv.reader(open(out_csv)))
    assert {r[0] for r in rows[1:]} == {"Grey Knights.7z.001"}


def test_extract_nested_archives_unpacks_and_recurses(tmp_path, monkeypatch):
    import utils
    root = tmp_path / "root"; root.mkdir()
    (root / "level1.zip").write_text("z")     # outer bundle -> inner archive -> model
    (root / "render.png").write_text("img")   # non-archive, kept

    def fake_extract(src, outdir):
        s = str(src)
        if s.endswith("level1.zip"):
            Path(outdir, "level2.zip").write_text("z")
        elif s.endswith("level2.zip"):
            Path(outdir, "deep.stl").write_text("mesh")
    monkeypatch.setattr("utils.patoolib.extract_archive", fake_extract)

    utils.extract_nested_archives(root)
    names = {p.name for p in root.rglob("*") if p.is_file()}
    assert "deep.stl" in names               # unpacked two levels deep
    assert "render.png" in names             # untouched
    assert not any(n.endswith((".zip", ".rar", ".7z")) for n in names)  # archives gone


def test_extract_nested_archives_skips_corrupt(tmp_path, monkeypatch):
    import utils
    root = tmp_path / "root"; root.mkdir()
    (root / "good.zip").write_text("z")
    (root / "bad.rar").write_text("z")

    def fake_extract(src, outdir):
        if "bad" in str(src):
            raise RuntimeError("Unexpected end of archive")
        Path(outdir, "m.stl").write_text("mesh")
    monkeypatch.setattr("utils.patoolib.extract_archive", fake_extract)

    utils.extract_nested_archives(root)        # must not raise
    names = {p.name for p in root.rglob("*") if p.is_file()}
    assert "m.stl" in names                   # good one unpacked
    assert "bad.rar" not in names             # corrupt one dropped, no infinite loop


def test_parse_tags_examples(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    from tagging import parse_tags

    cases = [
        (
            "1. Adepta Sororitas; 2. Sisters of Battle; 3. Sister Superior; 4. Sergeant; 5. Seraphim; 6. Veteran Sister Superior; 7. Novitiate Superior; 8. Power Armour; 9. Chainsword; 10. Boltgun",
            [
                "Adepta Sororitas",
                "Sisters of Battle",
                "Sister Superior",
                "Sergeant",
                "Seraphim",
                "Veteran Sister Superior",
                "Novitiate Superior",
                "Power Armour",
                "Chainsword",
                "Boltgun",
            ],
        ),
        (
            "Here are some suggested tags for the \"Wolfspear Techmarine\":; ; 1. Wolfspear; 2. Space Marines; 3. Adeptus Astartes; 4. Techmarine; 5. Adeptus Mechanicus; 6. Primaris; 7. Support; 8. Servo-Arm; 9. Mechadendrite; 10. Armoury Custodian",
            [
                "Wolfspear",
                "Space Marines",
                "Adeptus Astartes",
                "Techmarine",
                "Adeptus Mechanicus",
                "Primaris",
                "Support",
                "Servo-Arm",
                "Mechadendrite",
                "Armoury Custodian",
            ],
        ),
    ]

    for raw, expected in cases:
        assert parse_tags(raw) == expected


def test_local_embeddings(tmp_path):
    data = [{"url": "http://example.com/a", "text": "lore"}]
    lore_path = tmp_path / "lore.json"
    lore_path.write_text(json.dumps(data))
    vector_path = tmp_path / "chroma"
    mock_col = MagicMock()
    with patch("embed.chromadb.PersistentClient") as mock_pc, \
            patch("chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction") as mock_ef, \
            patch("embed.tiktoken.get_encoding") as mock_ge, \
            patch("embed.tiktoken.encoding_for_model") as mock_efm:
        mock_pc.return_value.get_or_create_collection.return_value = mock_col
        dummy_enc = MagicMock()
        dummy_enc.encode.side_effect = lambda x: x
        mock_ge.return_value = dummy_enc
        mock_efm.return_value = dummy_enc
        from embed import run_embedding
        run_embedding(str(lore_path), str(vector_path), use_local=True, embed_model="bge", model="gpt-4o")
        mock_ef.assert_called_once_with(model_name="bge")


def test_local_generation(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    zips = tmp_path / "zips"
    zips.mkdir()
    (zips / "m.stl").write_text("mesh")
    out_csv = tmp_path / "tags.csv"
    vector_path = tmp_path / "db"

    mock_col = MagicMock()
    mock_col.query.return_value = {"documents": [["lore1", "lore2"]], "distances": [[0.05, 0.1]]}
    mock_pc = MagicMock()
    mock_pc.get_or_create_collection.return_value = mock_col

    class DummyPull:
        def raise_for_status(self):
            pass
        def json(self):
            return {}

    class DummyGen:
        def raise_for_status(self):
            pass
        def json(self):
            return {"response": "tag1, tag2"}

    def post_side_effect(url, *args, **kwargs):
        if url.endswith("/api/pull"):
            return DummyPull()
        return DummyGen()

    with patch("tagging.PersistentClient", return_value=mock_pc), \
            patch("tagging.requests.get") as mock_get, \
            patch("tagging.requests.post", side_effect=post_side_effect) as mock_post, \
            patch("tagging.count_tokens", side_effect=lambda text, model=None: len(text)):
        mock_get.return_value.raise_for_status = lambda: None
        mock_get.return_value.json.return_value = {"models": []}
        from tagging import run_tagging
        run_tagging(str(zips), str(out_csv), str(vector_path), None, "warhammer", use_local=True)

    urls = [call.args[0] for call in mock_post.call_args_list]
    assert urls[0].endswith("/api/pull")
    assert urls[1].endswith("/api/generate")

    rows = list(csv.reader(open(out_csv)))
    # tags are title-cased during cleanup; tagged_at is the trailing column
    assert rows[0][-1] == "tagged_at"
    assert rows[1][-2].replace(" ", "") == "Tag1,Tag2"
    # a real UTC timestamp was recorded for the tagged row
    from datetime import datetime
    assert datetime.fromisoformat(rows[1][-1]).tzinfo is not None


def test_tagging_recurses_nested_folders_with_relative_paths(tmp_path, monkeypatch):
    # A nested incoming library: archives live in subfolders, and two share a
    # basename. Discovery must recurse, and each row must carry the path
    # RELATIVE to --zips so same-basename files don't collide on one row.
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    zips = tmp_path / "zips"
    (zips / "sub_a").mkdir(parents=True)
    (zips / "sub_b").mkdir(parents=True)
    (zips / "top.stl").write_text("mesh")
    (zips / "sub_a" / "Model.stl").write_text("mesh")
    (zips / "sub_b" / "Model.stl").write_text("mesh")  # same basename, different folder
    out_csv = tmp_path / "tags.csv"
    vector_path = tmp_path / "db"

    mock_col = MagicMock()
    mock_col.query.return_value = {"documents": [["lore1", "lore2"]], "distances": [[0.05, 0.1]]}
    mock_pc = MagicMock()
    mock_pc.get_or_create_collection.return_value = mock_col

    class DummyPull:
        def raise_for_status(self):
            pass
        def json(self):
            return {}

    class DummyGen:
        def raise_for_status(self):
            pass
        def json(self):
            return {"response": "tag1, tag2"}

    def post_side_effect(url, *args, **kwargs):
        if url.endswith("/api/pull"):
            return DummyPull()
        return DummyGen()

    with patch("tagging.PersistentClient", return_value=mock_pc), \
            patch("tagging.requests.get") as mock_get, \
            patch("tagging.requests.post", side_effect=post_side_effect), \
            patch("tagging.count_tokens", side_effect=lambda text, model=None: len(text)):
        mock_get.return_value.raise_for_status = lambda: None
        mock_get.return_value.json.return_value = {"models": []}
        from tagging import run_tagging
        run_tagging(str(zips), str(out_csv), str(vector_path), None, "warhammer", use_local=True)

    rows = list(csv.reader(open(out_csv)))
    filenames = {r[0] for r in rows[1:]}
    # nested archives discovered, stored as forward-slash relative paths, and
    # the basename collision is preserved as two distinct rows
    assert filenames == {"top.stl", "sub_a/Model.stl", "sub_b/Model.stl"}


def test_rerank(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    zips = tmp_path / "zips"
    zips.mkdir()
    (zips / "m.stl").write_text("mesh")
    out_csv = tmp_path / "tags.csv"
    vector_path = tmp_path / "db"

    docs = ["doc1", "doc2", "doc3"]
    mock_col = MagicMock()
    mock_col.query.return_value = {"documents": [docs], "distances": [[0.05, 0.05, 0.05]]}
    mock_pc = MagicMock()
    mock_pc.get_or_create_collection.return_value = mock_col

    class DummyResp:
        def raise_for_status(self):
            pass
        def json(self):
            return {"response": "tag"}

    mock_ce = MagicMock()
    mock_ce.predict.return_value = [0.1, 0.9, 0.2]
    mock_post = MagicMock(return_value=DummyResp())

    import types, sys
    sys.modules['sentence_transformers'] = types.SimpleNamespace(CrossEncoder=lambda *a, **k: mock_ce)

    with patch("tagging.PersistentClient", return_value=mock_pc), \
            patch("tagging.requests.get") as mock_get, \
            patch("tagging.requests.post", mock_post), \
            patch("tagging.count_tokens", side_effect=lambda text, model=None: len(text)):
        mock_get.return_value.raise_for_status = lambda: None
        mock_get.return_value.json.return_value = {"models": [{"model": "llama3.1:8b-instruct"}]}
        from tagging import run_tagging
        run_tagging(str(zips), str(out_csv), str(vector_path), None, "warhammer", use_local=True, rerank=True, token_budget=5000)

    prompt = mock_post.call_args[1]["json"]["prompt"]
    assert prompt.find("doc2") < prompt.find("doc1")


def test_tagging_skips_logged_files(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    zips = tmp_path / "zips"
    zips.mkdir()
    (zips / "m.stl").write_text("mesh")
    out_csv = tmp_path / "tags.csv"
    vector_path = tmp_path / "db"

    mock_col = MagicMock()
    mock_col.query.return_value = {"documents": [["lore1", "lore2"]], "distances": [[0.05, 0.1]]}
    mock_pc = MagicMock()
    mock_pc.get_or_create_collection.return_value = mock_col

    class DummyPull:
        def raise_for_status(self):
            pass
        def json(self):
            return {}

    class DummyGen:
        def raise_for_status(self):
            pass
        def json(self):
            return {"response": "tag1, tag2"}

    def post_side_effect(url, *args, **kwargs):
        if url.endswith("/api/pull"):
            return DummyPull()
        return DummyGen()

    with patch("tagging.PersistentClient", return_value=mock_pc), \
            patch("tagging.requests.get") as mock_get, \
            patch("tagging.requests.post", side_effect=post_side_effect) as mock_post, \
            patch("tagging.count_tokens", side_effect=lambda text, model=None: len(text)):
        mock_get.return_value.raise_for_status = lambda: None
        mock_get.return_value.json.return_value = {"models": []}
        from tagging import run_tagging
        run_tagging(str(zips), str(out_csv), str(vector_path), None, "warhammer", use_local=True)
        run_tagging(str(zips), str(out_csv), str(vector_path), None, "warhammer", use_local=True)

    rows = list(csv.reader(open(out_csv)))
    assert len(rows) == 2


def test_empty_vector_db_skips_gracefully(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    zips = tmp_path / "zips"
    zips.mkdir()
    (zips / "m.stl").write_text("mesh")
    out_csv = tmp_path / "tags.csv"
    vector_path = tmp_path / "db"

    mock_col = MagicMock()
    mock_col.query.return_value = {"documents": [[]], "distances": [[]]}
    mock_pc = MagicMock()
    mock_pc.get_or_create_collection.return_value = mock_col

    with patch("tagging.PersistentClient", return_value=mock_pc), \
            patch("tagging.requests.post") as mock_post:
        from tagging import run_tagging
        run_tagging(str(zips), str(out_csv), str(vector_path), None, "warhammer", use_local=True)

    # No LLM call, and the file is NOT written (so a re-run retries it once
    # the vector DB has lore) — only the header is present.
    assert not any(call.args[0].endswith("/api/generate") for call in mock_post.call_args_list)
    rows = list(csv.reader(open(out_csv)))
    assert len(rows) == 1  # header only
    assert "m.stl" not in {r[0] for r in rows[1:]}


def test_main_wires_prompt_override(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    import main

    captured = {}

    def fake_run_tagging(zips, out, db, prompt_override, mode, **kwargs):
        captured["prompt_override"] = prompt_override

    monkeypatch.setattr(main, "run_tagging", fake_run_tagging)
    monkeypatch.setattr(sys, "argv", [
        "main.py", "tag", "--zips", "z", "--tag-output", "t.csv",
        "--prompt-override", "custom prompt",
    ])
    main.main()
    assert captured["prompt_override"] == "custom prompt"


def test_failed_extraction_not_written_so_it_retries(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    zips = tmp_path / "zips"
    zips.mkdir()
    (zips / "bad.stl").write_text("mesh")
    out_csv = tmp_path / "tags.csv"
    vector_path = tmp_path / "db"

    mock_col = MagicMock()
    mock_pc = MagicMock()
    mock_pc.get_or_create_collection.return_value = mock_col

    with patch("tagging.PersistentClient", return_value=mock_pc), \
            patch("tagging.extract_to_temp", return_value=None):
        from tagging import run_tagging
        run_tagging(str(zips), str(out_csv), str(vector_path), None, "warhammer", use_local=True)

    # A failed file is NOT written — only the header — so a re-run retries it.
    rows = list(csv.reader(open(out_csv)))
    assert len(rows) == 1
    assert rows[0][0] == "filename"
