import json
import csv
from unittest.mock import MagicMock, patch
import sys
sys.path.append('src')


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
    # tags are title-cased during cleanup
    assert rows[1][-1].replace(" ", "") == "Tag1,Tag2"


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
