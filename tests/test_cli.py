import sys
import pytest

sys.path.append('src')

import main


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")


def _run(monkeypatch, argv, **fakes):
    calls = []

    def record(name):
        def fake(*args, **kwargs):
            calls.append((name, args, kwargs))
        return fake

    for fn in ("run_scraping", "run_embedding", "run_tagging", "run_upload"):
        monkeypatch.setattr(main, fn, fakes.get(fn) or record(fn))
    monkeypatch.setattr(sys, "argv", ["main.py"] + argv)
    main.main()
    return calls


def test_scrape_uses_preset_defaults(monkeypatch):
    calls = _run(monkeypatch, ["scrape"])
    assert calls == [("run_scraping", ("seeds/warhammer_seeds.txt", "lore/warhammer", 100, 2), {})]


def test_scrape_flags_override_preset(monkeypatch):
    calls = _run(monkeypatch, [
        "scrape", "--mode", "dnd", "--seeds", "my.txt", "--lore-dir", "out", "--max-pages", "7",
    ])
    assert calls == [("run_scraping", ("my.txt", "out", 7, 2), {})]


def test_embed_uses_preset_defaults(monkeypatch):
    calls = _run(monkeypatch, ["embed", "--mode", "dnd"])
    name, args, kwargs = calls[0]
    assert name == "run_embedding"
    assert args == ("lore/dnd", ".chroma/dnd")


def test_tag_uses_preset_defaults(monkeypatch):
    calls = _run(monkeypatch, ["tag", "--zips", "data/zips"])
    name, args, kwargs = calls[0]
    assert name == "run_tagging"
    assert args == ("data/zips", "tags-warhammer.csv", ".chroma/warhammer", None, "warhammer")


def test_upload_uses_preset_default_csv(monkeypatch):
    calls = _run(monkeypatch, ["upload"])
    name, args, kwargs = calls[0]
    assert name == "run_upload"
    assert args == ("tags-warhammer.csv",)
    assert kwargs["dry_run"] is False
    assert kwargs["check"] is False


def test_all_runs_pipeline_in_order(monkeypatch):
    calls = _run(monkeypatch, ["all", "--zips", "data/zips"])
    assert [c[0] for c in calls] == ["run_scraping", "run_embedding", "run_tagging"]
    # All steps share the preset paths
    assert calls[0][1][1] == "lore/warhammer"
    assert calls[1][1] == ("lore/warhammer", ".chroma/warhammer")
    assert calls[2][1][2] == ".chroma/warhammer"


def test_all_with_upload_flag(monkeypatch):
    calls = _run(monkeypatch, ["all", "--zips", "data/zips", "--upload"])
    assert [c[0] for c in calls] == ["run_scraping", "run_embedding", "run_tagging", "run_upload"]
    assert calls[3][1] == ("tags-warhammer.csv",)


def test_unknown_mode_exits(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "scrape", "--mode", "starwars"])
    with pytest.raises(SystemExit, match="Unknown mode"):
        main.main()


def test_missing_step_exits(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py"])
    with pytest.raises(SystemExit):
        main.main()
