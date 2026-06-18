import sys

sys.path.append('src')

import main


def test_load_env_reads_cwd_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("MODELTAGGER_TEST_KEY=from_dotenv\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MODELTAGGER_TEST_KEY", raising=False)
    main.load_env()
    import os
    assert os.environ.get("MODELTAGGER_TEST_KEY") == "from_dotenv"


def test_exported_env_wins_over_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("MODELTAGGER_TEST_KEY=from_dotenv\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MODELTAGGER_TEST_KEY", "from_export")
    main.load_env()
    import os
    assert os.environ.get("MODELTAGGER_TEST_KEY") == "from_export"


def test_load_env_no_dotenv_is_noop(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.delenv("MODELTAGGER_TEST_KEY", raising=False)
    main.load_env()  # must not raise
    import os
    assert "MODELTAGGER_TEST_KEY" not in os.environ
