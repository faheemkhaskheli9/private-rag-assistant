import json

import pytest

from prag.config import load_settings


def test_defaults_when_no_path():
    s = load_settings()
    assert s.max_upload_bytes > 0
    assert str(s.storage_dir)


def test_explicit_missing_path_is_hard_error():
    with pytest.raises(FileNotFoundError):
        load_settings("does/not/exist.json")


def test_explicit_path_is_parsed(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"storage_dir": "x/y", "max_upload_bytes": 42}))
    s = load_settings(cfg)
    assert str(s.storage_dir).replace("\\", "/") == "x/y"
    assert s.max_upload_bytes == 42


def test_non_positive_limit_rejected(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"max_upload_bytes": 0}))
    with pytest.raises(ValueError):
        load_settings(cfg)
