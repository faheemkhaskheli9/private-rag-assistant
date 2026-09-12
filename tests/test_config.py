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


def test_chunking_defaults_when_no_path():
    s = load_settings()
    assert s.chunk_size > 0
    assert 0 <= s.chunk_overlap < s.chunk_size
    assert s.splitter in ("fixed", "recursive")


def test_default_yaml_config_loads_and_validates():
    s = load_settings("configs/default.yaml")
    assert s.chunk_size == 800
    assert s.chunk_overlap == 100
    assert s.splitter == "fixed"


def test_yaml_config_overrides_chunking(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("chunk_size: 200\nchunk_overlap: 20\nsplitter: recursive\n")
    s = load_settings(cfg)
    assert s.chunk_size == 200
    assert s.chunk_overlap == 20
    assert s.splitter == "recursive"


def test_overlap_greater_than_chunk_size_rejected_with_clear_error(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("chunk_size: 100\nchunk_overlap: 150\n")
    with pytest.raises(ValueError, match="chunk_overlap"):
        load_settings(cfg)


def test_unknown_splitter_rejected(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("splitter: magic\n")
    with pytest.raises(ValueError, match="splitter"):
        load_settings(cfg)
