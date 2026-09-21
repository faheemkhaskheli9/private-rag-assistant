"""Runtime configuration.

Lenient with the default (no config file -> built-in defaults); strict with an
explicitly supplied path (missing -> hard error, never a silent fall-through
to defaults). See the portfolio robustness rules.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from .chunking import ChunkConfig

_DEFAULT_MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MiB


@dataclass(frozen=True)
class Settings:
    storage_dir: Path = Path("data/store")
    max_upload_bytes: int = _DEFAULT_MAX_UPLOAD_BYTES
    max_files_per_upload: int = 20
    chunk_size: int = 800
    chunk_overlap: int = 100
    splitter: str = "fixed"

    def __post_init__(self) -> None:
        # Both limits bound what one request can make the server buffer/store;
        # a zero or negative value would either reject everything or, worse,
        # be read as "no limit" by a careless comparison.
        for name in ("max_upload_bytes", "max_files_per_upload"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        # Reuses ChunkConfig's validation so "overlap >= chunk_size" and an
        # unknown splitter name fail with the same clear error whether they
        # came from a config file or a direct Settings() call.
        ChunkConfig(self.chunk_size, self.chunk_overlap, self.splitter)

    def with_overrides(self, **kwargs: object) -> "Settings":
        clean = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **clean)  # type: ignore[arg-type]

    def chunking(self) -> ChunkConfig:
        return ChunkConfig(self.chunk_size, self.chunk_overlap, self.splitter)


def _coerce(raw: dict[str, object]) -> Settings:
    base = Settings()
    storage_dir = raw.get("storage_dir")
    max_upload = raw.get("max_upload_bytes")
    if max_upload is not None:
        max_upload = int(max_upload)
        if max_upload <= 0:
            raise ValueError("max_upload_bytes must be positive")
    max_files = raw.get("max_files_per_upload")
    chunk_size = raw.get("chunk_size")
    chunk_overlap = raw.get("chunk_overlap")
    splitter = raw.get("splitter")
    return base.with_overrides(
        storage_dir=Path(storage_dir) if storage_dir is not None else None,
        max_upload_bytes=max_upload,
        max_files_per_upload=int(max_files) if max_files is not None else None,
        chunk_size=int(chunk_size) if chunk_size is not None else None,
        chunk_overlap=int(chunk_overlap) if chunk_overlap is not None else None,
        splitter=splitter,
    )


def load_settings(config_path: str | os.PathLike[str] | None = None) -> Settings:
    """Load settings.

    * ``config_path`` given but missing -> :class:`FileNotFoundError`.
    * ``config_path`` given and present -> parsed as YAML (``.yaml``/``.yml``)
      or JSON otherwise.
    * ``config_path`` omitted -> env overrides on top of built-in defaults.
    """

    if config_path is not None:
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(f"config file not found: {path}")
        text = path.read_text(encoding="utf-8")
        raw = yaml.safe_load(text) if path.suffix in (".yaml", ".yml") else json.loads(text)
        if not isinstance(raw, dict):
            raise ValueError("config file must contain a mapping/object at the top level")
        return _coerce(raw)

    env: dict[str, object] = {}
    if "PRAG_STORAGE_DIR" in os.environ:
        env["storage_dir"] = os.environ["PRAG_STORAGE_DIR"]
    if "PRAG_MAX_UPLOAD_BYTES" in os.environ:
        env["max_upload_bytes"] = os.environ["PRAG_MAX_UPLOAD_BYTES"]
    if "PRAG_MAX_FILES_PER_UPLOAD" in os.environ:
        env["max_files_per_upload"] = os.environ["PRAG_MAX_FILES_PER_UPLOAD"]
    if "PRAG_CHUNK_SIZE" in os.environ:
        env["chunk_size"] = os.environ["PRAG_CHUNK_SIZE"]
    if "PRAG_CHUNK_OVERLAP" in os.environ:
        env["chunk_overlap"] = os.environ["PRAG_CHUNK_OVERLAP"]
    if "PRAG_SPLITTER" in os.environ:
        env["splitter"] = os.environ["PRAG_SPLITTER"]
    return _coerce(env)
