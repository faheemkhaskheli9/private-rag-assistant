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

_DEFAULT_MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MiB


@dataclass(frozen=True)
class Settings:
    storage_dir: Path = Path("data/store")
    max_upload_bytes: int = _DEFAULT_MAX_UPLOAD_BYTES

    def with_overrides(self, **kwargs: object) -> "Settings":
        clean = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **clean)  # type: ignore[arg-type]


def _coerce(raw: dict[str, object]) -> Settings:
    base = Settings()
    storage_dir = raw.get("storage_dir")
    max_upload = raw.get("max_upload_bytes")
    if max_upload is not None:
        max_upload = int(max_upload)
        if max_upload <= 0:
            raise ValueError("max_upload_bytes must be positive")
    return base.with_overrides(
        storage_dir=Path(storage_dir) if storage_dir is not None else None,
        max_upload_bytes=max_upload,
    )


def load_settings(config_path: str | os.PathLike[str] | None = None) -> Settings:
    """Load settings.

    * ``config_path`` given but missing -> :class:`FileNotFoundError`.
    * ``config_path`` given and present -> parsed (JSON).
    * ``config_path`` omitted -> env overrides on top of built-in defaults.
    """

    if config_path is not None:
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(f"config file not found: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("config file must contain a JSON object")
        return _coerce(raw)

    env: dict[str, object] = {}
    if "PRAG_STORAGE_DIR" in os.environ:
        env["storage_dir"] = os.environ["PRAG_STORAGE_DIR"]
    if "PRAG_MAX_UPLOAD_BYTES" in os.environ:
        env["max_upload_bytes"] = os.environ["PRAG_MAX_UPLOAD_BYTES"]
    return _coerce(env)
