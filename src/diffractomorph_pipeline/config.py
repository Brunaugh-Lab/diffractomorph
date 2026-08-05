"""Legacy corpus-root compatibility and the public manifest entry point.

Precedence: ``DFM_DATA_ROOT`` env var → ``.dfm.toml`` at the repo root (key ``data_root``) → error.
New public workflows should call :func:`load_project`; ``data_root`` is retained for
the existing manuscript scripts until they migrate to a frozen study manifest.
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

from diffractomorph_pipeline.study.manifest import ProjectManifest, load_manifest


def load_project(path: Path | str) -> ProjectManifest:
    """Load the explicit public project manifest."""
    return load_manifest(path)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    """Legacy manuscript-corpus root (raises if unconfigured).

    This function is intentionally not used by the generic manifest workflow.
    """
    env = os.environ.get("DFM_DATA_ROOT")
    if env:
        return Path(env).expanduser()
    cfg = _repo_root() / ".dfm.toml"
    if cfg.exists():
        d = tomllib.loads(cfg.read_text())
        if d.get("data_root"):
            return Path(d["data_root"]).expanduser()
    raise RuntimeError(
        "data corpus root not configured: set DFM_DATA_ROOT, or add `data_root` to a "
        ".dfm.toml at the repository root.")


def data_root_or_none() -> Path | None:
    """``data_root()`` or None if unconfigured — for ``skipif`` guards in the corpus tests."""
    try:
        return data_root()
    except RuntimeError:
        return None


def corpus(*parts) -> Path:
    """``data_root()`` joined with ``parts``; a non-existent sentinel path if unconfigured,
    so ``.exists()`` is simply False (for ``skipif`` guards in the real-corpus tests)."""
    root = data_root_or_none() or Path("/__dfm_data_root_unset__")
    return root.joinpath(*parts)
