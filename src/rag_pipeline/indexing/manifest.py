"""Per-strategy active-manifest persistence.

Manifests are stored one-per-strategy (`<manifests_dir>/<strategy>.json`),
so fixed/recursive/semantic indexes can each have their own active
snapshot without overwriting one another.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import ChunkingStrategy, Settings
from .exceptions import ManifestError
from .models import IndexManifest


def manifest_path(settings: Settings, strategy: ChunkingStrategy) -> Path:
    return settings.manifests_dir / f"{strategy.value}.json"


def write_manifest(settings: Settings, manifest: IndexManifest) -> Path:
    """Atomically activate `manifest` as the active manifest for its strategy.

    Writes to a temporary file and renames it into place, so a crash or
    error mid-write never leaves a partially-written file mistaken for a
    valid active manifest, and never leaves the *previous* active manifest
    (if any) in a partially-overwritten state.
    """
    path = manifest_path(settings, manifest.chunking_strategy)
    payload = json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False)
    tmp_path = path.with_name(f"{path.name}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(payload, encoding="utf-8")
        tmp_path.replace(path)
    except OSError as exc:
        tmp_path.unlink(missing_ok=True)
        raise ManifestError(f"Failed to write manifest to {path}: {exc}") from exc
    return path


def load_manifest(settings: Settings, strategy: ChunkingStrategy) -> IndexManifest | None:
    """Load the active manifest for `strategy`, or `None` if none has been activated yet."""
    path = manifest_path(settings, strategy)
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestError(f"Failed to read manifest {path}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Manifest {path} is corrupt: {exc}") from exc
    return IndexManifest.from_dict(data)
