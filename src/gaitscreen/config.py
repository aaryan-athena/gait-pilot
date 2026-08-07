"""Configuration loading.

All tunable values -- especially every clinical threshold -- live in
``config/default.yaml`` rather than in code, so they can be reviewed and
changed without touching the pipeline. A user-supplied YAML file is deep-merged
over the defaults.

Config is dict-backed with dotted access (``cfg["flagging.absolute"]``,
``cfg.get("video.fps_warn_below")``) instead of a mirrored dataclass tree. With
this many thresholds, a mirror drifts out of sync with the YAML silently; a
dotted lookup that raises on an unknown key does not.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_YAML = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


class ConfigError(KeyError):
    """Raised when a requested config key does not exist."""


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


class Config:
    """Immutable-ish view over the merged configuration tree."""

    def __init__(self, data: dict[str, Any], source: Path | None = None):
        self._data = data
        self.source = source

    # -- construction ----------------------------------------------------
    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        """Load defaults, then deep-merge ``path`` over them if given."""
        with open(_DEFAULT_YAML, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        source = _DEFAULT_YAML
        if path is not None:
            path = Path(path)
            with open(path, encoding="utf-8") as fh:
                data = _deep_merge(data, yaml.safe_load(fh) or {})
            source = path
        return cls(data, source)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Build directly from a dict (used by tests)."""
        return cls(copy.deepcopy(data))

    def with_overrides(self, overrides: dict[str, Any]) -> "Config":
        """Return a copy with a nested dict deep-merged over this config."""
        return Config(_deep_merge(self._data, overrides), self.source)

    # -- access ----------------------------------------------------------
    def __getitem__(self, dotted: str) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                raise ConfigError(f"missing config key: {dotted!r}")
            node = node[part]
        return node

    def get(self, dotted: str, default: Any = None) -> Any:
        try:
            return self[dotted]
        except ConfigError:
            return default

    def section(self, dotted: str) -> "Config":
        node = self[dotted]
        if not isinstance(node, dict):
            raise ConfigError(f"config key {dotted!r} is not a section")
        return Config(node, self.source)

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def resolve_path(self, dotted: str, root: str | Path = ".") -> Path:
        """Resolve a path-valued config key against ``root`` if relative."""
        value = Path(self[dotted])
        return value if value.is_absolute() else Path(root) / value

    def __repr__(self) -> str:  # pragma: no cover
        return f"Config(source={self.source})"
