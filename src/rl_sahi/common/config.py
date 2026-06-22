from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml


T = TypeVar("T")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_RELATIVE = Path("configs") / "default.yaml"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / DEFAULT_CONFIG_RELATIVE
INCLUDE_KEYS = ("include", "includes")


class ProjectConfig:
    def __init__(self, path: Path, root: Path) -> None:
        self.path = Path(path)
        self.root = Path(root)
        self.data = load_yaml_config(self.path)

    def section(self, name: str) -> dict[str, Any]:
        value = self.data.get(name, {})
        if not isinstance(value, dict):
            raise ValueError(f"Config section [{name}] must be a mapping")
        return dict(value)

    def path_value(self, key: str) -> Path:
        value = self.section("paths")[key]
        path = Path(str(value)).expanduser()
        return path if path.is_absolute() else self.root / path

    def optional_str(self, section: str, key: str) -> str | None:
        value = self.section(section).get(key)
        if value is None or value == "":
            return None
        return str(value)

    def feature_layers(self, section: str) -> tuple[int, ...]:
        value = self.section(section).get("feature_layers", [10])
        if isinstance(value, str):
            return tuple(int(x.strip()) for x in value.split(",") if x.strip())
        return tuple(int(x) for x in value)

    def dataclass_kwargs(self, section: str, cls: type[T]) -> dict[str, Any]:
        if not is_dataclass(cls):
            raise TypeError(f"{cls} must be a dataclass type")
        allowed = {field.name for field in fields(cls)}
        values = self.section(section)
        return {key: value for key, value in values.items() if key in allowed}

    def dataclass_instance(self, section: str, cls: type[T]) -> T:
        return cls(**self.dataclass_kwargs(section, cls))


def load_yaml_config(path: Path) -> dict[str, Any]:
    return _load_yaml_config(path.resolve(), stack=())


def _load_yaml_config(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    if path in stack:
        chain = " -> ".join(str(p) for p in (*stack, path))
        raise ValueError(f"Circular config include detected: {chain}")
    data = _read_yaml_mapping(path)
    include_values = _pop_include_values(data, path)

    merged: dict[str, Any] = {}
    for include_value in include_values:
        include_path = Path(str(include_value)).expanduser()
        if not include_path.is_absolute():
            include_path = path.parent / include_path
        merged = _deep_merge(merged, _load_yaml_config(include_path.resolve(), (*stack, path)))

    return _deep_merge(merged, data)


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping")
    return dict(data)


def _pop_include_values(data: dict[str, Any], path: Path) -> list[str]:
    present = [key for key in INCLUDE_KEYS if key in data]
    if len(present) > 1:
        raise ValueError(f"Config file {path} must use only one of {INCLUDE_KEYS}")
    if not present:
        return []

    raw = data.pop(present[0])
    if raw is None:
        return []
    if isinstance(raw, (str, Path)):
        return [str(raw)]
    if isinstance(raw, list):
        if not all(isinstance(value, (str, Path)) for value in raw):
            raise ValueError(f"Config include list in {path} must contain only strings")
        return [str(value) for value in raw]
    raise ValueError(f"Config include in {path} must be a string or list of strings")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path, root: Path) -> ProjectConfig:
    return ProjectConfig(path=path, root=root)


def resolve_config_path(path: Path | str | None = None, root: Path | str | None = None) -> tuple[Path, Path]:
    project_root = Path(root).resolve() if root is not None else PROJECT_ROOT
    config_path = project_root / DEFAULT_CONFIG_RELATIVE if path is None else Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = project_root / config_path
    return config_path, project_root


def load_default_config(path: Path | str | None = None, root: Path | str | None = None) -> ProjectConfig:
    config_path, project_root = resolve_config_path(path, root)
    return load_config(config_path, project_root)
