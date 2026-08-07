"""Explicit, path-stable project manifest for generic DiffractoMorph studies."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

import yaml

from diffractomorph_pipeline.io import get_reader
from diffractomorph_pipeline.model import Run

_SCHEMA = json.loads(
    resources.files("diffractomorph_pipeline")
    .joinpath("data/schema/project_manifest_v1.json")
    .read_text()
)
PROFILE_ROLES = tuple(_SCHEMA["properties"]["profiles"]["required"])
RUN_KINDS = tuple(_SCHEMA["$defs"]["run"]["properties"]["kind"]["enum"])
_TOP_LEVEL_FIELDS = frozenset(_SCHEMA["properties"])
_PROFILE_FIELDS = frozenset(_SCHEMA["$defs"]["profile"]["properties"])
_RUN_FIELDS = frozenset(_SCHEMA["$defs"]["run"]["properties"])


class ManifestError(ValueError):
    """A project manifest is incomplete, ambiguous, or internally inconsistent."""


def _verify_sha256(path: Path, expected: str, field_name: str) -> str:
    expected = expected.strip().lower()
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise ManifestError(f"{field_name} must be a 64-character hexadecimal SHA-256")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ManifestError(f"{field_name} mismatch for {path}: expected {expected}, got {actual}")
    return expected


def _json_safe(value: Any, field_name: str) -> Any:
    """Normalize YAML-native values to immutable JSON-compatible provenance."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item, f"{field_name}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, field_name) for item in value]
    raise ManifestError(f"{field_name} contains a non-JSON value of type {type(value).__name__}")


@dataclass(frozen=True)
class ProfileSpec:
    role: str
    profile_id: str
    path: Path | None = None
    sha256: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    not_applicable_reason: str | None = None

    @classmethod
    def from_mapping(cls, role: str, value: Any, base: Path) -> "ProfileSpec":
        if not isinstance(value, Mapping):
            raise ManifestError(f"profiles.{role} must be a mapping")
        unknown = set(value) - _PROFILE_FIELDS
        if unknown:
            raise ManifestError(f"profiles.{role} has unknown fields: {', '.join(sorted(unknown))}")
        profile_id = str(value.get("id", "")).strip()
        if not profile_id:
            raise ManifestError(f"profiles.{role}.id is required")
        path_value = value.get("path")
        path = (base / str(path_value)).resolve() if path_value is not None else None
        parameters = value.get("parameters") or {}
        if not isinstance(parameters, Mapping):
            raise ManifestError(f"profiles.{role}.parameters must be a mapping")
        reason = value.get("not_applicable_reason")
        if reason is not None and not str(reason).strip():
            raise ManifestError(f"profiles.{role}.not_applicable_reason must be non-empty")
        declarations = int(path_value is not None) + int("parameters" in value) + int(reason is not None)
        if declarations != 1:
            raise ManifestError(
                f"profiles.{role} must declare exactly one of path, parameters, or "
                "not_applicable_reason"
            )
        if "parameters" in value and not parameters:
            raise ManifestError(f"profiles.{role}.parameters must not be empty")
        if path is not None and not path.exists():
            raise ManifestError(f"profiles.{role}.path does not exist: {path}")
        if path is not None and not path.is_file():
            raise ManifestError(f"profiles.{role}.path must be a file: {path}")
        sha256 = None
        if "sha256" in value:
            if path is None:
                raise ManifestError(f"profiles.{role}.sha256 requires profiles.{role}.path")
            sha256 = _verify_sha256(path, str(value["sha256"]), f"profiles.{role}.sha256")
        if path is not None:
            try:
                loaded = json.loads(path.read_text()) if path.suffix.lower() == ".json" else yaml.safe_load(path.read_text())
            except (ValueError, yaml.YAMLError) as exc:
                raise ManifestError(f"profiles.{role}.path could not be parsed: {path}") from exc
            if not isinstance(loaded, Mapping) or not loaded:
                raise ManifestError(f"profiles.{role}.path must contain a non-empty mapping")
            parameters = loaded
        return cls(
            role=role,
            profile_id=profile_id,
            path=path,
            sha256=sha256,
            parameters=_json_safe(dict(parameters), f"profiles.{role}.parameters"),
            not_applicable_reason=str(reason) if reason else None,
        )


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    source: Path
    adapter: str
    run_kind: str
    sample_id: str
    independent_unit_id: str | None
    technical_replicate: str | None = None
    instrument_id: str | None = None
    sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Any, data_root: Path) -> "RunSpec":
        if not isinstance(value, Mapping):
            raise ManifestError("each runs entry must be a mapping")
        unknown = set(value) - _RUN_FIELDS
        if unknown:
            raise ManifestError(f"run has unknown fields: {', '.join(sorted(unknown))}")
        required = ("id", "source", "adapter", "kind", "sample_id")
        missing = [name for name in required if not str(value.get(name, "")).strip()]
        if missing:
            raise ManifestError(f"run is missing required fields: {', '.join(missing)}")
        source = (data_root / str(value["source"])).resolve()
        if not source.is_relative_to(data_root):
            raise ManifestError(f"run source escapes data_root: {source}")
        if not source.exists():
            raise ManifestError(f"run source does not exist: {source}")
        if not source.is_file():
            raise ManifestError(f"run source must be a file: {source}")
        sha256 = None
        if "sha256" in value:
            sha256 = _verify_sha256(source, str(value["sha256"]), f"runs.{value['id']}.sha256")
        run_kind = str(value["kind"])
        if run_kind not in RUN_KINDS:
            raise ManifestError(
                f"run {value['id']!r} kind must be one of {', '.join(RUN_KINDS)}; got {run_kind!r}"
            )
        independent_unit_id = value.get("independent_unit_id")
        if run_kind == "measurement" and not str(independent_unit_id or "").strip():
            raise ManifestError(
                f"measurement run {value['id']!r} requires independent_unit_id; "
                "hierarchical analysis cannot infer the independent unit from a filename"
            )
        metadata = value.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise ManifestError(f"run {value['id']!r} metadata must be a mapping")
        return cls(
            run_id=str(value["id"]),
            source=source,
            adapter=str(value["adapter"]),
            run_kind=run_kind,
            sample_id=str(value["sample_id"]),
            independent_unit_id=(
                str(independent_unit_id) if independent_unit_id is not None else None
            ),
            technical_replicate=(
                str(value["technical_replicate"])
                if value.get("technical_replicate") is not None else None
            ),
            instrument_id=str(value["instrument_id"]) if value.get("instrument_id") else None,
            sha256=sha256,
            metadata=_json_safe(dict(metadata), f"runs.{value['id']}.metadata"),
        )


@dataclass(frozen=True)
class ProjectManifest:
    manifest_path: Path
    schema_version: int
    project_id: str
    data_root: Path
    independent_unit: str
    profiles: Mapping[str, ProfileSpec]
    runs: tuple[RunSpec, ...]

    def require_profile(self, role: str) -> ProfileSpec:
        """Return a usable profile or fail explicitly when it is not applicable."""
        if role not in self.profiles:
            raise ManifestError(f"unknown profile role: {role}")
        profile = self.profiles[role]
        if profile.not_applicable_reason:
            raise ManifestError(
                f"profile {role!r} is not applicable: {profile.not_applicable_reason}"
            )
        return profile

    def _validate_loaded_run(self, spec: RunSpec, run: Run) -> Run:
        instrument = self.require_profile("instrument")
        expected = instrument.parameters.get("channel_ids")
        if expected is not None and tuple(str(value) for value in expected) != run.channel_ids:
            raise ManifestError(
                f"run {spec.run_id!r} channels do not match profiles.instrument.channel_ids"
            )
        return run

    def _read_spec(self, spec: RunSpec) -> Run:
        reader = get_reader(spec.adapter)
        instrument = self.require_profile("instrument")
        profile_aware_read = getattr(reader, "read_with_instrument_profile", None)
        if profile_aware_read is not None:
            run = profile_aware_read(spec, instrument.parameters)
        else:
            run = reader.read(spec)
        return self._validate_loaded_run(spec, run)

    def read_run(self, run_id: str) -> Run:
        """Read one run through the adapter explicitly declared in the manifest."""
        matches = [spec for spec in self.runs if spec.run_id == run_id]
        if not matches:
            raise KeyError(f"unknown run_id {run_id!r}")
        spec = matches[0]
        return self._read_spec(spec)

    def read_all_runs(self) -> tuple[Run, ...]:
        return tuple(self._read_spec(spec) for spec in self.runs)


def bundled_example_manifest() -> Path:
    """Path to the redistributable example included in source and wheel installs."""
    return Path(
        resources.files("diffractomorph_pipeline")
        .joinpath("data/examples/synthetic_minimal/project.yaml")
    )


def load_manifest(path: Path | str) -> ProjectManifest:
    """Load and validate a YAML project manifest.

    Relative paths resolve against the manifest, never the installed package or
    process working directory.
    """
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.exists():
        raise ManifestError(f"manifest does not exist: {manifest_path}")
    raw = yaml.safe_load(manifest_path.read_text())
    if not isinstance(raw, Mapping):
        raise ManifestError("manifest root must be a mapping")
    unknown_top = set(raw) - _TOP_LEVEL_FIELDS
    if unknown_top:
        raise ManifestError(f"manifest has unknown fields: {', '.join(sorted(unknown_top))}")
    if raw.get("schema_version") != 1:
        raise ManifestError("schema_version must be 1")
    project_id = str(raw.get("project_id", "")).strip()
    if not project_id:
        raise ManifestError("project_id is required")
    root_value = raw.get("data_root")
    if root_value is None:
        raise ManifestError("data_root is required and must be explicit")
    data_root = (manifest_path.parent / str(root_value)).resolve()
    if not data_root.is_dir():
        raise ManifestError(f"data_root is not a directory: {data_root}")
    independent_unit = str(raw.get("independent_unit", "")).strip()
    if not independent_unit:
        raise ManifestError("independent_unit is required")

    profile_values = raw.get("profiles")
    if not isinstance(profile_values, Mapping):
        raise ManifestError("profiles must be a mapping")
    missing_roles = [role for role in PROFILE_ROLES if role not in profile_values]
    extra_roles = [role for role in profile_values if role not in PROFILE_ROLES]
    if missing_roles:
        raise ManifestError(f"profiles missing required roles: {', '.join(missing_roles)}")
    if extra_roles:
        raise ManifestError(f"profiles contain unknown roles: {', '.join(extra_roles)}")
    profiles = {
        role: ProfileSpec.from_mapping(role, profile_values[role], manifest_path.parent)
        for role in PROFILE_ROLES
    }

    run_values = raw.get("runs")
    if not isinstance(run_values, list) or not run_values:
        raise ManifestError("runs must be a non-empty list")
    runs = tuple(RunSpec.from_mapping(value, data_root) for value in run_values)
    ids = [run.run_id for run in runs]
    if len(ids) != len(set(ids)):
        raise ManifestError("run ids must be unique")
    for spec in runs:
        try:
            get_reader(spec.adapter)
        except KeyError as exc:
            raise ManifestError(str(exc)) from exc
    instrument = profiles["instrument"]
    if instrument.not_applicable_reason:
        raise ManifestError("profiles.instrument cannot be not applicable")
    declared_adapter = instrument.parameters.get("adapter")
    if not declared_adapter:
        raise ManifestError("profiles.instrument.parameters.adapter is required")
    mismatched = [spec.run_id for spec in runs if spec.adapter != declared_adapter]
    if mismatched:
        raise ManifestError(
            "run adapters do not match profiles.instrument.parameters.adapter: "
            + ", ".join(mismatched)
        )
    return ProjectManifest(
        manifest_path=manifest_path,
        schema_version=1,
        project_id=project_id,
        data_root=data_root,
        independent_unit=independent_unit,
        profiles=profiles,
        runs=runs,
    )
