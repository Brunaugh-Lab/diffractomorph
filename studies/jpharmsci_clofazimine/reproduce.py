"""Preflight, render, and verify the restricted JPharmSci reproduction bundle.

This driver never copies source data and never treats a successful render as permission to
redistribute the corpus. The manifest deliberately separates pipeline-generated artifacts from
externally assembled or renamed manuscript files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


STUDY_ROOT = Path(__file__).resolve().parent
REPO_ROOT = STUDY_ROOT.parents[1]
ROOT = REPO_ROOT  # compatibility name used by the hardened path checks
DEFAULT_MANIFEST = STUDY_ROOT / "manifest.json"
ALLOWED_PLACEHOLDERS = {
    "{ph_study}", "{arm_a}", "{loading}", "{arm_b}", "{recipe_output}",
}
ALLOWED_ARTIFACT_CLASSES = {
    "pipeline-generated",
    "pipeline-generated-renamed",
    "pipeline-generated-inputs-withheld",
    "study-generated-inputs-withheld",
    "writing-repo-assembly",
    "writing-repo-component",
}
FROZEN_ARM_B_SCIENCE_STATE = "frozen-copt-reference-before-summed-intensity-revision"
TEXT_OUTPUT_SUFFIXES = {".csv", ".json", ".md", ".txt"}
FORBIDDEN_OUTPUT_BYTES = (
    b"/" + b"users/",
    b"/" + b"home/",
    b"\\" + b"users\\",
    b"dropbox-university",
)
INSTITUTIONAL_EMAIL_BYTES = re.compile(
    rb"[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9-]+\.)*umich\.edu", re.I,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_state(manifest_path: Path, manifest: dict) -> dict:
    try:
        manifest_label = str(manifest_path.relative_to(ROOT))
    except ValueError:
        manifest_label = manifest_path.name
    files = {manifest_label: sha256(manifest_path)}
    files[str(Path(__file__).resolve().relative_to(REPO_ROOT))] = sha256(Path(__file__).resolve())
    for recipe in manifest["recipes"]:
        path = STUDY_ROOT / recipe["script"]
        files[str(path.relative_to(REPO_ROOT))] = sha256(path)
    for base in (STUDY_ROOT / "analysis", STUDY_ROOT / "figures",
                 REPO_ROOT / "src" / "diffractomorph_pipeline"):
        for path in sorted(base.rglob("*.py")):
            files[str(path.relative_to(REPO_ROOT))] = sha256(path)
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    try:
        actual_commit = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
        worktree_clean = not subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True).strip()
    except Exception:
        actual_commit = None
        worktree_clean = False
    try:
        from diffractomorph_pipeline import __version__ as pipeline_version
    except Exception:
        pipeline_version = None
    return {
        "declared_pipeline_commit": manifest.get("pipeline_commit"),
        "actual_pipeline_commit": actual_commit,
        "declared_commit_matches_head": actual_commit == manifest.get("pipeline_commit"),
        "worktree_clean": worktree_clean,
        "diffractomorph_pipeline_version": pipeline_version,
        "file_sha256": files,
        "digest": hashlib.sha256(canonical).hexdigest(),
    }


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported reproduction manifest schema")
    if manifest.get("status") != "restricted-local-draft":
        raise ValueError("this driver accepts only the reviewed restricted-local-draft manifest")
    ids = [recipe.get("id") for recipe in manifest.get("recipes", [])]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("recipe ids must be present and unique")
    if not re.fullmatch(r"[0-9a-f]{40}", manifest.get("pipeline_commit", "")):
        raise ValueError("pipeline_commit must be a full lowercase Git SHA")
    if not re.fullmatch(r"[0-9a-f]{40}", manifest.get("manuscript_commit", "")):
        raise ValueError("manuscript_commit must be a full lowercase Git SHA")
    science_state = manifest.get("migration_source", {}).get("science_state")
    if science_state != FROZEN_ARM_B_SCIENCE_STATE:
        raise ValueError(
            f"migration science_state must preserve {FROZEN_ARM_B_SCIENCE_STATE!r}"
        )
    for key, value in manifest.get("data_layout", {}).items():
        layout = Path(value)
        if layout.is_absolute() or ".." in layout.parts:
            raise ValueError(f"unsafe data-layout path for {key}: {layout}")
    for key, value in manifest.get("profiles", {}).items():
        if key not in {"DFM_ASSAY_PROFILE", "DFM_SOLUBILITY_PROFILE", "DFM_NOISE_SURFACE",
                       "DFM_OPTICAL_KERNEL"}:
            raise ValueError(f"unsupported study profile variable: {key}")
        profile = Path(value)
        if profile.is_absolute() or ".." in profile.parts:
            raise ValueError(f"unsafe study profile path for {key}: {profile}")
    for recipe in manifest["recipes"]:
        script = Path(recipe["script"])
        if script.is_absolute() or ".." in script.parts:
            raise ValueError(f"unsafe recipe script path: {script}")
        if not _inside(STUDY_ROOT / script, STUDY_ROOT):
            raise ValueError(f"recipe script resolves outside study root: {script}")
        for value in recipe.get("arguments", []):
            if isinstance(value, str) and ("{" in value or "}" in value) \
                    and value not in ALLOWED_PLACEHOLDERS:
                raise ValueError(f"unsupported or embedded recipe placeholder: {value}")
        for output in recipe.get("expected_outputs", []):
            path_out = Path(output)
            if path_out.is_absolute() or ".." in path_out.parts:
                raise ValueError(f"unsafe expected output path: {path_out}")
    for artifact in manifest.get("reference_artifacts", []):
        path_ref = Path(artifact["path"])
        if path_ref.is_absolute() or ".." in path_ref.parts:
            raise ValueError(f"unsafe reference artifact path: {path_ref}")
        if len(artifact.get("sha256", "")) != 64:
            raise ValueError(f"invalid reference digest: {path_ref}")
        artifact_class = artifact.get("class", "")
        generator = artifact.get("generator")
        if artifact_class not in ALLOWED_ARTIFACT_CLASSES:
            raise ValueError(f"unsupported artifact class {artifact_class!r}: {path_ref}")
        if artifact_class.startswith("pipeline-generated") and not generator:
            raise ValueError(f"pipeline-generated artifact lacks generator: {path_ref}")
        if artifact_class in {"pipeline-generated", "pipeline-generated-renamed"} \
                and generator not in ids:
            raise ValueError(f"unknown recipe generator {generator!r}: {path_ref}")
        if artifact_class == "pipeline-generated-renamed" and generator in ids:
            generated_path = artifact.get("generated_path")
            recipe = next(item for item in manifest["recipes"] if item["id"] == generator)
            if generated_path not in recipe["expected_outputs"]:
                raise ValueError(
                    f"generated_path {generated_path!r} is not an output of {generator}: {path_ref}"
                )
        if artifact_class == "pipeline-generated-inputs-withheld" and generator:
            if not re.fullmatch(r"diffractomorph_pipeline(?:\.[A-Za-z_]\w*)+", generator):
                raise ValueError(f"unsafe module generator {generator!r}: {path_ref}")
            module_path = REPO_ROOT / "src" / (generator.replace(".", "/") + ".py")
            if not _inside(module_path, REPO_ROOT / "src") or not module_path.is_file():
                raise ValueError(f"unresolvable module generator {generator!r}: {path_ref}")
        if artifact_class == "study-generated-inputs-withheld":
            generator_path = Path(generator or "")
            if (not generator or generator_path.is_absolute() or ".." in generator_path.parts
                    or generator_path.suffix != ".py"):
                raise ValueError(f"unsafe study generator {generator!r}: {path_ref}")
            module_path = STUDY_ROOT / generator_path
            if not _inside(module_path, STUDY_ROOT) or not module_path.is_file():
                raise ValueError(f"unresolvable study generator {generator!r}: {path_ref}")
        components = artifact.get("components", [])
        if not isinstance(components, list):
            raise ValueError(f"artifact components must be a list: {path_ref}")
        for component in components:
            if not isinstance(component, dict):
                raise ValueError(f"artifact component must be an object: {path_ref}")
            component_generator = component.get("generator")
            generated_path = component.get("generated_path")
            if component_generator not in ids:
                raise ValueError(
                    f"unknown component generator {component_generator!r}: {path_ref}"
                )
            recipe = next(
                item for item in manifest["recipes"] if item["id"] == component_generator
            )
            if generated_path not in recipe["expected_outputs"]:
                raise ValueError(
                    f"component path {generated_path!r} is not an output of "
                    f"{component_generator}: {path_ref}"
                )
    return manifest


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def preflight(manifest: dict, data_root: Path) -> list[dict]:
    checks: list[dict] = []
    checks.append({"check": "data_root", "path": str(data_root), "ok": data_root.is_dir()})
    for key, relative in manifest["data_layout"].items():
        path = data_root / relative
        checks.append({"check": f"data_layout:{key}", "path": str(path),
                       "ok": path.is_dir() and _inside(path, data_root)})
    for key, relative in manifest.get("profiles", {}).items():
        path = data_root / relative
        checks.append({"check": f"profile:{key}", "path": str(path),
                       "ok": path.is_file() and _inside(path, data_root)})
    for recipe in manifest["recipes"]:
        path = STUDY_ROOT / recipe["script"]
        checks.append({"check": f"recipe:{recipe['id']}", "path": str(path), "ok": path.is_file()})
        for relative in recipe.get("required_inputs", []):
            path = data_root / relative
            checks.append({"check": f"input:{recipe['id']}:{relative}",
                           "path": str(path),
                           "ok": path.exists() and _inside(path, data_root)})
        for pattern in recipe.get("required_globs", []):
            matches = sorted(data_root.glob(pattern))
            checks.append({"check": f"input-glob:{recipe['id']}:{pattern}",
                           "matches": len(matches),
                           "unsafe_matches": sum(not _inside(path, data_root) for path in matches),
                           "ok": bool(matches) and all(_inside(path, data_root) for path in matches)})
    return checks


def verify_reference(manifest: dict, manuscript_root: Path) -> list[dict]:
    results: list[dict] = []
    for artifact in manifest["reference_artifacts"]:
        path = manuscript_root / artifact["path"]
        path_safe = _inside(path, manuscript_root)
        actual = sha256(path) if path.is_file() and path_safe else None
        results.append({
            "path": artifact["path"],
            "class": artifact["class"],
            "expected_sha256": artifact["sha256"],
            "actual_sha256": actual,
            "path_safe": path_safe,
            "ok": path_safe and actual == artifact["sha256"],
        })
    return results


def manuscript_state(manifest: dict, manuscript_root: Path) -> dict:
    try:
        actual_commit = subprocess.check_output(
            ["git", "-C", str(manuscript_root), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        actual_commit = None
    declared_commit = manifest.get("manuscript_commit")
    return {
        "declared_manuscript_commit": declared_commit,
        "actual_manuscript_commit": actual_commit,
        "declared_commit_matches_head": actual_commit == declared_commit,
    }


def _expanded_arguments(recipe: dict, *, data_root: Path, recipe_output: Path,
                        data_layout: dict) -> list[str]:
    values = {"{recipe_output}": str(recipe_output)}
    for key in ("ph_study", "arm_a", "loading", "arm_b"):
        placeholder = "{" + key + "}"
        if placeholder in recipe["arguments"]:
            if key not in data_layout:
                raise ValueError(f"recipe uses {placeholder} but data_layout.{key} is absent")
            values[placeholder] = str(data_root / data_layout[key])
    return [values.get(value, value) for value in recipe["arguments"]]


def _scrub_text(value: str, *, data_root: Path, output_dir: Path) -> str:
    scrubbed = value.replace(str(data_root), "${DFM_DATA_ROOT}").replace(
        str(output_dir), "${OUTPUT_DIR}").replace(str(REPO_ROOT), "${PIPELINE_ROOT}")
    scrubbed = re.sub(r"/" + r"Users/[^/\s`]+", "${USER_HOME}", scrubbed, flags=re.I)
    scrubbed = re.sub(r"/" + r"home/[^/\s`]+", "${USER_HOME}", scrubbed, flags=re.I)
    return re.sub(r"(?:[A-Za-z]:)?\\" + r"Users\\[^\\\s`]+", "${USER_HOME}", scrubbed,
                  flags=re.I)


def _portable_text_outputs(recipe_output: Path, data_root: Path) -> list[dict]:
    """Replace the caller's corpus root in text deliverables, then reject disclosure markers."""
    findings: list[dict] = []
    for path in sorted(recipe_output.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in TEXT_OUTPUT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
            portable = _scrub_text(text, data_root=data_root, output_dir=recipe_output)
            if portable != text:
                path.write_text(portable, encoding="utf-8")
        payload = path.read_bytes().lower()
        hits = [marker.decode() for marker in FORBIDDEN_OUTPUT_BYTES if marker in payload]
        if INSTITUTIONAL_EMAIL_BYTES.search(payload):
            hits.append("institutional umich email")
        if path.suffix.lower() == ".pdf":
            try:
                import pdfplumber
                with pdfplumber.open(path) as pdf:
                    extracted = "\n".join(page.extract_text() or "" for page in pdf.pages)
                    extracted += "\n" + json.dumps(pdf.metadata or {}, default=str)
                extracted_bytes = extracted.encode("utf-8", errors="replace").lower()
                hits.extend(marker.decode() + " (PDF text/metadata)"
                            for marker in FORBIDDEN_OUTPUT_BYTES if marker in extracted_bytes)
                if INSTITUTIONAL_EMAIL_BYTES.search(extracted_bytes):
                    hits.append("institutional umich email (PDF text/metadata)")
            except Exception as exc:
                hits.append(f"PDF text scan failed: {type(exc).__name__}")
        if hits:
            findings.append({"path": str(path.relative_to(recipe_output)),
                             "markers": sorted(set(hits))})
    return findings


def _declared_input_state(recipe: dict, data_root: Path) -> dict[str, str]:
    """Hash every declared input file so a recipe cannot silently modify its corpus."""
    paths: set[Path] = set()
    for relative in recipe.get("required_inputs", []):
        candidate = data_root / relative
        if candidate.is_file():
            paths.add(candidate)
        elif candidate.is_dir():
            paths.update(path for path in candidate.rglob("*") if path.is_file())
    for pattern in recipe.get("required_globs", []):
        for candidate in data_root.glob(pattern):
            if candidate.is_file():
                paths.add(candidate)
            elif candidate.is_dir():
                paths.update(path for path in candidate.rglob("*") if path.is_file())
    return {str(path.relative_to(data_root)): sha256(path) for path in sorted(paths)}


def _portable(value: str, *, data_root: Path, output_dir: Path) -> str:
    return _scrub_text(value, data_root=data_root, output_dir=output_dir)


def _portable_record(value, *, data_root: Path, output_dir: Path):
    if isinstance(value, str):
        return _portable(value, data_root=data_root, output_dir=output_dir)
    if isinstance(value, list):
        return [_portable_record(item, data_root=data_root, output_dir=output_dir)
                for item in value]
    if isinstance(value, dict):
        return {key: _portable_record(item, data_root=data_root, output_dir=output_dir)
                for key, item in value.items()}
    return value


def render_recipes(manifest: dict, data_root: Path, output_dir: Path,
                   selected: set[str] | None = None) -> list[dict]:
    env = os.environ.copy()
    env["DFM_DATA_ROOT"] = str(data_root)
    for key, relative in manifest.get("profiles", {}).items():
        env[key] = str(data_root / relative)
    env.setdefault("MPLBACKEND", "Agg")
    results: list[dict] = []
    for recipe in manifest["recipes"]:
        if selected is not None and recipe["id"] not in selected:
            continue
        recipe_output = output_dir / recipe["id"]
        recipe_output.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(STUDY_ROOT / recipe["script"]), *_expanded_arguments(
            recipe, data_root=data_root, recipe_output=recipe_output,
            data_layout=manifest.get("data_layout", {}))]
        input_state_before = _declared_input_state(recipe, data_root)
        # Run from the recipe destination so any incidental relative output is confined there.
        # Manuscript scripts are invoked by absolute path and import their siblings from analysis/.
        study_pythonpath = os.pathsep.join((str(STUDY_ROOT), str(STUDY_ROOT / "analysis")))
        env["PYTHONPATH"] = os.pathsep.join(
            value for value in (study_pythonpath, env.get("PYTHONPATH", "")) if value
        )
        completed = subprocess.run(command, cwd=recipe_output, env=env, text=True,
                                   capture_output=True)
        input_state_after = _declared_input_state(recipe, data_root)
        inputs_unchanged = input_state_before == input_state_after
        disclosure_findings = _portable_text_outputs(recipe_output, data_root)
        outputs = []
        for relative in recipe["expected_outputs"]:
            path = recipe_output / relative
            outputs.append({"path": relative, "exists": path.is_file(),
                            "sha256": sha256(path) if path.is_file() else None})
        results.append({
            "id": recipe["id"],
            "command": [_portable(value, data_root=data_root, output_dir=output_dir)
                        for value in command],
            "returncode": completed.returncode,
            "stdout": _portable(completed.stdout, data_root=data_root, output_dir=output_dir),
            "stderr": _portable(completed.stderr, data_root=data_root, output_dir=output_dir),
            "outputs": outputs,
            "declared_input_files_hashed": len(input_state_before),
            "declared_inputs_unchanged": inputs_unchanged,
            "disclosure_findings": disclosure_findings,
            "ok": completed.returncode == 0 and all(item["exists"] for item in outputs)
                  and not disclosure_findings and inputs_unchanged,
        })
    return results


def _all_ok(items: list[dict]) -> bool:
    return bool(items) and all(item["ok"] for item in items)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manuscript-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "verify-reference", "render", "all"),
                        default="all")
    parser.add_argument("--recipe", action="append", help="Render only this recipe id; repeatable.")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest.resolve())
    state = code_state(args.manifest.resolve(), manifest)
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    known = {recipe["id"] for recipe in manifest["recipes"]}
    selected = set(args.recipe) if args.recipe else None
    if selected and not selected <= known:
        parser.error("unknown recipe id(s): " + ", ".join(sorted(selected - known)))
    if args.mode in {"verify-reference", "all"} and args.manuscript_root is None:
        parser.error("--manuscript-root is required for reference verification")

    report = {
        "bundle_id": manifest["bundle_id"],
        "status": manifest["status"],
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "selected_recipe_ids": sorted(selected) if selected else sorted(known),
        "scope": "complete" if args.mode == "all" and selected is None else "partial",
        "pipeline_root": "${PIPELINE_ROOT}",
        "data_root": "${DFM_DATA_ROOT}",
        "publication_authorized": False,
        "code_state": state,
    }
    report["preflight"] = preflight(manifest, data_root)
    if args.mode in {"verify-reference", "all"}:
        report["manuscript_state"] = manuscript_state(
            manifest, args.manuscript_root.resolve())
        report["reference_verification"] = verify_reference(
            manifest, args.manuscript_root.resolve())
    if args.mode in {"render", "all"} and _all_ok(report["preflight"]):
        report["recipes"] = render_recipes(manifest, data_root, output_dir, selected)
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    groups = [report["preflight"]]
    for key in ("reference_verification", "recipes"):
        if key in report:
            groups.append(report[key])
    report["requested_checks_ok"] = bool(
        state["worktree_clean"]
        and report.get("manuscript_state", {}).get("declared_commit_matches_head", True)
        and all(_all_ok(group) for group in groups)
    )
    report["ok"] = bool(report["scope"] == "complete" and report["requested_checks_ok"])
    scope_label = "complete" if report["scope"] == "complete" else "partial"
    selection_label = "all" if selected is None else hashlib.sha256(
        "\n".join(sorted(selected)).encode()).hexdigest()[:8]
    report_path = output_dir / (
        f"reproduction_report_{args.mode}_{scope_label}_{selection_label}_{state['digest'][:12]}.json"
    )
    portable_report = _portable_record(report, data_root=data_root, output_dir=output_dir)
    report_path.write_text(json.dumps(portable_report, indent=2) + "\n")
    print(f"wrote {report_path}")
    if report["ok"]:
        print("PASS: complete bundle")
    elif report["requested_checks_ok"]:
        print("PASS: requested checks; bundle scope is partial")
    else:
        print("FAIL")
    return 0 if report["requested_checks_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
