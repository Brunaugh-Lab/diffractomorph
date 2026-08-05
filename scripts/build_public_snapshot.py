"""Build a clean, non-published DiffractoMorph public snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "release" / "public_snapshot_policy.json"
TEXT_SUFFIXES = {
    "", ".cff", ".cfg", ".csv", ".ini", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml",
}
MAX_PUBLIC_FILE_BYTES = 2_000_000
FORBIDDEN = {
    "absolute user path": re.compile("/" + "Users" + "/"),
    "Linux user path": re.compile("/" + "home" + r"/[^/\s]+/", re.I),
    "Windows user path": re.compile(
        r"(?:[A-Za-z]:)?\\" + "Users" + r"\\[^\\\s]+\\", re.I),
    "institutional Dropbox path": re.compile("Dropbox" + "-University"),
    "institutional email": re.compile(
        r"[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9-]+\.)*" + "umich" + r"\.edu", re.I),
    "operator initials": re.compile(r"\b" + "NS" + "/" + "SK" + r"\b"),
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "GitHub token": re.compile(r"(?:github_pat_|gh[pousr]_)[A-Za-z0-9_]{20,}"),
    "OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    "private key": re.compile(r"BEGIN (?:RSA|OPENSSH|EC|DSA) PRIVATE KEY"),
}


def _included(relative: Path, policy: dict) -> bool:
    name = relative.as_posix()
    if name in policy["root_files"]:
        return True
    if not any(name == root or name.startswith(root + "/") for root in policy["directory_roots"]):
        return False
    if any(name == prefix or name.startswith(prefix + "/") for prefix in policy["excluded_prefixes"]):
        return False
    if name.startswith("src/diffractomorph_pipeline/data/"):
        return any(name.startswith(prefix + "/") for prefix in policy["allowed_data_prefixes"])
    return True


def _source_files(policy: dict):
    for path in sorted(ROOT.rglob("*")):
        if path.is_dir() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(ROOT)
        if _included(relative, policy):
            yield path, relative


def _audit_text(path: Path, relative: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix not in TEXT_SUFFIXES:
        return [f"{relative}: unreviewed file suffix {suffix or '<none>'}"]
    if path.stat().st_size > MAX_PUBLIC_FILE_BYTES:
        return [f"{relative}: exceeds {MAX_PUBLIC_FILE_BYTES} byte public-file limit"]
    text = path.read_text(encoding="utf-8", errors="replace")
    return [f"{relative}: {label}" for label, pattern in FORBIDDEN.items() if pattern.search(text)]


def build_snapshot(output: Path) -> None:
    policy = json.loads(POLICY_PATH.read_text())
    missing_root_files = [name for name in policy["root_files"] if not (ROOT / name).is_file()]
    if missing_root_files:
        raise FileNotFoundError(
            "public snapshot policy requires missing root file(s): "
            + ", ".join(sorted(missing_root_files))
        )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    selected = list(_source_files(policy))
    findings: list[str] = []
    for source, relative in selected:
        if source.is_symlink():
            findings.append(f"{relative}: symlink")
            continue
        findings.extend(_audit_text(source, relative))
    if findings:
        raise RuntimeError("public snapshot disclosure check failed:\n" + "\n".join(findings))

    output.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for source, relative in selected:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(relative)
    manifest = []
    for relative in sorted(copied):
        digest = hashlib.sha256((output / relative).read_bytes()).hexdigest()
        manifest.append(f"{digest}  {relative.as_posix()}")
    (output / "PUBLIC_SNAPSHOT_MANIFEST.sha256").write_text("\n".join(manifest) + "\n")
    print(f"built clean public snapshot: {output} ({len(copied)} files)")


def verify_snapshot(snapshot: Path) -> None:
    manifest_path = snapshot / "PUBLIC_SNAPSHOT_MANIFEST.sha256"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    entries: dict[Path, str] = {}
    for line in manifest_path.read_text().splitlines():
        digest, name = line.split("  ", 1)
        entries[Path(name)] = digest
    actual = {
        path.relative_to(snapshot)
        for path in snapshot.rglob("*")
        if path.is_file()
        and path != manifest_path
        and ".git" not in path.relative_to(snapshot).parts
    }
    if actual != set(entries):
        missing = sorted(str(path) for path in set(entries) - actual)
        extra = sorted(str(path) for path in actual - set(entries))
        raise RuntimeError(f"snapshot manifest file-set mismatch; missing={missing}, extra={extra}")
    bad = [
        str(relative)
        for relative, expected in entries.items()
        if hashlib.sha256((snapshot / relative).read_bytes()).hexdigest() != expected
    ]
    if bad:
        raise RuntimeError("snapshot manifest digest mismatch: " + ", ".join(bad))
    print(f"verified public snapshot manifest: {snapshot} ({len(entries)} files)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument(
        "--verify-existing", type=Path, metavar="SNAPSHOT",
        help="Verify an existing clean snapshot against PUBLIC_SNAPSHOT_MANIFEST.sha256.",
    )
    args = parser.parse_args(argv)
    if (args.output is None) == (args.verify_existing is None):
        parser.error("provide exactly one of OUTPUT or --verify-existing SNAPSHOT")
    if args.verify_existing is not None:
        verify_snapshot(args.verify_existing.resolve())
    else:
        build_snapshot(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
