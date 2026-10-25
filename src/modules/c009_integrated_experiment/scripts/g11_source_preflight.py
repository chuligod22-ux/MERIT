"""Read-only preflight for locally vendored G11 official source snapshots.

The check deliberately does not import or execute upstream code.  It verifies
the exact Git commit, clean worktree, license/config/architecture presence,
expected class declarations, and a deterministic source-tree digest before an
isolated runtime is allowed to import the repositories.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
C009_ROOT = HERE.parent
DEFAULT_MANIFEST = C009_ROOT / "protocol" / "g11_implementation_manifest_v3.json"
DEFAULT_SOURCE_ROOT = C009_ROOT / "third_party"
DEFAULT_OUTPUT = C009_ROOT / "protocol" / "g11_source_preflight_v3.json"

SOURCE_SPECS = {
    "nafnet": {
        "directory": "NAFNet",
        "architecture_path": "basicsr/models/archs/NAFNet_arch.py",
        "expected_classes": ["NAFNet", "NAFNetLocal"],
        "license_candidates": ["LICENSE"],
    },
    "restormer": {
        "directory": "Restormer",
        "architecture_path": "basicsr/models/archs/restormer_arch.py",
        "expected_classes": ["Restormer"],
        "license_candidates": ["LICENSE.md", "LICENSE"],
    },
    "mprnet": {
        "directory": "MPRNet",
        "architecture_path": "Deblurring/MPRNet.py",
        "expected_classes": ["MPRNet"],
        "license_candidates": ["LICENSE.md", "LICENSE"],
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def git_text(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout.strip()


def declared_classes(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return sorted(
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef))
        and isinstance(node, ast.ClassDef)
    )


def source_files(repository: Path) -> list[Path]:
    excluded_directories = {".git", "__pycache__", ".pytest_cache"}
    paths: list[Path] = []
    for path in repository.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(repository)
        if any(part in excluded_directories for part in relative.parts):
            continue
        if path.suffix.lower() in {".pyc", ".pyo"}:
            continue
        paths.append(path)
    return sorted(paths, key=lambda item: item.relative_to(repository).as_posix())


def source_tree_sha256(repository: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    total_bytes = 0
    files = source_files(repository)
    for path in files:
        relative = path.relative_to(repository).as_posix().encode("utf-8")
        raw = path.read_bytes()
        digest.update(len(relative).to_bytes(8, byteorder="big"))
        digest.update(relative)
        digest.update(len(raw).to_bytes(8, byteorder="big"))
        digest.update(raw)
        total_bytes += len(raw)
    return digest.hexdigest(), len(files), total_bytes


def find_license(repository: Path, candidates: list[str]) -> Path | None:
    for relative in candidates:
        path = repository / relative
        if path.is_file():
            return path
    return None


def inspect_candidate(
    source_root: Path,
    candidate_name: str,
    candidate: dict[str, Any],
    spec: dict[str, Any],
) -> dict[str, Any]:
    repository = (source_root / spec["directory"]).resolve()
    source_root_resolved = source_root.resolve()
    if source_root_resolved not in repository.parents:
        raise RuntimeError(f"Source path escaped third_party: {repository}")
    record: dict[str, Any] = {
        "candidate": candidate_name,
        "repository": candidate["repository"],
        "expected_commit": candidate["commit"],
        "local_snapshot": str(repository),
        "status": "BLOCKED",
        "errors": [],
    }
    if not repository.is_dir():
        record["errors"].append("LOCAL_SNAPSHOT_MISSING")
        return record
    if not (repository / ".git").is_dir():
        record["errors"].append("GIT_METADATA_MISSING")
    else:
        try:
            actual_commit = git_text(repository, "rev-parse", "HEAD")
            record["actual_commit"] = actual_commit
            if actual_commit != candidate["commit"]:
                record["errors"].append("COMMIT_MISMATCH")
            worktree_status = git_text(repository, "status", "--porcelain")
            record["worktree_clean"] = not bool(worktree_status)
            if worktree_status:
                record["errors"].append("WORKTREE_DIRTY")
        except (OSError, subprocess.CalledProcessError) as error:
            record["errors"].append(f"GIT_CHECK_FAILED:{type(error).__name__}")

    config_path = repository / candidate["configuration_path"]
    architecture_path = repository / spec["architecture_path"]
    license_path = find_license(repository, spec["license_candidates"])
    record["configuration_path"] = str(config_path)
    record["architecture_path"] = str(architecture_path)
    if not config_path.is_file():
        record["errors"].append("CONFIGURATION_MISSING")
    else:
        record["configuration_sha256"] = sha256_file(config_path)
    if not architecture_path.is_file():
        record["errors"].append("ARCHITECTURE_SOURCE_MISSING")
    else:
        try:
            classes = declared_classes(architecture_path)
            record["declared_classes"] = classes
            missing_classes = sorted(set(spec["expected_classes"]) - set(classes))
            if missing_classes:
                record["missing_expected_classes"] = missing_classes
                record["errors"].append("EXPECTED_CLASS_MISSING")
            record["architecture_sha256"] = sha256_file(architecture_path)
        except (SyntaxError, UnicodeDecodeError) as error:
            record["errors"].append(
                f"ARCHITECTURE_PARSE_FAILED:{type(error).__name__}"
            )
    if license_path is None:
        record["errors"].append("LICENSE_MISSING")
    else:
        record["license_path"] = str(license_path)
        record["license_sha256"] = sha256_file(license_path)

    tree_hash, file_count, byte_count = source_tree_sha256(repository)
    record["source_tree_sha256"] = tree_hash
    record["source_file_count"] = file_count
    record["source_total_bytes"] = byte_count
    if not record["errors"]:
        record["status"] = "PASS"
    return record


def run_preflight(
    manifest_path: Path,
    source_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    candidates = manifest["official_candidates"]
    unexpected = sorted(set(candidates) - set(SOURCE_SPECS))
    missing = sorted(set(SOURCE_SPECS) - set(candidates))
    if unexpected or missing:
        raise RuntimeError(
            f"Candidate/spec mismatch: unexpected={unexpected}, missing={missing}"
        )
    records = [
        inspect_candidate(source_root, name, candidates[name], SOURCE_SPECS[name])
        for name in ("nafnet", "restormer", "mprnet")
    ]
    all_pass = all(record["status"] == "PASS" for record in records)
    output = {
        "schema_version": "MERIT-C009-G11-SOURCE-PREFLIGHT-v3",
        "status": "PASS" if all_pass else "BLOCKED",
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "source_root": str(source_root.resolve()),
        "preflight_path": str(Path(__file__).resolve()),
        "preflight_sha256": sha256_file(Path(__file__).resolve()),
        "upstream_imported_or_executed": False,
        "records": records,
        "g11_effect": (
            "Source snapshot subgate verified; dependency, import, parameter/FLOP, "
            "and forward/backward smoke gates remain separate."
            if all_pass
            else "Official source snapshot subgate remains blocked."
        ),
        "full_training_allowed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_preflight(
        args.manifest.resolve(), args.source_root.resolve(), args.output.resolve()
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
