"""Assemble the MERIT revision-v2 public tree from the internal revision workspace.

The script copies executable source, frozen protocols, row-level processed evidence,
and final paper figures. Raw tunnel imagery, masks, checkpoints, progress shards,
third-party source trees, caches, and machine-local paths are intentionally excluded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path


MODULES = (
    "c006_execution",
    "c007_d0_condition_eval",
    "c007_d2_external_synthetic",
    "c007_d2_multidetector",
    "c009_integrated_experiment",
    "c010_merit_v2_development",
    "c012_field_model",
    "c016_merit_v24_development",
    "c017_transformer_mamba_validation",
    "c018_merit_v25_degradation_robustness",
    "c039_c036_independent_confirmation",
    "c049_restormer_psnr_independent_confirmation",
    "c050_framework_backbone_factorial",
    "c051_framework_dominance",
    "c055_factorial_main_effect_confirmation",
    "c057_mtf50_estimator_v2",
    "c058_optical_fidelity_adapter",
    "c060_r2_3_table5",
    "c061_motion_deblur_development",
    "c063_full737_restoration_battery",
)

SOURCE_SUFFIXES = {".py", ".md"}
PROTOCOL_SUFFIXES = {".json", ".csv", ".sha256", ".yaml", ".yml", ".toml"}
EVIDENCE_SUFFIXES = {".json", ".csv", ".sha256"}
EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "cache",
    "checkpoints",
    "pretrained_models",
    "progress",
    "runtime",
    "third_party",
    "weights",
}
EXCLUDED_SUFFIXES = {".bin", ".ckpt", ".npy", ".npz", ".pt", ".pth", ".safetensors"}
WINDOWS_ABS = re.compile(r"(?i)[A-Z]:[\\/]+Users[\\/]+[^\"'\r\n,}]+")
FORWARD_ABS = re.compile(r"(?i)/" + r"home/[^\"'\r\n,}]+")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def allowed(path: Path) -> bool:
    lower_parts = {part.lower() for part in path.parts}
    return not (lower_parts & EXCLUDED_PARTS) and path.suffix.lower() not in EXCLUDED_SUFFIXES


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_module(source_root: Path, repository: Path, module: str) -> None:
    root = source_root / module
    if not root.is_dir():
        raise FileNotFoundError(root)

    for source in root.rglob("*"):
        if not source.is_file() or not allowed(source.relative_to(root)):
            continue
        relative = source.relative_to(root)
        suffix = source.suffix.lower()
        parts = {part.lower() for part in relative.parts}

        if suffix in SOURCE_SUFFIXES and (
            len(relative.parts) == 1 or "tests" in parts or suffix == ".py"
        ):
            copy_file(source, repository / "src" / "modules" / module / relative)

        if suffix in PROTOCOL_SUFFIXES and (
            "protocol" in parts or relative.name.lower().startswith("protocol")
        ):
            copy_file(source, repository / "protocols" / module / relative)

        if suffix in EVIDENCE_SUFFIXES and parts & {"artifacts", "evaluation", "runs"}:
            copy_file(source, repository / "data" / "processed" / module / relative)


def copy_figures(source_root: Path, repository: Path) -> None:
    root = source_root / "manuscript_r3_6"
    for source in root.iterdir():
        name = source.name.lower()
        if source.is_file() and (
            re.fullmatch(r"fig(?:[1-9]|1[0-5])\.(?:png|pdf)", name)
            or re.fullmatch(r"fig(?:[1-9]|1[0-5])_provenance\.json", name)
            or (name.startswith("plot_") and name.endswith(".py"))
        ):
            copy_file(source, repository / "figures" / source.name)


def redact_local_paths(repository: Path) -> int:
    changed = 0
    text_suffixes = {".csv", ".json", ".md", ".py", ".sha256", ".toml", ".txt", ".yaml", ".yml"}
    for path in repository.rglob("*"):
        if (
            not path.is_file()
            or ".git" in path.parts
            or "tools" in path.relative_to(repository).parts
            or path.suffix.lower() not in text_suffixes
        ):
            continue
        try:
            original = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        clean = WINDOWS_ABS.sub("<LOCAL_PATH_REDACTED>", original)
        clean = FORWARD_ABS.sub("<LOCAL_PATH_REDACTED>", clean)
        if clean:
            clean = clean.rstrip(" \t\r\n") + "\n"
        if clean != original:
            path.write_text(clean, encoding="utf-8", newline="")
            changed += 1
    return changed


def rewrite_request_only_cohort(repository: Path) -> None:
    candidates = list(repository.glob("data/processed/c007_d2_external_synthetic/**/cohort.csv"))
    for path in candidates:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            continue
        fields = list(rows[0])
        for row in rows:
            tunnel = row.get("tunnel", "X")
            name = row.get("name", "asset")
            row["image_path"] = f"REQUEST_ONLY/tunnel_{tunnel}/images/{name}"
            row["mask_path"] = f"REQUEST_ONLY/tunnel_{tunnel}/masks/{name}"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def write_manifest(repository: Path) -> None:
    records: list[dict[str, object]] = []
    for path in sorted(repository.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(repository).as_posix()
        if relative in {"data/manifests/MANIFEST.csv", "data/manifests/SHA256SUMS"}:
            continue
        if relative.startswith("src/") or relative.startswith("tools/"):
            license_id = "MIT"
        elif relative.startswith("data/") or relative.startswith("figures/"):
            license_id = "CC-BY-4.0"
        else:
            license_id = "SEE_FILE"
        records.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "license": license_id,
            }
        )

    manifest = repository / "data" / "manifests" / "MANIFEST.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "size_bytes", "sha256", "license"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    sums = repository / "data" / "manifests" / "SHA256SUMS"
    sums.write_text("".join(f"{row['sha256']}  {row['path']}\n" for row in records), encoding="utf-8")


def clean_generated_roots(repository: Path) -> None:
    for relative in ("src/modules", "protocols", "data/processed", "figures"):
        target = repository / relative
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    repository = args.repository.resolve()
    if not (repository / ".git").exists():
        raise RuntimeError(f"Not a Git staging repository: {repository}")

    clean_generated_roots(repository)
    for module in MODULES:
        copy_module(source_root, repository, module)
    copy_figures(source_root, repository)
    rewrite_request_only_cohort(repository)
    changed = redact_local_paths(repository)
    write_manifest(repository)
    print(json.dumps({"modules": len(MODULES), "redacted_files": changed}, indent=2))


if __name__ == "__main__":
    main()
