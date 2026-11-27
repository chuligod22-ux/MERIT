"""Fail-closed integrity, scope, and privacy checks for the public MERIT package."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests" / "MANIFEST.csv"
EXCLUDED_FROM_MANIFEST = {
    "data/manifests/MANIFEST.csv",
    "data/manifests/SHA256SUMS",
}
PROHIBITED_SUFFIXES = {".bin", ".ckpt", ".npy", ".npz", ".pt", ".pth", ".safetensors"}
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".sha256", ".toml", ".txt", ".yaml", ".yml"}
ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"(?i)[A-Z]:[\\/]+Users[\\/]"),
    re.compile(r"(?i)/" + r"home/[^/\s]+/"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def find_one(pattern: str) -> Path:
    found = list(ROOT.glob(pattern))
    if len(found) != 1:
        raise AssertionError(f"Expected one file for {pattern}, found {len(found)}")
    return found[0]


def check_manifest() -> list[str]:
    failures: list[str] = []
    with MANIFEST.open("r", encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    recorded = {row["path"]: row for row in records}
    actual = {
        path.relative_to(ROOT).as_posix(): path
        for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.parts
        and path.relative_to(ROOT).as_posix() not in EXCLUDED_FROM_MANIFEST
    }
    if set(recorded) != set(actual):
        missing = sorted(set(actual) - set(recorded))
        extra = sorted(set(recorded) - set(actual))
        failures.append(f"manifest membership mismatch missing={missing[:5]} extra={extra[:5]}")
    for relative in sorted(set(recorded) & set(actual)):
        path = actual[relative]
        row = recorded[relative]
        if int(row["size_bytes"]) != path.stat().st_size:
            failures.append(f"size mismatch: {relative}")
        if row["sha256"] != sha256(path):
            failures.append(f"hash mismatch: {relative}")
    return failures


def check_chart_archive(path: Path) -> None:
    expected = {"Degraded input", "Wiener (oracle PSF)", "Richardson–Lucy", "Unsupervised Wiener",
                "Wiener (wrong PSF)", "TV denoise", "NL-means", "Unsharp mask", "SD-x4",
                "DiffBIR v2.1", "NAFNet-big", "MERIT"}
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) != len(expected) or {item.filename for item in members} != {name + ".npy" for name in expected}:
            raise ValueError("Unexpected fixed-chart arrays")
        for member in members:
            with archive.open(member) as stream:
                if stream.read(8) != bytes.fromhex("93 4e 55 4d 50 59 01 00"):
                    raise ValueError("Unexpected fixed-chart array format")
                length = int.from_bytes(stream.read(2), "little")
                if length > 1024:
                    raise ValueError("Unexpected fixed-chart array header length")
                header = ast.literal_eval(stream.read(length).decode("ascii"))
                if header != {"descr": "<f8", "fortran_order": False, "shape": (396, 196)}:
                    raise ValueError("Unexpected fixed-chart array type or dimensions")
                if member.file_size != 10 + length + 396 * 196 * 8:
                    raise ValueError("Unexpected fixed-chart array size")


def check_scope() -> list[str]:
    failures: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(ROOT).as_posix()
        # This archive contains only the released, fixed scoring regions of the controlled chart.
        chart_archive = "round2/optics/chart/scoring_rois.npz"
        if relative == chart_archive:
            check_chart_archive(path)
        elif path.suffix.lower() in PROHIBITED_SUFFIXES:
            failures.append(f"prohibited model/data binary: {relative}")
        if path.stat().st_size >= 100 * 1024 * 1024:
            failures.append(f"Git object at or above 100 MiB: {relative}")
        if path.suffix.lower() in TEXT_SUFFIXES:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                failures.append(f"non-UTF-8 text: {relative}")
                continue
            if any(pattern.search(text) for pattern in ABSOLUTE_PATH_PATTERNS):
                failures.append(f"machine-local absolute path: {relative}")

    scope = json.loads((ROOT / "data" / "manifests" / "PUBLIC_SCOPE.json").read_text(encoding="utf-8"))
    if not scope["request_only"]["four_tunnel_source_images"]:
        failures.append("four-tunnel source images are not marked request-only")
    if scope["request_only"]["public_license_applies"]:
        failures.append("public license must not apply to four-tunnel inputs")

    cohort = find_one("data/processed/c007_d2_external_synthetic/**/cohort.csv")
    with cohort.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1703:
        failures.append(f"field cohort count {len(rows)} != 1703")
    if any(not row["image_path"].startswith("REQUEST_ONLY/") for row in rows):
        failures.append("public field cohort contains a non-request-only image path")
    if any(not row["mask_path"].startswith("REQUEST_ONLY/") for row in rows):
        failures.append("public field cohort contains a non-request-only mask path")

    expected_csv_rows = {
        "data/processed/c057_mtf50_estimator_v2/**/restoration_v2_roi_results.csv": 12159,
        "data/processed/c057_mtf50_estimator_v2/**/restoration_v2_complete_case_rois.csv": 1731,
        "protocols/c009_integrated_experiment/**/width_chart_roi_final_v1.csv": 500,
        "protocols/c063_full737_restoration_battery/**/full737_battery_manifest_v1.csv": 737,
    }
    for pattern, expected in expected_csv_rows.items():
        path = find_one(pattern)
        observed = csv_rows(path)
        if observed != expected:
            failures.append(f"row count {relative_or(path)}={observed}, expected {expected}")
    return failures


def relative_or(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def main() -> int:
    failures = check_manifest() + check_scope()
    if failures:
        print("PUBLIC PACKAGE VERIFICATION: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    with MANIFEST.open("r", encoding="utf-8", newline="") as stream:
        count = sum(1 for _ in csv.DictReader(stream))
    print("PUBLIC PACKAGE VERIFICATION: PASS")
    print(f"- manifest files: {count}")
    print("- field inputs: request-only; no image or mask pixels in data/processed")
    print("- key counts: cam1 12,159 run-ROI rows; 1,731 complete ROIs; 500 width ROIs; cam2 737 frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
