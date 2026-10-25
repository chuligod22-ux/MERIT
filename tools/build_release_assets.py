"""Build the large MERIT revision-v2 GitHub release assets.

The release assets are written outside the Git working tree. Controlled cam1 and
cam2 images are decoded and re-encoded as pixel-exact analysis crops so that
machine metadata and unused image regions are not redistributed. Printed-width
inputs are copied byte-for-byte because the complete images are the analysis
inputs. Four-tunnel field images and masks are never read by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image


CAM2_MANIFEST = Path(
    "c063_full737_restoration_battery/protocol/full737_battery_manifest_v1.csv"
)
CAM1_COHORT = Path(
    "c007_d0_condition_eval/artifacts/cam1_v1/d0_cam1_cohort.csv"
)
WIDTH_LEDGER = Path(
    "c009_integrated_experiment/protocol/width_chart_source_ledger.csv"
)
CAM1_MARGIN = 64
CAM1_MIN_SIZE = 448
CAM2_TOP_FRACTION = 0.40
CAM2_HEIGHT = 896
CAM2_WIDTH = 1344
ARCHIVE_TARGET_BYTES = 1_400_000_000
GITHUB_ASSET_LIMIT_BYTES = 2 * 1024**3

CHECKPOINTS = (
    (
        "restoration/task_off_seed0.pt",
        "c006_execution/runs/final/task_off_seed0/final.pt",
        "10466a39dbb50d0d4b959b79e6c802ef47177f406f01869980668c11cd8d857c",
    ),
    (
        "restoration/task_on_seed0.pt",
        "c006_execution/runs/final/task_on_seed0/final.pt",
        "10f105d19280f1f91a6022eeb51f05a17c5596c3d9261cbff6bfa561d4bdc65d",
    ),
    (
        "restoration/task_off_seed1.pt",
        "c006_execution/runs/final/task_off_seed1/final.pt",
        "f85662536c91977ed3d14b9a8d89aaf0da0ea74d0e7910e6076796cf159e368b",
    ),
    (
        "restoration/task_on_seed1.pt",
        "c006_execution/runs/final/task_on_seed1/final.pt",
        "8e5f1482376a8e4061627c0450f65c56c771aeeb6944c9e012e4b4092337dffd",
    ),
    (
        "restoration/task_off_seed2.pt",
        "c006_execution/runs/final/task_off_seed2/final.pt",
        "507038a52ba73cd6a8413a6c4be86e227ba97f0e97e63b8a35239cf472be8d4e",
    ),
    (
        "restoration/task_on_seed2.pt",
        "c006_execution/runs/final/task_on_seed2/final.pt",
        "7980d0034f81355d4a5bab09453bc9d7bf7ce6460510977c730c935ddee82107",
    ),
    (
        "detector/best_d2_resnet34_unet.pt",
        "c005_freeze/d2_detector/artifacts/d2_training_run/best_d2_resnet34_unet.pt",
        "aa5cf2bb42d042c9b3c4b2e92553844bc2ef79d8b8cbb30f72f7107f2c9393bc",
    ),
)

README_TEXT = {
    "revision-analysis-v2": """# MERIT revision analysis package

Snapshot of the public revision repository without Git metadata. It contains
analysis code, frozen protocols, anonymized row-level results, final figures,
and package verification tools. Four-tunnel image and mask pixels are absent.
""",
    "revision-cam2-crops-v2": """# MERIT cam2 fixed analysis crops

Lossless grayscale 896 x 1344 crops for all 737 controlled cam2 frames. Each
crop is centered horizontally and vertically within the upper 40% of the source
frame, exactly as used by the full-dataset restoration comparison. The manifest
records the source byte hash, crop coordinates, and public crop hash.
""",
    "revision-cam1-optical-crops-v2": """# MERIT cam1 optical context crops

Lossless grayscale union-context crops for 479 controlled cam1 frames and the
1,737 adjusted chart-ROI coordinates used by the optical analysis. The context
is the union of a frame's ROIs with a 64-pixel margin, clipped to sensor bounds.
""",
    "revision-width-inputs-v2": """# MERIT printed-width inputs

The 50 complete controlled printed-target images used by the revised width
analysis. Files are copied byte-for-byte; the manifest verifies source and
released hashes. These controlled targets are separate from the request-only
four-tunnel field images and masks.
""",
    "revision-models-essential-v2": """# MERIT essential controlled-analysis models

Six author-trained restoration checkpoints (task-OFF/task-ON, three seeds) and
the frozen ResNet34-U-Net detector used for the controlled cam1/cam2 analyses.
Field-study pixels are request-only, so the much larger detector-family and
factorial checkpoint banks are not redistributed in this public release.
""",
    "revision-samples-v2": """# MERIT quick-look controlled samples

Five cam1 optical context crops and ten cam2 fixed analysis crops selected by
sorted manifest order. They provide a small inspection set without duplicating
or disclosing any request-only four-tunnel field image or mask.
""",
}


class BuildError(RuntimeError):
    pass


def sha256_bytes(data: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="")


def decode_bytes(path: Path, mode: int) -> tuple[bytes, np.ndarray]:
    raw = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), mode)
    if image is None:
        raise BuildError(f"Could not decode image: {path}")
    return raw, image


def encode_png(path: Path, image: np.ndarray) -> tuple[int, str]:
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise BuildError(f"Could not encode PNG: {path}")
    data = encoded.tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return len(data), sha256_bytes(data)


def require_hash(label: str, observed: str, expected: str) -> None:
    if observed.lower() != expected.lower():
        raise BuildError(f"SHA-256 mismatch for {label}: {observed} != {expected}")


def prepare_release(staging_root: Path, name: str) -> Path:
    target = staging_root / name
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    write_text(target / "README.md", README_TEXT[name])
    return target


def build_cam2(revision_root: Path, staging_root: Path) -> Path:
    name = "revision-cam2-crops-v2"
    target = prepare_release(staging_root, name)
    rows = read_csv(revision_root / CAM2_MANIFEST)
    if len(rows) != 737:
        raise BuildError(f"Expected 737 cam2 rows, found {len(rows)}")
    output_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        source = Path(row["image_path"])
        raw, bgr = decode_bytes(source, cv2.IMREAD_COLOR)
        source_hash = sha256_bytes(raw)
        require_hash(row["frame"], source_hash, row["image_sha256"])
        crop_rows = int(np.floor(bgr.shape[0] * CAM2_TOP_FRACTION))
        x0 = (bgr.shape[1] - CAM2_WIDTH) // 2
        y0 = (crop_rows - CAM2_HEIGHT) // 2
        if min(x0, y0) < 0:
            raise BuildError(f"Source is smaller than fixed cam2 crop: {source}")
        crop = cv2.cvtColor(
            bgr[y0 : y0 + CAM2_HEIGHT, x0 : x0 + CAM2_WIDTH],
            cv2.COLOR_BGR2GRAY,
        )
        relative = Path("crops") / row["condition_cell"] / f"{int(row['ordinal']):04d}_{row['frame']}"
        size, public_hash = encode_png(target / relative, crop)
        output_rows.append(
            {
                "ordinal": int(row["ordinal"]),
                "condition": row["condition_cell"],
                "frame": row["frame"],
                "speed_kmh": row["speed_kmh"],
                "distance_m": row["distance_m"],
                "iso": row["iso"],
                "source_sha256": source_hash,
                "source_width_px": bgr.shape[1],
                "source_height_px": bgr.shape[0],
                "crop_x0": x0,
                "crop_y0": y0,
                "crop_width_px": CAM2_WIDTH,
                "crop_height_px": CAM2_HEIGHT,
                "public_path": relative.as_posix(),
                "public_bytes": size,
                "public_sha256": public_hash,
            }
        )
        if index % 50 == 0 or index == len(rows):
            print(f"cam2 {index}/{len(rows)}", flush=True)
    write_csv(target / "CROP_MANIFEST.csv", output_rows, output_rows[0].keys())
    return target


def build_cam1(revision_root: Path, staging_root: Path) -> Path:
    name = "revision-cam1-optical-crops-v2"
    target = prepare_release(staging_root, name)
    rows = read_csv(revision_root / CAM1_COHORT)
    if len(rows) != 1737:
        raise BuildError(f"Expected 1,737 cam1 ROIs, found {len(rows)}")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["frame_id"]].append(row)
    if len(grouped) != 479:
        raise BuildError(f"Expected 479 cam1 frames, found {len(grouped)}")

    crop_rows: list[dict[str, object]] = []
    roi_rows: list[dict[str, object]] = []
    for index, frame_id in enumerate(sorted(grouped), start=1):
        frame = sorted(grouped[frame_id], key=lambda item: int(item["roi_id"]))
        source = Path(frame[0]["image_path"])
        raw, gray = decode_bytes(source, cv2.IMREAD_GRAYSCALE)
        source_hash = sha256_bytes(raw)
        require_hash(frame_id, source_hash, frame[0]["image_sha256"])
        x0 = max(0, min(int(item["x1"]) for item in frame) - CAM1_MARGIN)
        y0 = max(0, min(int(item["y1"]) for item in frame) - CAM1_MARGIN)
        x1 = min(gray.shape[1], max(int(item["x2"]) for item in frame) + CAM1_MARGIN)
        y1 = min(gray.shape[0], max(int(item["y2"]) for item in frame) + CAM1_MARGIN)
        crop = gray[y0:y1, x0:x1]
        pad_bottom = max(0, CAM1_MIN_SIZE - crop.shape[0])
        pad_right = max(0, CAM1_MIN_SIZE - crop.shape[1])
        if pad_bottom or pad_right:
            crop = cv2.copyMakeBorder(
                crop,
                0,
                pad_bottom,
                0,
                pad_right,
                borderType=cv2.BORDER_REFLECT_101,
            )
        relative = Path("crops") / Path(frame_id)
        size, public_hash = encode_png(target / relative, crop)
        crop_rows.append(
            {
                "frame_id": frame_id,
                "condition": frame[0]["condition"],
                "frame": frame[0]["frame"],
                "speed_kmh": frame[0]["speed_kmh"],
                "distance_m": frame[0]["distance_m"],
                "iso": frame[0]["iso"],
                "source_sha256": source_hash,
                "source_width_px": gray.shape[1],
                "source_height_px": gray.shape[0],
                "crop_x0": x0,
                "crop_y0": y0,
                "crop_x1_exclusive": x1,
                "crop_y1_exclusive": y1,
                "pad_right_px": pad_right,
                "pad_bottom_px": pad_bottom,
                "public_width_px": crop.shape[1],
                "public_height_px": crop.shape[0],
                "public_path": relative.as_posix(),
                "public_bytes": size,
                "public_sha256": public_hash,
            }
        )
        for item in frame:
            roi_rows.append(
                {
                    "frame_id": frame_id,
                    "roi_id": int(item["roi_id"]),
                    "orientation": item["orientation"],
                    "x1": int(item["x1"]) - x0,
                    "y1": int(item["y1"]) - y0,
                    "x2_exclusive": int(item["x2"]) - x0,
                    "y2_exclusive": int(item["y2"]) - y0,
                    "imatest_roi_mtf50": item["imatest_roi_mtf50"],
                    "public_path": relative.as_posix(),
                }
            )
        if index % 50 == 0 or index == len(grouped):
            print(f"cam1 {index}/{len(grouped)}", flush=True)
    write_csv(target / "CROP_MANIFEST.csv", crop_rows, crop_rows[0].keys())
    write_csv(target / "ROI_MANIFEST.csv", roi_rows, roi_rows[0].keys())
    return target


def audit_png_metadata(path: Path) -> tuple[str, str]:
    with Image.open(path) as image:
        if image.getexif():
            raise BuildError(f"Unexpected EXIF metadata in controlled width input: {path.name}")
        keys = sorted(str(key) for key in image.info)
        suspicious = {"author", "comment", "copyright", "description", "software", "title"}
        if suspicious.intersection(key.lower() for key in keys):
            raise BuildError(f"Potential identifying PNG text metadata in {path.name}: {keys}")
        return image.mode, ";".join(keys)


def build_width(revision_root: Path, width_source_root: Path, staging_root: Path) -> Path:
    name = "revision-width-inputs-v2"
    target = prepare_release(staging_root, name)
    rows = read_csv(revision_root / WIDTH_LEDGER)
    if len(rows) != 50:
        raise BuildError(f"Expected 50 printed-width inputs, found {len(rows)}")
    output_rows: list[dict[str, object]] = []
    for row in rows:
        source = width_source_root / row["relative_path"]
        observed = sha256_file(source)
        require_hash(row["source_id"], observed, row["sha256"])
        mode, metadata_keys = audit_png_metadata(source)
        relative = Path("inputs") / row["filename"]
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        output_rows.append(
            {
                **{key: value for key, value in row.items() if key != "roi_status"},
                "pixel_mode": mode,
                "png_metadata_keys": metadata_keys,
                "public_path": relative.as_posix(),
                "public_sha256": observed,
            }
        )
        print(f"width {row['source_id']}/W050", flush=True)
    write_csv(target / "INPUT_MANIFEST.csv", output_rows, output_rows[0].keys())
    return target


def build_models(revision_root: Path, staging_root: Path) -> Path:
    name = "revision-models-essential-v2"
    target = prepare_release(staging_root, name)
    output_rows: list[dict[str, object]] = []
    for public_path, source_relative, expected_hash in CHECKPOINTS:
        source = revision_root / source_relative
        observed = sha256_file(source)
        require_hash(source_relative, observed, expected_hash)
        destination = target / public_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        output_rows.append(
            {
                "public_path": public_path,
                "size_bytes": destination.stat().st_size,
                "sha256": observed,
                "role": "detector" if public_path.startswith("detector/") else "restoration",
            }
        )
    write_csv(target / "MODEL_MANIFEST.csv", output_rows, output_rows[0].keys())
    return target


def copy_repository(repository: Path, staging_root: Path) -> Path:
    name = "revision-analysis-v2"
    target = prepare_release(staging_root, name)
    excluded = {".git", "__pycache__", ".pytest_cache"}
    for source in sorted(repository.rglob("*")):
        relative = source.relative_to(repository)
        if any(part in excluded for part in relative.parts) or not source.is_file():
            continue
        # Keep the repository at the release root. Besides being simpler for users,
        # this avoids the legacy Windows MAX_PATH boundary on the longest audit files.
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return target


def reusable_release(staging_root: Path, name: str, manifest: str, expected_rows: int) -> Path | None:
    """Return a completed image/model staging directory after a failed later step."""
    target = staging_root / name
    manifest_path = target / manifest
    if not manifest_path.is_file():
        return None
    if len(read_csv(manifest_path)) != expected_rows:
        return None
    return target


def build_samples(cam1: Path, cam2: Path, staging_root: Path) -> Path:
    name = "revision-samples-v2"
    target = prepare_release(staging_root, name)
    selected = (
        [("cam1", path) for path in sorted((cam1 / "crops").rglob("*.png"))[:5]]
        + [("cam2", path) for path in sorted((cam2 / "crops").rglob("*.png"))[:10]]
    )
    rows: list[dict[str, object]] = []
    for family, source in selected:
        relative = Path(family) / source.relative_to(cam1 if family == "cam1" else cam2)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        rows.append(
            {
                "family": family,
                "public_path": relative.as_posix(),
                "size_bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        )
    if len(rows) != 15:
        raise BuildError(f"Expected 15 quick-look samples, found {len(rows)}")
    write_csv(target / "SAMPLE_MANIFEST.csv", rows, rows[0].keys())
    return target


def write_integrity_manifest(release: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(release.rglob("*")):
        if not path.is_file() or path.name in {"MANIFEST.csv", "SHA256SUMS"}:
            continue
        records.append(
            {
                "path": path.relative_to(release).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    write_csv(release / "MANIFEST.csv", records, ("path", "size_bytes", "sha256"))
    write_text(
        release / "SHA256SUMS",
        "".join(f"{row['sha256']}  {row['path']}\n" for row in records),
    )
    return records


def partition_files(release: Path) -> list[list[Path]]:
    payload = [
        path
        for path in sorted(release.rglob("*"))
        if path.is_file() and path.name not in {"README.md", "MANIFEST.csv", "SHA256SUMS"}
    ]
    parts: list[list[Path]] = []
    current: list[Path] = []
    current_size = 0
    for path in payload:
        size = path.stat().st_size
        if current and current_size + size > ARCHIVE_TARGET_BYTES:
            parts.append(current)
            current = []
            current_size = 0
        current.append(path)
        current_size += size
    if current or not parts:
        parts.append(current)
    return parts


def zip_release(release: Path, archive_root: Path) -> list[Path]:
    archive_root.mkdir(parents=True, exist_ok=True)
    parts = partition_files(release)
    archives: list[Path] = []
    common = [release / "README.md", release / "MANIFEST.csv", release / "SHA256SUMS"]
    for index, files in enumerate(parts, start=1):
        suffix = "" if len(parts) == 1 else f"-part{index:02d}-of-{len(parts):02d}"
        archive = archive_root / f"{release.name}{suffix}.zip"
        with zipfile.ZipFile(archive, "w", allowZip64=True) as zipped:
            for path in common + files:
                relative = Path(release.name) / path.relative_to(release)
                compression = (
                    zipfile.ZIP_STORED
                    if path.suffix.lower() in {".png", ".pt", ".pth", ".zip"}
                    else zipfile.ZIP_DEFLATED
                )
                # Use fixed ZIP metadata so identical release content produces an
                # identical archive SHA-256 across rebuilds and machines.
                info = zipfile.ZipInfo(relative.as_posix(), date_time=(2026, 10, 26, 0, 0, 0))
                info.compress_type = compression
                info.create_system = 3
                info.external_attr = (0o100644 & 0xFFFF) << 16
                if compression == zipfile.ZIP_DEFLATED:
                    info._compresslevel = 6
                with path.open("rb") as source, zipped.open(info, "w", force_zip64=True) as destination:
                    shutil.copyfileobj(source, destination, length=4 * 1024 * 1024)
        if archive.stat().st_size >= GITHUB_ASSET_LIMIT_BYTES:
            raise BuildError(f"GitHub release asset is too large: {archive}")
        archives.append(archive)
        print(
            f"archive {archive.name}: {archive.stat().st_size / 1024**2:.1f} MiB",
            flush=True,
        )
    return archives


def verify_no_field_pixels(releases: Iterable[Path]) -> None:
    forbidden = ("four_tunnel", "tunnel_a", "tunnel_b", "tunnel_c", "tunnel_d", "expert_mask")
    for release in releases:
        for path in release.rglob("*"):
            relative = path.relative_to(release).as_posix().lower()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
                if any(token in relative for token in forbidden):
                    raise BuildError(f"Potential field pixel found in release staging: {relative}")


def write_archive_manifest(output_root: Path, archives: list[Path]) -> None:
    rows = [
        {
            "asset": archive.name,
            "size_bytes": archive.stat().st_size,
            "sha256": sha256_file(archive),
        }
        for archive in sorted(archives)
    ]
    write_csv(output_root / "RELEASE_ASSETS.csv", rows, rows[0].keys())
    write_text(
        output_root / "SHA256SUMS",
        "".join(f"{row['sha256']}  archives/{row['asset']}\n" for row in rows),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision-root", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--width-source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed, manifest-validated staging directories",
    )
    args = parser.parse_args()

    revision_root = args.revision_root.resolve()
    repository = args.repository.resolve()
    width_source_root = args.width_source_root.resolve()
    output_root = args.output_root.resolve()
    if output_root == repository or repository in output_root.parents:
        raise BuildError("Release assets must be built outside the Git repository")
    staging_root = output_root / "staging"
    archive_root = output_root / "archives"
    staging_root.mkdir(parents=True, exist_ok=True)
    archive_root.mkdir(parents=True, exist_ok=True)
    for old in archive_root.glob("revision-*.zip"):
        old.unlink()

    cam2 = (
        reusable_release(staging_root, "revision-cam2-crops-v2", "CROP_MANIFEST.csv", 737)
        if args.resume
        else None
    ) or build_cam2(revision_root, staging_root)
    cam1 = (
        reusable_release(
            staging_root,
            "revision-cam1-optical-crops-v2",
            "CROP_MANIFEST.csv",
            479,
        )
        if args.resume
        else None
    ) or build_cam1(revision_root, staging_root)
    width = (
        reusable_release(staging_root, "revision-width-inputs-v2", "INPUT_MANIFEST.csv", 50)
        if args.resume
        else None
    ) or build_width(revision_root, width_source_root, staging_root)
    models = (
        reusable_release(staging_root, "revision-models-essential-v2", "MODEL_MANIFEST.csv", 7)
        if args.resume
        else None
    ) or build_models(revision_root, staging_root)
    analysis = copy_repository(repository, staging_root)
    samples = build_samples(cam1, cam2, staging_root)
    releases = [analysis, cam2, cam1, width, models, samples]
    verify_no_field_pixels(releases)

    archives: list[Path] = []
    for release in releases:
        records = write_integrity_manifest(release)
        print(f"manifest {release.name}: {len(records)} files", flush=True)
        archives.extend(zip_release(release, archive_root))
    write_archive_manifest(output_root, archives)
    summary = {
        "release_count": len(releases),
        "archive_count": len(archives),
        "archives_bytes": sum(path.stat().st_size for path in archives),
        "four_tunnel_pixels": 0,
    }
    write_text(output_root / "BUILD_SUMMARY.json", json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
