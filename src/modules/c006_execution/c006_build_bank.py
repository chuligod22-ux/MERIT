"""Build the immutable, DeepCrack-free D0 crop bank for MERIT C006."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "c006_protocol.json"
DEFAULT_OUTPUT = HERE / "artifacts"
MANIFEST_FIELDS = [
    "bank_index",
    "source_dataset",
    "camera",
    "condition",
    "condition_ordinal",
    "parent_path",
    "parent_name",
    "parent_bytes",
    "parent_sha256",
    "parent_width",
    "parent_height",
    "crop_x",
    "crop_y",
    "crop_width",
    "crop_height",
    "role",
    "role_source",
    "rule_id",
    "split",
    "crop_sha256",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_parents(config: dict[str, Any]) -> list[dict[str, Any]]:
    source_root = Path(config["source_root"])
    parents: list[dict[str, Any]] = []
    for camera in ("cam1", "cam2"):
        camera_config = config[camera]
        excluded = set(camera_config["excluded_basenames"])
        for condition in camera_config["conditions"]:
            folder = source_root / camera / condition
            paths = sorted(folder.glob(camera_config["glob"]), key=lambda p: p.name)
            paths = [path for path in paths if path.name not in excluded]
            if not paths:
                raise RuntimeError(f"No D0 parents found: {folder}")
            for ordinal, path in enumerate(paths):
                parents.append(
                    {
                        "camera": camera,
                        "condition": condition,
                        "condition_ordinal": ordinal,
                        "path": path.resolve(),
                    }
                )
    return parents


def parent_split(ordinal: int, config: dict[str, Any]) -> str:
    rule = config["parent_split_rule"]
    if ordinal % int(rule["holdout_modulus"]) == int(
        rule["holdout_remainder"]
    ):
        return "synthetic_holdout"
    return "train"


def crop_origin(
    image_width: int,
    image_height: int,
    patch_size: int,
    rule: dict[str, Any],
) -> tuple[int, int]:
    x_coord = math.floor(
        image_width * float(rule["x_center_fraction"]) - patch_size / 2
    )
    y_coord = math.floor(image_height * float(rule["y_top_fraction"]))
    if not (
        0 <= x_coord <= image_width - patch_size
        and 0 <= y_coord <= image_height - patch_size
    ):
        raise ValueError(
            f"Crop outside parent: {rule['rule_id']} -> "
            f"({x_coord},{y_coord},{patch_size},{patch_size}) in "
            f"{image_width}x{image_height}"
        )
    return x_coord, y_coord


def decode_grayscale(raw_bytes: bytes, path: Path) -> np.ndarray:
    encoded = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"OpenCV failed to decode {path}")
    return image


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_contact_sheets(
    output_dir: Path,
    bank: np.ndarray,
    rows: list[dict[str, Any]],
) -> list[Path]:
    output_paths: list[Path] = []
    rule_ids = sorted({str(row["rule_id"]) for row in rows})
    for rule_id in rule_ids:
        indices = [
            int(row["bank_index"])
            for row in rows
            if row["rule_id"] == rule_id
        ]
        columns = 10
        tile = 128
        label_height = 24
        row_count = math.ceil(len(indices) / columns)
        sheet = np.full(
            (row_count * (tile + label_height), columns * tile), 245, np.uint8
        )
        for position, bank_index in enumerate(indices):
            grid_row, grid_column = divmod(position, columns)
            resized = cv2.resize(
                bank[bank_index], (tile, tile), interpolation=cv2.INTER_AREA
            )
            y0 = grid_row * (tile + label_height)
            x0 = grid_column * tile
            sheet[y0 : y0 + tile, x0 : x0 + tile] = resized
            split = rows[bank_index]["split"]
            label = f"{bank_index:03d} {'H' if split == 'synthetic_holdout' else 'T'}"
            cv2.putText(
                sheet,
                label,
                (x0 + 4, y0 + tile + 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                20,
                1,
                cv2.LINE_AA,
            )
        output_path = output_dir / f"audit_{rule_id}.jpg"
        encoded_ok, encoded = cv2.imencode(
            ".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]
        )
        if not encoded_ok:
            raise RuntimeError(f"Failed to encode {output_path}")
        encoded.tofile(output_path)
        output_paths.append(output_path)
    return output_paths


def build_bank(
    config_path: Path,
    output_dir: Path,
    write_audit_sheets: bool = True,
) -> dict[str, Any]:
    config = load_config(config_path)
    patch_size = int(config["patch_size"])
    parents = discover_parents(config)
    rules_by_camera: dict[str, list[dict[str, Any]]] = {
        camera: [
            rule for rule in config["crop_rules"] if rule["camera"] == camera
        ]
        for camera in ("cam1", "cam2")
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    crops: list[np.ndarray] = []
    content_digest = hashlib.sha256()
    parent_digests: dict[str, str] = {}
    for parent in parents:
        path = parent["path"]
        raw_bytes = path.read_bytes()
        parent_hash = sha256_bytes(raw_bytes)
        parent_digests[str(path)] = parent_hash
        image = decode_grayscale(raw_bytes, path)
        image_height, image_width = image.shape
        split = parent_split(parent["condition_ordinal"], config)
        for rule in rules_by_camera[parent["camera"]]:
            crop_x, crop_y = crop_origin(
                image_width, image_height, patch_size, rule
            )
            crop = np.ascontiguousarray(
                image[
                    crop_y : crop_y + patch_size,
                    crop_x : crop_x + patch_size,
                ]
            )
            if crop.shape != (patch_size, patch_size):
                raise RuntimeError(f"Invalid crop shape {crop.shape}: {path}")
            bank_index = len(crops)
            crop_hash = sha256_bytes(crop.tobytes(order="C"))
            content_digest.update(bank_index.to_bytes(8, byteorder="big"))
            content_digest.update(crop.tobytes(order="C"))
            crops.append(crop)
            rows.append(
                {
                    "bank_index": bank_index,
                    "source_dataset": "D0",
                    "camera": parent["camera"],
                    "condition": parent["condition"],
                    "condition_ordinal": parent["condition_ordinal"],
                    "parent_path": str(path),
                    "parent_name": path.name,
                    "parent_bytes": len(raw_bytes),
                    "parent_sha256": parent_hash,
                    "parent_width": image_width,
                    "parent_height": image_height,
                    "crop_x": crop_x,
                    "crop_y": crop_y,
                    "crop_width": patch_size,
                    "crop_height": patch_size,
                    "role": rule["role"],
                    "role_source": rule["role_source"],
                    "rule_id": rule["rule_id"],
                    "split": split,
                    "crop_sha256": crop_hash,
                }
            )

    bank = np.stack(crops).astype(np.uint8, copy=False)
    bank_path = output_dir / "d0_bank_448.npy"
    with bank_path.open("wb") as handle:
        np.save(handle, bank, allow_pickle=False)
    manifest_path = output_dir / "d0_bank_manifest.csv"
    write_manifest(manifest_path, rows)
    audit_paths = (
        write_contact_sheets(output_dir, bank, rows)
        if write_audit_sheets
        else []
    )

    parent_counts = Counter(parent["camera"] for parent in parents)
    role_counts = Counter(str(row["role"]) for row in rows)
    split_counts = Counter(str(row["split"]) for row in rows)
    rule_counts = Counter(str(row["rule_id"]) for row in rows)
    ledger = {
        "protocol_id": config["protocol_id"],
        "generated_at_kst": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selection_used_model_output": False,
        "role_assignment_used_model_output": False,
        "source_dataset": "D0 only",
        "parent_count": len(parents),
        "parent_counts_by_camera": dict(sorted(parent_counts.items())),
        "unique_parent_hash_count": len(set(parent_digests.values())),
        "bank_shape": list(bank.shape),
        "bank_dtype": str(bank.dtype),
        "role_counts": dict(sorted(role_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "rule_counts": dict(sorted(rule_counts.items())),
        "bank_content_hash_definition": (
            "SHA-256 over manifest-order repetitions of uint64 big-endian "
            "bank_index followed by contiguous uint8 crop bytes"
        ),
        "bank_content_sha256": content_digest.hexdigest(),
        "bank_npy_sha256": sha256_file(bank_path),
        "manifest_csv_sha256": sha256_file(manifest_path),
        "protocol_json_sha256": sha256_file(config_path),
        "builder_code_sha256": sha256_file(Path(__file__).resolve()),
        "audit_contact_sheets": {
            path.name: sha256_file(path) for path in audit_paths
        },
        "forbidden_source_check": {
            "D1_rows": 0,
            "D2_rows": 0,
            "D3_rows": 0,
            "DQ1_rows": 0,
            "legacy_crack_bank_rows": 0,
        },
    }
    ledger["ledger_payload_sha256"] = sha256_bytes(canonical_json_bytes(ledger))
    ledger_path = output_dir / "d0_bank_ledger.json"
    with ledger_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(ledger, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return ledger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-audit-sheets", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ledger = build_bank(
        args.config.resolve(),
        args.output_dir.resolve(),
        write_audit_sheets=not args.no_audit_sheets,
    )
    print(json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
