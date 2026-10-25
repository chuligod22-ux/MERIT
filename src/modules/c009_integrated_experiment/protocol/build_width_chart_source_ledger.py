"""Freeze the C009 printed-width chart source inventory.

This script records source-file identity and the image-visible 4/4/2 label
layout.  It intentionally does not estimate crack width or generate ROIs.
Those fields stay blocked until a separate per-image ROI ledger is reviewed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


HERE = Path(__file__).resolve().parent
SOURCE_ROOT = Path(
    r"<LOCAL_PATH_REDACTED>"
    r"\01_tunnelscanning\03_src\data\raw\crack\cam1"
)
LEDGER_PATH = HERE / "width_chart_source_ledger.csv"
FREEZE_PATH = HERE / "width_chart_source_freeze.json"
NAME_PATTERN = re.compile(
    r"^cam1_v(?P<speed>\d+)_iso(?P<iso>\d+)_d(?P<distance>\d+)"
    r"(?P<x_flag>_x)?\.png$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_ledger(rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "source_id",
        "filename",
        "relative_path",
        "sha256",
        "bytes",
        "width_px",
        "height_px",
        "speed_kmh",
        "iso",
        "distance_m",
        "legacy_x_suffix",
        "width_layout",
        "roi_status",
    ]
    with LEDGER_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    all_png = sorted(SOURCE_ROOT.glob("*.png"), key=lambda path: path.name.lower())
    sidecars = [path for path in all_png if path.name.startswith("._")]
    files = [path for path in all_png if not path.name.startswith("._")]
    if len(files) != 50:
        raise RuntimeError(f"Expected 50 PNG sources, found {len(files)}")

    rows: list[dict[str, object]] = []
    for index, path in enumerate(files, start=1):
        match = NAME_PATTERN.fullmatch(path.name)
        if match is None:
            raise RuntimeError(f"Unexpected source filename: {path.name}")
        with Image.open(path) as image:
            width_px, height_px = image.size
            image.verify()
        rows.append(
            {
                "source_id": f"W{index:03d}",
                "filename": path.name,
                "relative_path": path.relative_to(SOURCE_ROOT).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "width_px": width_px,
                "height_px": height_px,
                "speed_kmh": int(match.group("speed")),
                "iso": int(match.group("iso")),
                "distance_m": int(match.group("distance")) / 10.0,
                "legacy_x_suffix": bool(match.group("x_flag")),
                "width_layout": "top:0.1,0.2,0.3,0.4;middle:0.5,0.6,0.7,0.8;bottom:0.9,1.0",
                "roi_status": "PENDING_PER_IMAGE_REVIEW",
            }
        )

    write_ledger(rows)
    representative = next(
        row for row in rows if row["filename"] == "cam1_v60_iso100_d25.png"
    )
    freeze = {
        "schema_version": "MERIT-C009-WIDTH-SOURCE-FREEZE-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(SOURCE_ROOT),
        "source_count": len(rows),
        "excluded_sidecars": {
            "rule": "filename starts with ._ (AppleDouble metadata, not image data)",
            "count": len(sidecars),
            "files": [path.name for path in sidecars],
        },
        "dimensions": sorted(
            {f"{row['width_px']}x{row['height_px']}" for row in rows}
        ),
        "factor_grid": {
            "speed_kmh": sorted({row["speed_kmh"] for row in rows}),
            "iso": sorted({row["iso"] for row in rows}),
            "distance_m": sorted({row["distance_m"] for row in rows}),
        },
        "layout": {
            "status": "MANUALLY_CONFIRMED_ON_REPRESENTATIVE_FRAME",
            "representative_filename": representative["filename"],
            "representative_sha256": representative["sha256"],
            "rows": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0]],
            "label_basis": "image-visible handwritten nominal labels",
            "claim_limit": "nominal apparent printed width; not traceable physical metrology",
        },
        "legacy_artifact_disposition": {
            "path": str(
                SOURCE_ROOT.parent.parent.parent / "cam1_crack_detectability.csv"
            ),
            "status": "DO_NOT_USE_FOR_WIDTH_ENDPOINT",
            "reason": "legacy script encoded an incorrect 3/3/4 grid instead of the observed 4/4/2 layout",
        },
        "ledger": {
            "path": str(LEDGER_PATH),
            "sha256": sha256_file(LEDGER_PATH),
            "rows": len(rows),
        },
        "generator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "gate_status": "PARTIAL",
        "remaining_blocker": "Create and independently review a per-image 10-ROI ledger before any quantitative width measurement.",
    }
    FREEZE_PATH.write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(freeze, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
