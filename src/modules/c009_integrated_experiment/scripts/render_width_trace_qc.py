"""Render raw-only contact sheets for visual audit of frozen centreline traces."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from measure_width_chart_raw import (
    DEFAULT_LEDGER,
    DEFAULT_SOURCE_ROOT,
    MERIT_ROOT,
    polygon,
    read_image,
    rectify,
)


DEFAULT_ARTIFACT_DIR = (
    MERIT_ROOT
    / "40_수정본"
    / "c009_integrated_experiment"
    / "artifacts"
    / "width_raw_measurement_v1"
)
STATUS_COLOURS = {
    "ESTIMABLE": (50, 205, 50),
    "PREDECLARED_SUBPIXEL_NON_ESTIMABLE": (255, 170, 30),
    "QC_NON_ESTIMABLE": (45, 45, 230),
    "UNCERTAINTY_NON_ESTIMABLE": (0, 165, 255),
}


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image, [cv2.IMWRITE_JPEG_QUALITY, 91])
    if not ok:
        raise RuntimeError(f"Unable to encode image: {path}")
    encoded.tofile(path)


def render_tile(
    gray: np.ndarray,
    trace: dict[str, object],
    result: dict[str, str],
    tile_size: int = 260,
) -> np.ndarray:
    canvas = cv2.cvtColor(np.clip(gray, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    xs = np.asarray(trace["trace_x_rectified"], dtype=np.float32)
    ys = np.asarray(trace["trace_y_rectified"], dtype=np.float32)
    points = np.column_stack([xs, ys]).round().astype(np.int32)
    colour = STATUS_COLOURS[result["status"]]
    cv2.polylines(canvas, [points], False, colour, 2, cv2.LINE_AA)
    canvas = cv2.resize(canvas, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
    bar = np.zeros((48, tile_size, 3), dtype=np.uint8)
    short_status = {
        "ESTIMABLE": "EST",
        "PREDECLARED_SUBPIXEL_NON_ESTIMABLE": "SUBPX",
        "QC_NON_ESTIMABLE": "QC-NE",
        "UNCERTAINTY_NON_ESTIMABLE": "UNC-NE",
    }[result["status"]]
    width_text = result["apparent_width_mm_design_gsd"]
    if width_text:
        width_text = f" -> {float(width_text):.3f} mm"
    text = f"{result['source_id']} {result['nominal_width_mm']} mm {short_status}{width_text}"
    cv2.putText(
        bar,
        text,
        (6, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        colour,
        1,
        cv2.LINE_AA,
    )
    return np.vstack([bar, canvas])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--version", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()

    result_path = args.artifact_dir / f"width_chart_raw_measurements_v{args.version}.csv"
    trace_path = args.artifact_dir / f"width_chart_raw_traces_v{args.version}.jsonl"
    with args.ledger.open("r", encoding="utf-8-sig", newline="") as handle:
        ledger = list(csv.DictReader(handle))
    with result_path.open("r", encoding="utf-8-sig", newline="") as handle:
        results = list(csv.DictReader(handle))
    with trace_path.open("r", encoding="utf-8") as handle:
        traces = [json.loads(line) for line in handle if line.strip()]
    if not (len(ledger) == len(results) == len(traces) == 500):
        raise RuntimeError("Ledger, results, and traces must each contain 500 rows")

    tiles: dict[float, list[np.ndarray]] = defaultdict(list)
    image_cache: dict[str, np.ndarray] = {}
    for ledger_row, result, trace in zip(ledger, results, traces, strict=True):
        key = (ledger_row["source_id"], float(ledger_row["nominal_width_mm"]))
        if key != (result["source_id"], float(result["nominal_width_mm"])):
            raise RuntimeError(f"Result alignment failure at {key}")
        if key != (trace["source_id"], float(trace["nominal_width_mm"])):
            raise RuntimeError(f"Trace alignment failure at {key}")
        filename = ledger_row["filename"]
        if filename not in image_cache:
            image_cache.clear()
            image_cache[filename] = read_image(args.source_root / filename)
        patch = rectify(image_cache[filename], polygon(ledger_row))
        tiles[key[1]].append(render_tile(patch.gray, trace, result))

    output_dir = args.artifact_dir / "trace_qc"
    output_dir.mkdir(parents=True, exist_ok=True)
    for width, width_tiles in sorted(tiles.items()):
        if len(width_tiles) != 50:
            raise RuntimeError(f"Expected 50 tiles for {width:.1f} mm")
        rows = [np.hstack(width_tiles[index : index + 5]) for index in range(0, 50, 5)]
        sheet = np.vstack(rows)
        output_path = output_dir / f"trace_qc_{width:.1f}mm.jpg"
        write_image(output_path, sheet)
        print(output_path)


if __name__ == "__main__":
    main()
