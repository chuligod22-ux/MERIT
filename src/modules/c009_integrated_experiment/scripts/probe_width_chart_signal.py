"""Render raw-only diagnostic views for the frozen printed-width ROIs.

This script does not estimate crack width.  It rectifies the already frozen
quadrilaterals, preserves their approximate native sampling, and renders both
the grayscale patch and a local-background-subtracted dark-line signal.  The
output is used to freeze a measurement algorithm without inspecting any
restoration result.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_DIR = REPO_ROOT / "40_수정본" / "c009_integrated_experiment" / "protocol"
DEFAULT_LEDGER = PROTOCOL_DIR / "width_chart_roi_final_v1.csv"
DEFAULT_SOURCE_ROOT = Path(
    r"<LOCAL_PATH_REDACTED>"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "40_수정본"
    / "c009_integrated_experiment"
    / "artifacts"
    / "width_raw_probe_v1"
    / "raw_width_signal_probe.png"
)


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unable to read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"Unable to encode image: {path}")
    encoded.tofile(path)


def polygon(row: dict[str, str]) -> np.ndarray:
    return np.asarray(
        [[float(row[f"p{i}_x"]), float(row[f"p{i}_y"])] for i in range(4)],
        dtype=np.float32,
    )


def rectify(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    side_lengths = [
        np.linalg.norm(points[1] - points[0]),
        np.linalg.norm(points[2] - points[1]),
        np.linalg.norm(points[3] - points[2]),
        np.linalg.norm(points[0] - points[3]),
    ]
    side = max(128, int(round(float(np.mean(side_lengths)))))
    target = np.asarray(
        [[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(points, target)
    return cv2.warpPerspective(
        image,
        transform,
        (side, side),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def local_dark_signal(gray: np.ndarray) -> np.ndarray:
    side = min(gray.shape)
    sigma = max(7.0, side / 22.0)
    background = cv2.GaussianBlur(
        gray.astype(np.float32),
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT101,
    )
    residual = background - gray.astype(np.float32)
    lo, hi = np.percentile(residual, [5.0, 99.5])
    if hi <= lo:
        return np.zeros_like(gray)
    return np.clip((residual - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def labelled_panel(image: np.ndarray, title: str, width: int = 360) -> np.ndarray:
    height = width
    panel = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    if panel.ndim == 2:
        panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
    bar = np.zeros((38, width, 3), dtype=np.uint8)
    cv2.putText(
        bar,
        title,
        (8, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([bar, panel])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--source-ids",
        nargs="+",
        default=["W001", "W005", "W021", "W025", "W046", "W050"],
    )
    args = parser.parse_args()

    with args.ledger.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    selected = [row for row in rows if row["source_id"] in set(args.source_ids)]
    expected = len(args.source_ids) * 10
    if len(selected) != expected:
        raise RuntimeError(f"Expected {expected} rows, found {len(selected)}")

    image_cache: dict[str, np.ndarray] = {}
    rendered_rows: list[np.ndarray] = []
    for source_id in args.source_ids:
        source_rows = sorted(
            (row for row in selected if row["source_id"] == source_id),
            key=lambda row: float(row["nominal_width_mm"]),
        )
        filename = source_rows[0]["filename"]
        image_cache.setdefault(filename, read_image(args.source_root / filename))
        panels: list[np.ndarray] = []
        for row in source_rows:
            patch = rectify(image_cache[filename], polygon(row))
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            signal = local_dark_signal(gray)
            pair = np.hstack(
                [
                    labelled_panel(gray, f"{source_id} {row['nominal_width_mm']} mm RAW"),
                    labelled_panel(signal, "local dark residual"),
                ]
            )
            panels.append(pair)
        rendered_rows.append(np.hstack(panels))

    output = np.vstack(rendered_rows)
    write_image(args.output, output)
    overview = cv2.resize(output, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    overview_path = args.output.with_name(f"{args.output.stem}_overview.jpg")
    write_image(overview_path, overview)
    for source_id, rendered in zip(args.source_ids, rendered_rows, strict=True):
        source_path = args.output.with_name(f"{source_id}_raw_width_signal.jpg")
        write_image(source_path, rendered)
    print(args.output)
    print(overview_path)


if __name__ == "__main__":
    main()
