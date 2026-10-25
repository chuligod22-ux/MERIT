from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE_ROOT = Path(
    r"<LOCAL_PATH_REDACTED>"
)
CANDIDATE_CSV = HERE / "width_chart_roi_candidates_v1.csv"
CANDIDATE_JSON = HERE / "width_chart_roi_candidates_v1.json"
FINAL_CSV = HERE / "width_chart_roi_final_v1.csv"
FINAL_JSON = HERE / "width_chart_roi_final_v1.json"
REVIEW_DIR = HERE / "width_chart_roi_review_v1"


# Coordinates were read from the raw source only. Values are in the original
# 8192 x 5460 pixel coordinate system. No restoration or measurement output was
# used. The ISO1600/d35 patches were recovered from the visible patch boundaries
# in a 1/4-scale raw rendering and mapped back exactly by x4.
MANUAL_POLYGONS: dict[tuple[str, float], list[list[float]]] = {
    ("cam1_v60_iso1600_d35.png", 0.1): [[2904, 28], [3544, 28], [3544, 684], [2904, 684]],
    ("cam1_v60_iso1600_d35.png", 0.2): [[3788, 28], [4428, 28], [4428, 684], [3788, 684]],
    ("cam1_v60_iso1600_d35.png", 0.3): [[4624, 28], [5260, 28], [5260, 684], [4624, 684]],
    ("cam1_v60_iso1600_d35.png", 0.4): [[5484, 40], [6128, 40], [6128, 700], [5484, 700]],
    ("cam1_v60_iso1600_d35.png", 0.5): [[2920, 1020], [3560, 1020], [3560, 1668], [2920, 1668]],
    ("cam1_v60_iso1600_d35.png", 0.6): [[3788, 1004], [4428, 1004], [4428, 1652], [3788, 1652]],
    ("cam1_v60_iso1600_d35.png", 0.7): [[4616, 1004], [5256, 1004], [5256, 1648], [4616, 1648]],
    ("cam1_v60_iso1600_d35.png", 0.8): [[5508, 952], [6148, 952], [6148, 1608], [5508, 1608]],
    ("cam1_v60_iso1600_d35.png", 0.9): [[3768, 1892], [4420, 1892], [4420, 2544], [3768, 2544]],
    ("cam1_v60_iso1600_d35.png", 1.0): [[4604, 1832], [5240, 1832], [5240, 2472], [4604, 2472]],
    # The upper edge of this patch is outside the captured board, but the full
    # printed crack line is visible. Freeze the visible in-frame patch region.
    ("cam1_v60_iso400_d45.png", 0.3): [[4097, 10], [4754, 10], [4754, 403], [4097, 403]],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot decode image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


def polygon_from_row(row: dict[str, str]) -> list[list[float]]:
    return [
        [float(row[f"p{i}_x"]), float(row[f"p{i}_y"])]
        for i in range(4)
    ]


def polygon_area(points: list[list[float]]) -> float:
    contour = np.asarray(points, dtype=np.float32)
    return float(abs(cv2.contourArea(contour)))


def polygon_columns(points: list[list[float]]) -> dict[str, str]:
    columns: dict[str, str] = {}
    for index, (x, y) in enumerate(points):
        columns[f"p{index}_x"] = f"{x:.3f}"
        columns[f"p{index}_y"] = f"{y:.3f}"
    return columns


def render_correction(
    filename: str,
    candidate_rows: list[dict[str, str]],
    final_rows: list[dict[str, str]],
) -> Path:
    source = read_image(SOURCE_ROOT / filename)
    scale = 0.2
    canvas = cv2.resize(source, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    header = np.zeros((110, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        header,
        f"{filename} | red=candidate, green=final raw-only correction",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.15,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        "No restoration/detector/width result used",
        (20, 87),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (180, 255, 180),
        2,
        cv2.LINE_AA,
    )

    for candidate, final in zip(candidate_rows, final_rows, strict=True):
        width = float(candidate["nominal_width_mm"])
        candidate_poly = np.asarray(polygon_from_row(candidate), dtype=np.float32)
        final_poly = np.asarray(polygon_from_row(final), dtype=np.float32)
        candidate_poly = np.rint(candidate_poly * scale).astype(np.int32)
        final_poly = np.rint(final_poly * scale).astype(np.int32)
        if final["review_disposition"] == "CORRECTED":
            cv2.polylines(canvas, [candidate_poly], True, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.polylines(canvas, [final_poly], True, (0, 255, 0), 3, cv2.LINE_AA)
            label_at = tuple(final_poly[0])
            cv2.putText(
                canvas,
                f"{width:.1f}",
                (int(label_at[0]), max(25, int(label_at[1]) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    output = REVIEW_DIR / f"final_correction_{Path(filename).stem}.jpg"
    write_image(output, np.vstack([header, canvas]))
    return output


def main() -> None:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    candidate_meta = json.loads(CANDIDATE_JSON.read_text(encoding="utf-8"))
    with CANDIDATE_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        candidate_rows = list(csv.DictReader(handle))

    if len(candidate_rows) != 500:
        raise RuntimeError(f"Expected 500 candidate rows, found {len(candidate_rows)}")

    flagged_frames = {
        entry["filename"] for entry in candidate_meta["strict_qc_flagged_pages"]
    }
    image_sizes: dict[str, tuple[int, int]] = {}
    final_rows: list[dict[str, str]] = []

    for candidate in candidate_rows:
        filename = candidate["filename"]
        width = round(float(candidate["nominal_width_mm"]), 1)
        key = (filename, width)
        candidate_polygon = polygon_from_row(candidate)
        final_polygon = MANUAL_POLYGONS.get(key, candidate_polygon)

        if filename not in image_sizes:
            image = read_image(SOURCE_ROOT / filename)
            image_sizes[filename] = (image.shape[1], image.shape[0])
        image_width, image_height = image_sizes[filename]
        points = np.asarray(final_polygon, dtype=np.float32)
        in_bounds = bool(
            np.all(points[:, 0] >= 0)
            and np.all(points[:, 0] < image_width)
            and np.all(points[:, 1] >= 0)
            and np.all(points[:, 1] < image_height)
        )
        convex = bool(cv2.isContourConvex(points))
        area = polygon_area(final_polygon)
        if not in_bounds or not convex or area < 1000:
            raise RuntimeError(
                f"Invalid final polygon: {filename}, {width}, "
                f"in_bounds={in_bounds}, convex={convex}, area={area}"
            )

        corrected = key in MANUAL_POLYGONS
        if filename == "cam1_v60_iso400_d45.png" and width == 0.3:
            reason = "RAW_VISIBLE_REGION_CORRECTION_FULL_CRACK_LINE_VISIBLE"
        elif corrected:
            reason = "RAW_PATCH_BOUNDARY_MANUAL_CORRECTION"
        elif filename in flagged_frames:
            reason = "RAW_FLAGGED_FRAME_VISUAL_ACCEPT"
        else:
            reason = "CONTACT_SHEET_VISUAL_ACCEPT"

        final = dict(candidate)
        final.update(polygon_columns(final_polygon))
        final["projected_area_px2"] = f"{area:.3f}"
        final["candidate_human_review"] = candidate["human_review"]
        final["candidate_final_status"] = candidate["final_status"]
        final["review_disposition"] = "CORRECTED" if corrected else "ACCEPT_AS_IS"
        final["review_reason"] = reason
        final["reviewer"] = "CODEX_INDEPENDENT_RAW_ONLY_VISUAL_AUDIT"
        final["measurement_status"] = "NOT_RUN"
        final["final_status"] = "ROI_FROZEN_MEASUREMENT_PENDING"
        del final["human_review"]
        final_rows.append(final)

    dispositions = Counter(row["review_disposition"] for row in final_rows)
    if dispositions != Counter({"ACCEPT_AS_IS": 489, "CORRECTED": 11}):
        raise RuntimeError(f"Unexpected dispositions: {dispositions}")

    fieldnames = list(final_rows[0].keys())
    with FINAL_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(final_rows)

    correction_frames = sorted({filename for filename, _ in MANUAL_POLYGONS})
    correction_pages = []
    for filename in correction_frames:
        source_rows = [row for row in candidate_rows if row["filename"] == filename]
        frozen_rows = [row for row in final_rows if row["filename"] == filename]
        page = render_correction(filename, source_rows, frozen_rows)
        correction_pages.append({"path": str(page), "sha256": sha256(page)})

    review_pages = []
    for page in sorted(REVIEW_DIR.glob("width_roi_review_*.jpg")):
        review_pages.append({"path": str(page), "sha256": sha256(page)})
    flagged_pages = []
    for page in sorted(REVIEW_DIR.glob("flagged_*.jpg")):
        flagged_pages.append({"path": str(page), "sha256": sha256(page)})

    final_meta = {
        "schema_version": "MERIT-C009-WIDTH-ROI-FINAL-v1",
        "created_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "status": "ROI_LEDGER_FROZEN_MEASUREMENT_INPUT_READY",
        "review_scope": "50 raw frames x 10 nominal widths = 500 ROIs",
        "review_rule": (
            "Raw source pixels and pre-measurement overlays only; no restoration, "
            "detector, or width-estimation output was inspected."
        ),
        "reviewer": "CODEX_INDEPENDENT_RAW_ONLY_VISUAL_AUDIT",
        "candidate_input": {
            "csv_path": str(CANDIDATE_CSV),
            "csv_sha256": sha256(CANDIDATE_CSV),
            "json_path": str(CANDIDATE_JSON),
            "json_sha256": sha256(CANDIDATE_JSON),
        },
        "final_ledger": {
            "path": str(FINAL_CSV),
            "sha256": sha256(FINAL_CSV),
            "rows": len(final_rows),
            "unique_frames": len({row["filename"] for row in final_rows}),
            "unique_widths": sorted(
                {float(row["nominal_width_mm"]) for row in final_rows}
            ),
        },
        "disposition_counts": dict(sorted(dispositions.items())),
        "non_estimable_count_at_roi_freeze": 0,
        "measurement_status_counts": dict(
            Counter(row["measurement_status"] for row in final_rows)
        ),
        "strict_qc_flagged_frame_count_reviewed": len(flagged_frames),
        "manual_corrections": [
            {
                "filename": filename,
                "nominal_width_mm": width,
                "polygon": polygon,
            }
            for (filename, width), polygon in sorted(MANUAL_POLYGONS.items())
        ],
        "review_pages": review_pages,
        "strict_qc_flagged_pages": flagged_pages,
        "correction_pages": correction_pages,
        "authorization": {
            "raw_width_measurement": True,
            "restored_width_comparison": "REQUIRES_G11_OFFICIAL_MODEL_OUTPUTS",
            "full_training": False,
        },
        "generator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
    }
    FINAL_JSON.write_text(
        json.dumps(final_meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(final_meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
