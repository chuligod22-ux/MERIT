"""Build reviewable 4/4/2 printed-width ROI candidates for C009 WP8.

The 50 source images are read-only.  A SIFT/RANSAC planar registration maps
ten manually constrained reference-cell detections to every acquisition
condition.  Outputs remain AUTO_CANDIDATE until a human reviews the rendered
contact sheets; they are not quantitative crack-width measurements.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE_ROOT = Path(
    r"<LOCAL_PATH_REDACTED>"
    r"\01_tunnelscanning\03_src\data\raw\crack\cam1"
)
SOURCE_LEDGER = HERE / "width_chart_source_ledger.csv"
ROI_LEDGER = HERE / "width_chart_roi_candidates_v1.csv"
FREEZE_PATH = HERE / "width_chart_roi_candidates_v1.json"
CONTACT_DIR = HERE / "width_chart_roi_review_v1"
REFERENCE_NAME = "cam1_v60_iso100_d25.png"
ANCHOR_BY_DISTANCE = {
    "2.5": "cam1_v60_iso100_d25.png",
    "3.5": "cam1_v60_iso100_d35.png",
    "4.5": "cam1_v60_iso100_d45.png",
    "5.5": "cam1_v60_iso100_d55.png",
    "6.5": "cam1_v60_iso100_d65_x.png",
}
BRIDGE_BY_TARGET = {
    "cam1_v60_iso1600_d35.png": "cam1_v80_iso1600_d35.png",
    "cam1_v60_iso400_d25.png": "cam1_v60_iso200_d25.png",
    "cam1_v60_iso400_d35.png": "cam1_v60_iso200_d35.png",
    "cam1_v80_iso400_d25.png": "cam1_v80_iso200_d25.png",
    "cam1_v80_iso400_d35.png": "cam1_v80_iso200_d35.png",
}
REGISTRATION_SCALE = 0.20
WIDTHS = ((0.1, 0.2, 0.3, 0.4), (0.5, 0.6, 0.7, 0.8), (0.9, 1.0))

# Constrained search boxes on the 8192 x 5460 reference image.  They locate
# the dark specimen rectangles, not the crack line itself.
REFERENCE_SEARCH = (
    ((2200, 50, 3200, 850), (3200, 50, 4100, 850),
     (4000, 50, 4900, 850), (4900, 50, 5900, 850)),
    ((2200, 1050, 3200, 1950), (3200, 1050, 4100, 1950),
     (4000, 1050, 4900, 1950), (4900, 1050, 5900, 1950)),
    ((3200, 2050, 4100, 3000), (4000, 2050, 4900, 3000)),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not decode {path}")
    return image


def write_jpeg(path: Path, image: np.ndarray, quality: int = 90) -> None:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"Could not encode {path}")
    encoded.tofile(path)


def detect_reference_cell(gray: np.ndarray, search: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = search
    crop = gray[y0:y1, x0:x1]
    blurred = cv2.GaussianBlur(crop, (9, 9), 0)
    _, mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    diagnostics: list[tuple[float, int, int, int, int, float, float]] = []
    crop_area = crop.shape[0] * crop.shape[1]
    for contour in contours:
        area = cv2.contourArea(contour)
        bx, by, bw, bh = cv2.boundingRect(contour)
        aspect = bw / max(bh, 1)
        rectangularity = area / max(bw * bh, 1)
        diagnostics.append((area, bx, by, bw, bh, aspect, rectangularity))
        if area < 0.12 * crop_area or area > 0.90 * crop_area:
            continue
        if not 0.55 <= aspect <= 1.65:
            continue
        candidates.append((area * rectangularity, (bx, by, bw, bh)))
    if not candidates:
        largest = sorted(diagnostics, reverse=True)[:8]
        raise RuntimeError(f"No reference cell in search box {search}; contours={largest}")
    _, (bx, by, bw, bh) = max(candidates, key=lambda item: item[0])
    # Inset removes the printed rectangle boundary; the interior crack is the
    # future measurement domain.
    inset_x = max(8, round(bw * 0.035))
    inset_y = max(8, round(bh * 0.035))
    left, top = x0 + bx + inset_x, y0 + by + inset_y
    right, bottom = x0 + bx + bw - inset_x, y0 + by + bh - inset_y
    return np.array(
        [[left, top], [right, top], [right, bottom], [left, bottom]],
        dtype=np.float32,
    )


def order_source_rows() -> list[dict[str, str]]:
    with SOURCE_LEDGER.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 50:
        raise RuntimeError(f"Expected 50 source-ledger rows, found {len(rows)}")
    return rows


def registration_features(image: np.ndarray, sift: cv2.SIFT) -> tuple[list, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(
        gray, None, fx=REGISTRATION_SCALE, fy=REGISTRATION_SCALE,
        interpolation=cv2.INTER_AREA,
    )
    # CLAHE improves matching across exposure settings without altering the
    # source used later for measurement.
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(small)
    keypoints, descriptors = sift.detectAndCompute(enhanced, None)
    if descriptors is None or len(keypoints) < 20:
        raise RuntimeError("Insufficient registration features")
    return keypoints, descriptors


def estimate_homography(
    ref_keypoints: list,
    ref_descriptors: np.ndarray,
    target_keypoints: list,
    target_descriptors: np.ndarray,
) -> tuple[np.ndarray, int, int, float]:
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(ref_descriptors, target_descriptors, k=2)
    good = [first for first, second in pairs if first.distance < 0.72 * second.distance]
    if len(good) < 20:
        raise RuntimeError(f"Only {len(good)} ratio-test matches")
    src = np.float32([ref_keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([target_keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    small_h, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if small_h is None or inlier_mask is None:
        raise RuntimeError("RANSAC homography failed")
    inliers = int(inlier_mask.ravel().sum())
    ratio = inliers / len(good)
    if inliers < 15 or ratio < 0.25:
        raise RuntimeError(f"Weak homography: {inliers}/{len(good)} inliers")
    scale = np.array(
        [[REGISTRATION_SCALE, 0, 0], [0, REGISTRATION_SCALE, 0], [0, 0, 1]],
        dtype=np.float64,
    )
    full_h = np.linalg.inv(scale) @ small_h @ scale
    full_h /= full_h[2, 2]
    return full_h, len(good), inliers, ratio


def polygon_qc(polygon: np.ndarray, width: int, height: int) -> tuple[bool, str, float]:
    points = polygon.reshape(4, 2)
    if not np.isfinite(points).all():
        return False, "NONFINITE", 0.0
    if (points[:, 0] < 0).any() or (points[:, 0] >= width).any():
        return False, "X_OUT_OF_BOUNDS", 0.0
    if (points[:, 1] < 0).any() or (points[:, 1] >= height).any():
        return False, "Y_OUT_OF_BOUNDS", 0.0
    area = abs(float(cv2.contourArea(points.astype(np.float32))))
    if area < 5000:
        return False, "AREA_TOO_SMALL", area
    if not cv2.isContourConvex(points.astype(np.float32)):
        return False, "NONCONVEX", area
    return True, "PASS", area


def order_box(points: np.ndarray) -> np.ndarray:
    """Return rectangle corners as top-left, top-right, bottom-right, bottom-left."""
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).ravel()
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered


def snap_cell_to_dark_rectangle(
    gray: np.ndarray, predicted: np.ndarray
) -> tuple[np.ndarray, bool, str]:
    """Snap a predicted specimen ROI to the local dark rectangular patch."""
    points = predicted.reshape(4, 2)
    min_xy = np.floor(points.min(axis=0)).astype(int)
    max_xy = np.ceil(points.max(axis=0)).astype(int)
    span = np.maximum(max_xy - min_xy, 20)
    margin = np.maximum((span * 0.80).astype(int), 30)
    x0, y0 = np.maximum(min_xy - margin, 0)
    x1, y1 = np.minimum(max_xy + margin, [gray.shape[1] - 1, gray.shape[0] - 1])
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return predicted, False, "EMPTY_SEARCH"
    blur_size = max(5, int(round(min(crop.shape) * 0.012)) | 1)
    blurred = cv2.GaussianBlur(crop, (blur_size, blur_size), 0)
    _, mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    kernel_size = max(3, int(round(min(crop.shape) * 0.012)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    predicted_area = abs(float(cv2.contourArea(points.astype(np.float32))))
    predicted_center = points.mean(axis=0) - np.array([x0, y0], dtype=np.float32)
    predicted_edges = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    predicted_aspect = max(predicted_edges) / max(min(predicted_edges), 1.0)
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not 0.25 * predicted_area <= area <= 2.50 * predicted_area:
            continue
        rectangle = cv2.minAreaRect(contour)
        (_, _), (rect_width, rect_height), _ = rectangle
        if min(rect_width, rect_height) < 20:
            continue
        rect_area = rect_width * rect_height
        rectangularity = area / max(rect_area, 1.0)
        aspect = max(rect_width, rect_height) / max(min(rect_width, rect_height), 1.0)
        if rectangularity < 0.72 or not 0.65 * predicted_aspect <= aspect <= 1.55 * predicted_aspect:
            continue
        box = order_box(cv2.boxPoints(rectangle))
        center = box.mean(axis=0)
        center_distance = np.linalg.norm((center - predicted_center) / span)
        area_penalty = abs(np.log(max(rect_area, 1.0) / max(predicted_area, 1.0)))
        aspect_penalty = abs(np.log(aspect / predicted_aspect))
        score = 3.0 * center_distance + area_penalty + aspect_penalty + 2.0 * (1.0 - rectangularity)
        candidates.append((float(score), box))
    if not candidates:
        return predicted, False, "NO_LOCAL_RECTANGLE"
    score, box = min(candidates, key=lambda item: item[0])
    if score > 2.8:
        return predicted, False, f"WEAK_LOCAL_RECTANGLE:{score:.3f}"
    box += np.array([x0, y0], dtype=np.float32)
    center = box.mean(axis=0)
    inset = center + 0.93 * (box - center)
    return inset.astype(np.float32), True, f"SNAPPED:{score:.3f}"


def template_translation(
    anchor_gray: np.ndarray,
    target_gray: np.ndarray,
    anchor_polygons: list[np.ndarray],
) -> tuple[np.ndarray, float]:
    """Estimate same-distance board translation with gradient template matching."""
    points = np.vstack(anchor_polygons).reshape(-1, 2)
    min_xy = np.floor(points.min(axis=0)).astype(int)
    max_xy = np.ceil(points.max(axis=0)).astype(int)
    span = np.maximum(max_xy - min_xy, 1)
    template_margin = np.maximum((0.10 * span).astype(int), 30)
    search_margin = np.maximum((0.45 * span).astype(int), 100)
    template_min = np.maximum(min_xy - template_margin, 0)
    template_max = np.minimum(
        max_xy + template_margin,
        [anchor_gray.shape[1] - 1, anchor_gray.shape[0] - 1],
    )
    search_min = np.maximum(template_min - search_margin, 0)
    search_max = np.minimum(
        template_max + search_margin,
        [target_gray.shape[1] - 1, target_gray.shape[0] - 1],
    )
    template = anchor_gray[
        template_min[1]:template_max[1], template_min[0]:template_max[0]
    ]
    search = target_gray[search_min[1]:search_max[1], search_min[0]:search_max[0]]
    scale = 0.20

    def edge_map(array: np.ndarray) -> np.ndarray:
        small = cv2.resize(array, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(small)
        dx = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(dx, dy)
        return cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)

    template_edge = edge_map(template)
    search_edge = edge_map(search)
    if (search_edge.shape[0] < template_edge.shape[0] or
            search_edge.shape[1] < template_edge.shape[1]):
        raise RuntimeError("Template search window is smaller than template")
    response = cv2.matchTemplate(search_edge, template_edge, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(response)
    match_xy = search_min.astype(np.float32) + np.array(location, dtype=np.float32) / scale
    delta = match_xy - template_min.astype(np.float32)
    return delta, float(score)


def render_review_tile(
    image: np.ndarray,
    polygons: list[tuple[float, np.ndarray]],
    title: str,
    registration_note: str,
    output_size: tuple[int, int] = (480, 300),
) -> np.ndarray:
    canvas = image.copy()
    all_points = np.vstack([polygon for _, polygon in polygons]).reshape(-1, 2)
    min_xy = np.floor(all_points.min(axis=0)).astype(int)
    max_xy = np.ceil(all_points.max(axis=0)).astype(int)
    span = np.maximum(max_xy - min_xy, 1)
    margin = np.maximum((span * 0.12).astype(int), 40)
    x0, y0 = np.maximum(min_xy - margin, 0)
    x1, y1 = np.minimum(max_xy + margin, [image.shape[1] - 1, image.shape[0] - 1])
    for width_mm, polygon in polygons:
        points = np.rint(polygon).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [points], True, (0, 255, 255), 14, cv2.LINE_AA)
        anchor = tuple(points[0, 0])
        cv2.putText(
            canvas, f"{width_mm:.1f}", anchor, cv2.FONT_HERSHEY_SIMPLEX,
            2.2, (0, 0, 255), 7, cv2.LINE_AA,
        )
    crop = canvas[y0:y1, x0:x1]
    tile = cv2.resize(crop, output_size, interpolation=cv2.INTER_AREA)
    scale = output_size[0] / 480.0
    header_height = round(52 * scale)
    cv2.rectangle(tile, (0, 0), (output_size[0] - 1, header_height), (0, 0, 0), -1)
    cv2.putText(tile, title, (round(8 * scale), round(20 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48 * scale,
                (255, 255, 255), max(1, round(scale)), cv2.LINE_AA)
    cv2.putText(tile, registration_note, (round(8 * scale), round(43 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42 * scale,
                (160, 255, 160), max(1, round(scale)), cv2.LINE_AA)
    return tile


def main() -> None:
    CONTACT_DIR.mkdir(parents=True, exist_ok=True)
    for stale_flag in CONTACT_DIR.glob("flagged_*.jpg"):
        stale_flag.unlink()
    source_rows = order_source_rows()
    reference_path = SOURCE_ROOT / REFERENCE_NAME
    reference = read_image(reference_path)
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    reference_cells: list[tuple[int, int, float, np.ndarray]] = []
    for row_index, (width_row, search_row) in enumerate(zip(WIDTHS, REFERENCE_SEARCH)):
        for col_index, (width_mm, search) in enumerate(zip(width_row, search_row)):
            polygon = detect_reference_cell(ref_gray, search)
            reference_cells.append((row_index, col_index, width_mm, polygon))

    sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.02)
    ref_keypoints, ref_descriptors = registration_features(reference, sift)
    anchors: dict[str, dict[str, object]] = {}
    for distance, anchor_name in ANCHOR_BY_DISTANCE.items():
        anchor_image = read_image(SOURCE_ROOT / anchor_name)
        anchor_keypoints, anchor_descriptors = registration_features(anchor_image, sift)
        if anchor_name == REFERENCE_NAME:
            ref_to_anchor = np.eye(3, dtype=np.float64)
            anchor_matches = anchor_inliers = len(ref_keypoints)
            anchor_ratio = 1.0
        else:
            try:
                ref_to_anchor, anchor_matches, anchor_inliers, anchor_ratio = estimate_homography(
                    ref_keypoints, ref_descriptors, anchor_keypoints, anchor_descriptors
                )
            except RuntimeError as error:
                raise RuntimeError(f"Anchor {anchor_name}: {error}") from error
        anchors[distance] = {
            "name": anchor_name,
            "gray": cv2.cvtColor(anchor_image, cv2.COLOR_BGR2GRAY),
            "keypoints": anchor_keypoints,
            "descriptors": anchor_descriptors,
            "ref_to_anchor": ref_to_anchor,
            "matches": anchor_matches,
            "inliers": anchor_inliers,
            "ratio": anchor_ratio,
        }
    roi_rows: list[dict[str, object]] = []
    frame_rows: list[dict[str, object]] = []
    review_tiles: list[np.ndarray] = []
    flagged_pages: list[dict[str, object]] = []

    for source in source_rows:
        path = SOURCE_ROOT / source["relative_path"]
        image = read_image(path)
        anchor = anchors[source["distance_m"]]
        if path.name == anchor["name"]:
            homography = np.asarray(anchor["ref_to_anchor"])
            matches = int(anchor["matches"])
            inliers = int(anchor["inliers"])
            inlier_ratio = float(anchor["ratio"])
            method = "REFERENCE_IDENTITY" if path.name == REFERENCE_NAME else "DISTANCE_ANCHOR"
        else:
            target_keypoints, target_descriptors = registration_features(image, sift)
            registration_error = ""
            try:
                anchor_to_target, matches, inliers, inlier_ratio = estimate_homography(
                    anchor["keypoints"], anchor["descriptors"],
                    target_keypoints, target_descriptors,
                )
            except RuntimeError as error:
                direct_error = str(error)
                bridge_name = BRIDGE_BY_TARGET.get(path.name)
                bridge_succeeded = False
                if bridge_name is not None:
                    bridge_image = read_image(SOURCE_ROOT / bridge_name)
                    bridge_keypoints, bridge_descriptors = registration_features(
                        bridge_image, sift
                    )
                    try:
                        anchor_to_bridge, _, _, _ = estimate_homography(
                            anchor["keypoints"], anchor["descriptors"],
                            bridge_keypoints, bridge_descriptors,
                        )
                        bridge_to_target, matches, inliers, inlier_ratio = estimate_homography(
                            bridge_keypoints, bridge_descriptors,
                            target_keypoints, target_descriptors,
                        )
                    except RuntimeError as bridge_error:
                        registration_error = f"{direct_error}; bridge={bridge_name}:{bridge_error}"
                    else:
                        homography = (
                            bridge_to_target @ anchor_to_bridge
                            @ np.asarray(anchor["ref_to_anchor"])
                        )
                        homography /= homography[2, 2]
                        method = "DISTANCE_BRIDGE_SIFT_RANSAC"
                        registration_error = f"direct_failed={direct_error}; bridge={bridge_name}"
                        bridge_succeeded = True
                if not bridge_succeeded:
                    homography = np.asarray(anchor["ref_to_anchor"]).copy()
                    matches = inliers = 0
                    inlier_ratio = 0.0
                    anchor_polygons = [
                        cv2.perspectiveTransform(
                            ref_polygon.reshape(1, -1, 2), homography
                        ).reshape(-1, 2)
                        for _, _, _, ref_polygon in reference_cells
                    ]
                    delta, template_score = template_translation(
                        anchor["gray"], cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                        anchor_polygons,
                    )
                    translation = np.eye(3, dtype=np.float64)
                    translation[0, 2], translation[1, 2] = delta
                    homography = translation @ homography
                    method = "DISTANCE_ANCHOR_TEMPLATE_FALLBACK"
                    registration_error = (
                        f"{registration_error or direct_error}; "
                        f"template_score={template_score:.6f}"
                    )
            else:
                homography = anchor_to_target @ np.asarray(anchor["ref_to_anchor"])
                homography /= homography[2, 2]
                method = "DISTANCE_ANCHOR_SIFT_RANSAC"
        if path.name == anchor["name"]:
            registration_error = ""

        target_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        predicted_cells: list[tuple[int, int, float, np.ndarray]] = []
        initial_deltas: list[np.ndarray] = []
        for row_index, col_index, width_mm, ref_polygon in reference_cells:
            predicted = cv2.perspectiveTransform(
                ref_polygon.reshape(1, -1, 2), homography
            ).reshape(-1, 2)
            snapped_polygon, snapped, _ = snap_cell_to_dark_rectangle(
                target_gray, predicted
            )
            if snapped:
                initial_deltas.append(snapped_polygon.mean(axis=0) - predicted.mean(axis=0))
            predicted_cells.append((row_index, col_index, width_mm, predicted))

        # When feature registration fails, the board may shift between passes.
        # The median displacement from locally detected cells supplies a robust
        # translation refinement while preserving the anchor scale/perspective.
        if len(initial_deltas) >= 3:
            median_delta = np.median(np.vstack(initial_deltas), axis=0).astype(np.float32)
        else:
            median_delta = np.zeros(2, dtype=np.float32)

        transformed: list[tuple[float, np.ndarray]] = []
        frame_pass = True
        frame_notes: list[str] = []
        for row_index, col_index, width_mm, predicted_polygon in predicted_cells:
            refined_prediction = predicted_polygon + median_delta
            polygon, snapped, snap_note = snap_cell_to_dark_rectangle(
                target_gray, refined_prediction
            )
            passed, note, area = polygon_qc(
                polygon, image.shape[1], image.shape[0]
            )
            passed = passed and snapped
            frame_pass &= passed
            if not passed:
                frame_notes.append(f"{width_mm:.1f}:{note}/{snap_note}")
            transformed.append((width_mm, polygon))
            flat = polygon.reshape(-1)
            roi_rows.append(
                {
                    "source_id": source["source_id"],
                    "filename": source["filename"],
                    "source_sha256": source["sha256"],
                    "row_index": row_index,
                    "col_index": col_index,
                    "nominal_width_mm": f"{width_mm:.1f}",
                    "p0_x": f"{flat[0]:.3f}", "p0_y": f"{flat[1]:.3f}",
                    "p1_x": f"{flat[2]:.3f}", "p1_y": f"{flat[3]:.3f}",
                    "p2_x": f"{flat[4]:.3f}", "p2_y": f"{flat[5]:.3f}",
                    "p3_x": f"{flat[6]:.3f}", "p3_y": f"{flat[7]:.3f}",
                    "projected_area_px2": f"{area:.3f}",
                    "auto_qc": f"{note}/{snap_note}",
                    "human_review": "PENDING",
                    "final_status": "AUTO_CANDIDATE",
                }
            )
        note = f"{method}; inliers={inliers}/{matches} ({inlier_ratio:.3f})"
        review_tiles.append(render_review_tile(image, transformed, path.name, note))
        if not frame_pass:
            flagged_path = CONTACT_DIR / f"flagged_{path.stem}.jpg"
            write_jpeg(
                flagged_path,
                render_review_tile(
                    image, transformed, path.name, note, output_size=(1440, 900)
                ),
                quality=94,
            )
            flagged_pages.append(
                {"filename": path.name, "path": str(flagged_path)}
            )
        frame_rows.append(
            {
                "source_id": source["source_id"],
                "filename": source["filename"],
                "method": method,
                "distance_anchor": anchor["name"],
                "ratio_test_matches": matches,
                "ransac_inliers": inliers,
                "inlier_ratio": inlier_ratio,
                "registration_error": registration_error,
                "local_translation_refinement_px": median_delta.tolist(),
                "auto_qc": "PASS" if frame_pass else ";".join(frame_notes),
                "homography": homography.tolist(),
            }
        )

    fieldnames = list(roi_rows[0])
    with ROI_LEDGER.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(roi_rows)

    pages: list[dict[str, object]] = []
    for page_index in range(0, len(review_tiles), 25):
        page_tiles = review_tiles[page_index:page_index + 25]
        page = np.zeros((5 * 300, 5 * 480, 3), dtype=np.uint8)
        for local_index, tile in enumerate(page_tiles):
            row, col = divmod(local_index, 5)
            page[row * 300:(row + 1) * 300, col * 480:(col + 1) * 480] = tile
        page_path = CONTACT_DIR / f"width_roi_review_{page_index // 25 + 1:02d}.jpg"
        write_jpeg(page_path, page, quality=92)
        pages.append({"path": str(page_path), "sha256": sha256_file(page_path)})
    for flagged_page in flagged_pages:
        flagged_page["sha256"] = sha256_file(Path(flagged_page["path"]))

    freeze = {
        "schema_version": "MERIT-C009-WIDTH-ROI-CANDIDATES-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "AUTO_CANDIDATE_REVIEW_REQUIRED",
        "quantitative_measurement_allowed": False,
        "source_ledger": {"path": str(SOURCE_LEDGER), "sha256": sha256_file(SOURCE_LEDGER)},
        "reference": {
            "filename": REFERENCE_NAME,
            "sha256": sha256_file(reference_path),
            "layout": [list(row) for row in WIDTHS],
            "cell_polygons_px": [
                {
                    "row_index": row, "col_index": col, "nominal_width_mm": width,
                    "polygon": polygon.tolist(),
                }
                for row, col, width, polygon in reference_cells
            ],
        },
        "registration": {
            "method": "SIFT ratio-test plus RANSAC planar homography",
            "scale": REGISTRATION_SCALE,
            "frame_count": len(frame_rows),
            "auto_qc_pass": sum(row["auto_qc"] == "PASS" for row in frame_rows),
            "minimum_nonreference_inliers": min(
                row["ransac_inliers"] for row in frame_rows
                if row["method"] == "DISTANCE_ANCHOR_SIFT_RANSAC"
            ),
            "minimum_nonreference_inlier_ratio": min(
                row["inlier_ratio"] for row in frame_rows
                if row["method"] == "DISTANCE_ANCHOR_SIFT_RANSAC"
            ),
            "frames": frame_rows,
        },
        "roi_ledger": {
            "path": str(ROI_LEDGER),
            "sha256": sha256_file(ROI_LEDGER),
            "rows": len(roi_rows),
            "expected_rows": 500,
        },
        "review_pages": pages,
        "strict_qc_flagged_pages": flagged_pages,
        "closure_requirement": (
            "Review every rendered frame, record ACCEPT/REJECT per ROI, correct rejected "
            "polygons without using restoration outputs, then freeze a separate final ledger."
        ),
        "generator": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
    }
    FREEZE_PATH.write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": freeze["status"],
        "frames": len(frame_rows),
        "roi_rows": len(roi_rows),
        "auto_qc_pass": freeze["registration"]["auto_qc_pass"],
        "minimum_nonreference_inliers": freeze["registration"]["minimum_nonreference_inliers"],
        "minimum_nonreference_inlier_ratio": freeze["registration"]["minimum_nonreference_inlier_ratio"],
        "roi_ledger": str(ROI_LEDGER),
        "review_pages": [page["path"] for page in pages],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
