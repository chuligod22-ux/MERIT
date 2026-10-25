"""Recompose Figure 15 with a fixed 2x source-selected crack ROI.

The numerical panels and annotations come from the frozen Figure 15 v1
artifact.  Only the interiors of the 18 qualitative image tiles are replaced
with the top-left 224 x 224 region of the same 448 x 448 crop, enlarged to the
original tile size.  The operation is therefore a display-only magnification:
it does not alter the evaluated arrays, metrics, selected sample, or seed.

This script exists so that the display correction can be reproduced without
rerunning the CUDA model.  The source-generation script uses the identical
ROI directly on the underlying arrays when a CUDA environment is available.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
from pathlib import Path
from typing import Any

import fitz
import matplotlib
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import find_objects, label


HERE = Path(__file__).resolve().parent
C018 = HERE.parent
REVISION = C018.parent
BASE_ROOT = C018 / "runs" / "figure15_v1"
OUTPUT_ROOT = C018 / "runs" / "figure15_v3"
MANUSCRIPT_R3 = REVISION / "manuscript_r3_6"
MANUSCRIPT_R2 = REVISION / "manuscript_r2_4"

BASE_PNG = BASE_ROOT / "fig15_perturbed_degradation.png"
BASE_PDF = BASE_ROOT / "fig15_perturbed_degradation.pdf"
BASE_PROVENANCE = BASE_ROOT / "fig15_provenance.json"

# Border colours used by the frozen plotting script.
REFERENCE_BORDER = (107, 114, 128)
INPUT_BORDER = (217, 119, 6)
OUTPUT_BORDER = (31, 119, 180)
EXPECTED_PANELS = 6
COLOUR_TOLERANCE = 8
MIN_BORDER_PIXELS = 6_000
INTERIOR_INSET_PX = 5
FOOTER_CROP_FRACTION = 0.965


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def border_boxes(rgb: np.ndarray, colour: tuple[int, int, int]) -> list[tuple[int, int, int, int]]:
    distance = np.max(
        np.abs(rgb.astype(np.int16) - np.asarray(colour, dtype=np.int16)), axis=2
    )
    components, _count = label(distance < COLOUR_TOLERANCE)
    boxes: list[tuple[int, int, int, int]] = []
    for index, slices in enumerate(find_objects(components), start=1):
        if slices is None:
            continue
        area = int(np.count_nonzero(components[slices] == index))
        y0, y1 = int(slices[0].start), int(slices[0].stop)
        x0, x1 = int(slices[1].start), int(slices[1].stop)
        if area >= MIN_BORDER_PIXELS and (x1 - x0) > 400 and (y1 - y0) > 400:
            boxes.append((x0, y0, x1, y1))
    boxes.sort(key=lambda box: (box[1], box[0]))
    if len(boxes) != EXPECTED_PANELS:
        raise RuntimeError(f"Expected {EXPECTED_PANELS} panel boxes for {colour}, found {len(boxes)}")
    return boxes


def zoomed_tile(rgb: np.ndarray, box: tuple[int, int, int, int]) -> tuple[Image.Image, tuple[int, int, int, int]]:
    x0, y0, x1, y1 = box
    target = (
        x0 + INTERIOR_INSET_PX,
        y0 + INTERIOR_INSET_PX,
        x1 - INTERIOR_INSET_PX,
        y1 - INTERIOR_INSET_PX,
    )
    tx0, ty0, tx1, ty1 = target
    interior = Image.fromarray(rgb[ty0:ty1, tx0:tx1])
    width, height = interior.size
    # The rendered interior maps the full 448 x 448 source crop.  Its top-left
    # half is therefore the fixed source-only ROI [0:224, 0:224].
    roi = interior.crop((0, 0, width // 2, height // 2))
    enlarged = roi.resize((width, height), Image.Resampling.LANCZOS)
    return enlarged, target


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def grayscale(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image).astype(np.float32) / 255.0
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def add_error_inset(image: Image.Image, error: np.ndarray, vmax: float) -> Image.Image:
    revised = image.copy()
    width, height = revised.size
    size = int(round(min(width, height) * 0.38))
    margin = int(round(min(width, height) * 0.04))
    normalised = np.clip(error / max(vmax, 1e-6), 0.0, 1.0)
    colour = matplotlib.colormaps["magma"](normalised, bytes=True)[..., :3]
    inset = Image.fromarray(colour).resize((size, size), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(revised)
    x0 = width - margin - size
    y0 = height - margin - size
    border = 3
    draw.rectangle(
        (x0 - border, y0 - border, x0 + size + border - 1, y0 + size + border - 1),
        fill="white",
    )
    revised.paste(inset, (x0, y0))
    draw = ImageDraw.Draw(revised)
    draw.rectangle((x0 + 3, y0 + 3, x0 + 42, y0 + 20), fill=(0, 0, 0))
    draw.text((x0 + 6, y0 + 4), "|diff|", fill="white")
    return revised


def main() -> None:
    for path in (BASE_PNG, BASE_PDF, BASE_PROVENANCE):
        if not path.exists():
            raise FileNotFoundError(path)

    base_provenance = json.loads(BASE_PROVENANCE.read_text(encoding="utf-8"))
    base_image = Image.open(BASE_PNG).convert("RGB")
    rgb = np.asarray(base_image)
    height, width = rgb.shape[:2]

    grouped_boxes = {
        "reference": border_boxes(rgb, REFERENCE_BORDER),
        "degraded_input": border_boxes(rgb, INPUT_BORDER),
        "merit_output": border_boxes(rgb, OUTPUT_BORDER),
    }
    replacements: list[tuple[Image.Image, tuple[int, int, int, int]]] = []
    revised_image = base_image.copy()
    for panel_index in range(EXPECTED_PANELS):
        reference, reference_target = zoomed_tile(
            rgb, grouped_boxes["reference"][panel_index]
        )
        degraded, degraded_target = zoomed_tile(
            rgb, grouped_boxes["degraded_input"][panel_index]
        )
        restored, restored_target = zoomed_tile(
            rgb, grouped_boxes["merit_output"][panel_index]
        )
        reference_gray = grayscale(reference)
        degraded_error = np.abs(
            grayscale(degraded.resize(reference.size, Image.Resampling.BILINEAR))
            - reference_gray
        )
        restored_error = np.abs(
            grayscale(restored.resize(reference.size, Image.Resampling.BILINEAR))
            - reference_gray
        )
        vmax = max(
            float(
                np.percentile(
                    np.concatenate([degraded_error.ravel(), restored_error.ravel()]),
                    99.0,
                )
            ),
            1e-6,
        )
        degraded = add_error_inset(degraded, degraded_error, vmax)
        restored = add_error_inset(restored, restored_error, vmax)
        for tile, target in (
            (reference, reference_target),
            (degraded, degraded_target),
            (restored, restored_target),
        ):
            revised_image.paste(tile, (target[0], target[1]))
            replacements.append((tile, target))

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_png = OUTPUT_ROOT / "fig15_perturbed_degradation.png"
    output_pdf = OUTPUT_ROOT / "fig15_perturbed_degradation.pdf"
    # The figure-level grey footer duplicates the manuscript caption.  Crop it
    # after the frozen tile recomposition while retaining all panel captions.
    cropped_height = int(round(height * FOOTER_CROP_FRACTION))
    revised_image = revised_image.crop((0, 0, width, cropped_height))
    revised_image.save(output_png, format="PNG", dpi=(300, 300), optimize=True)

    with fitz.open(BASE_PDF) as document:
        if document.page_count != 1:
            raise RuntimeError("Figure 15 base PDF must contain exactly one page")
        page = document[0]
        for tile, (x0, y0, x1, y1) in replacements:
            rectangle = fitz.Rect(
                x0 / width * page.rect.width,
                y0 / height * page.rect.height,
                x1 / width * page.rect.width,
                y1 / height * page.rect.height,
            )
            page.insert_image(
                rectangle,
                stream=png_bytes(tile),
                keep_proportion=False,
                overlay=True,
            )
        footer = fitz.Rect(
            0,
            page.rect.height * FOOTER_CROP_FRACTION,
            page.rect.width,
            page.rect.height,
        )
        page.add_redact_annot(footer, fill=(1, 1, 1))
        page.apply_redactions()
        page.set_cropbox(
            fitz.Rect(
                0,
                0,
                page.rect.width,
                page.rect.height * FOOTER_CROP_FRACTION,
            )
        )
        document.save(output_pdf, garbage=4, deflate=True)

    destinations: list[str] = []
    # The active readable manuscript is R3-6.  Keep the archived R2-4 package
    # byte-stable while applying this presentation-only revision.
    for directory in (MANUSCRIPT_R3,):
        directory.mkdir(parents=True, exist_ok=True)
        for source, name in ((output_png, "fig15.png"), (output_pdf, "fig15.pdf")):
            destination = directory / name
            shutil.copy2(source, destination)
            destinations.append(str(destination.resolve()))

    provenance = {
        "schema_version": "MERIT-C018-FIGURE15-v3",
        "selection_status": base_provenance["selection_status"],
        "selection_rule": base_provenance["selection_rule"],
        "selection": base_provenance["selection"],
        "severity": base_provenance["severity"],
        "display_gamma": base_provenance["display_gamma"],
        "metrics_use_untransformed_arrays": True,
        "families": base_provenance["families"],
        "qualitative_roi_y0_y1_x0_x1": [0, 224, 0, 224],
        "qualitative_zoom_factor": 2.0,
        "qualitative_roi_selected_from": "SOURCE_IMAGE_ONLY",
        "qualitative_error_insets": (
            "Absolute display-domain difference from the reference; degraded and restored "
            "insets share the same family-specific 99th-percentile scale."
        ),
        "display_change_only": True,
        "redundant_footer_removed": True,
        "footer_crop_fraction": FOOTER_CROP_FRACTION,
        "display_recomposition": (
            "Each frozen 448 x 448 qualitative tile was replaced by its rendered "
            "top-left 224 x 224 source-only ROI at 2x magnification. Numerical panels, "
            "annotations, metrics, sample selection, and stochastic seed are unchanged."
        ),
        "rendered_base_png_sha256": sha256_file(BASE_PNG),
        "rendered_base_pdf_sha256": sha256_file(BASE_PDF),
        "rendered_base_provenance_sha256": sha256_file(BASE_PROVENANCE),
        "source_generation_script_sha256": sha256_file(
            HERE / "plot_c018_perturbed_degradation.py"
        ),
        "recomposition_script_sha256": sha256_file(Path(__file__).resolve()),
        "png_sha256": sha256_file(output_png),
        "pdf_sha256": sha256_file(output_pdf),
        "destinations": destinations,
    }
    provenance_path = OUTPUT_ROOT / "fig15_provenance.json"
    write_json(provenance_path, provenance)
    shutil.copy2(provenance_path, MANUSCRIPT_R3 / "fig15_provenance.json")
    shutil.copy2(provenance_path, MANUSCRIPT_R2 / "fig15_provenance.json")
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
