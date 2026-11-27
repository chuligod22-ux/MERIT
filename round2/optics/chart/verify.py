"""Verify the fixed chart from the archived scoring arrays; optional AlexNet LPIPS and PDF regeneration."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from slanted_edge import esf_mtf


HERE = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_rows(filename):
    with (HERE / filename).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def check_sweep(reference, audit):
    rows = read_rows("sweep16.csv")
    assert len(rows) == 32
    for model, summary in audit["sweep_original_float_protocol"].items():
        part = [r for r in rows if r["model"] == model]
        assert {(float(r["sigma_h_px"]), int(r["noise_seed"])) for r in part} == {
            (blur, seed) for blur in [1.5, 2.0, 2.5, 3.0] for seed in range(4)}
        mtfs = [float(r["mtf50"]) for r in part]
        scores = [float(r["psnr_db"]) for r in part]
        assert len(part) == summary["n"] == 16
        np.testing.assert_allclose([np.mean(mtfs), np.std(mtfs), 100*np.mean(mtfs)/reference,
                                    np.mean(scores)], [summary["mtf_mean"], summary["mtf_sd"],
                                    summary["reference_percent"], summary["psnr_mean"]], atol=1e-10, rtol=0)


def check_perceptual(arrays, clean, rows, device):
    import lpips
    import torch

    metric = lpips.LPIPS(net="alex").to(device).eval()

    def rgb(array):
        scaled = np.clip(array, 0, 255) / 127.5 - 1
        return torch.from_numpy(scaled)[None, None].float().to(device).repeat(1, 3, 1, 1)

    with torch.inference_mode():
        target = rgb(clean)
        for row in rows:
            value = metric(rgb(arrays[row["method"]]), target).item()
            np.testing.assert_allclose(value, row["lpips_alex"], atol=1e-5, rtol=0)
    print("AlexNet LPIPS verified for all 12 rows.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-crop", type=Path, help="Optional public cam1 frame_000102.png union crop")
    parser.add_argument("--lpips", action="store_true", help="Recompute LPIPS using pretrained AlexNet")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"), help="Optional LPIPS device")
    parser.add_argument("--figures", type=Path, help="Optional directory for regenerated PDFs")
    args = parser.parse_args()
    prov = json.loads((HERE / "provenance.json").read_text(encoding="utf-8"))
    audit = json.loads((HERE / "chart_audit.json").read_text(encoding="utf-8"))
    sd = json.loads((HERE / "sdx4_run.json").read_text(encoding="utf-8"))
    images = {}
    for name, manifest in prov["images"].items():
        path = HERE / name
        with Image.open(path) as im:
            assert im.mode == manifest["mode"]
            arr = np.array(im)
        assert list(arr.shape) == manifest["shape"] and str(arr.dtype) == manifest["dtype"]
        assert sha(path.read_bytes()) == manifest["encoded_png_sha256"]
        assert sha(arr.tobytes(order="C")) == manifest["decoded_c_order_sha256"]
        images[name] = arr
    assert audit["original_chart_sha256"] == prov["source_frame_encoded_sha256"]
    assert sd["input_sha256"] == audit["input_sha256"] == prov["images"]["common_degraded.png"]["encoded_png_sha256"]
    assert sd["output_sha256"] == prov["images"]["sdx4_chart.png"]["encoded_png_sha256"]
    assert audit["diffbir_output_sha256"] == prov["images"]["diffbir_v21_chart.png"]["encoded_png_sha256"]
    if args.public_crop:
        assert sha(args.public_crop.read_bytes()) == prov["public_crop_encoded_sha256"]
        with Image.open(args.public_crop) as parent:
            crop = np.array(parent.crop(prov["chart_xyxy_in_public_crop"]))
        np.testing.assert_array_equal(crop, images["source_chart.png"])

    big = images["source_chart.png"].astype(np.float64)
    float_input = gaussian_filter(big, (2.4, 2.6)) + np.random.RandomState(5).randn(*big.shape) * 2.5
    shared = np.clip(float_input, 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(np.repeat(shared[:, :, None], 3, axis=2), images["common_degraded.png"])
    delta = shared.astype(np.float64) - float_input
    np.testing.assert_allclose([delta.mean(), np.abs(delta).max()],
                               [audit["uint8_mean_minus_float_dn"], audit["uint8_max_abs_difference_dn"]])
    x0, y0, x1, y1 = prov["score_roi_xyxy_in_chart"]
    bounds = np.s_[y0:y1, x0:x1]
    clean = big[bounds]
    reference = float(esf_mtf(clean)[2])
    np.testing.assert_allclose(reference, audit["reference_mtf50"], rtol=0, atol=1e-11)
    np.testing.assert_array_equal(np.percentile(clean, audit["source_display_percentiles"]),
                                  audit["shared_source_display_limits_dn"])
    diffbir = cv2.cvtColor(images["diffbir_v21_chart.png"], cv2.COLOR_RGB2GRAY)
    archive = prov["scoring_archive"]
    archive_path = HERE / archive["file"]
    assert sha(archive_path.read_bytes()) == archive["encoded_npz_sha256"]
    with np.load(archive_path, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    rows = [{k: v if k == "method" else float(v) for k, v in row.items()}
            for row in read_rows("fixed_chart.csv")]
    assert len(rows) == len(arrays) == 12 and set(arrays) == {r["method"] for r in rows}
    for name, full in [("Degraded input", shared), ("SD-x4", images["sdx4_chart.png"]),
                       ("DiffBIR v2.1", diffbir)]:
        np.testing.assert_array_equal(full[bounds], arrays[name])
    for row in rows:
        im = arrays[row["method"]]
        stamp = archive["arrays"][row["method"]]
        assert sha(im.tobytes(order="C")) == stamp["c_order_sha256"] and im.dtype.str == stamp["dtype"]
        assert list(im.shape) == stamp["shape"]
        assert im.shape == clean.shape and im.dtype == np.float64 and np.isfinite(im).all()
        mtf = float(esf_mtf(im)[2])
        mse = np.mean((np.clip(im, 0, 255) - clean) ** 2)
        measured = [mtf, 100*mtf/reference, abs(mtf-reference), 10*np.log10(255**2 / mse), im[:, :40].std()]
        fields = ["mtf50", "reference_percent", "absolute_mtf_error", "psnr_db", "flat_sd_dn"]
        np.testing.assert_allclose(measured, [row[key] for key in fields], atol=1e-8, rtol=0, err_msg=row["method"])
    check_sweep(reference, audit)
    print("PNG and decoded-pixel hashes, fixed input, 12 chart rows and 2 x 16 sweep aggregates verified.")
    for row in rows:
        if row["method"] in ("SD-x4", "DiffBIR v2.1", "MERIT"):
            print(f"{row['method']}: MTF50 {row['mtf50']:.5f} cycles/pixel; PSNR {row['psnr_db']:.2f} dB")
    if args.lpips:
        check_perceptual(arrays, clean, rows, args.device)
    if args.figures:
        from plot_figures import plot_all
        if args.figures.resolve() == (HERE / "figures").resolve():
            parser.error("Choose a separate directory for regenerated figures")
        args.figures.mkdir(exist_ok=True, parents=True)
        plot_all(clean, arrays, rows, reference, args.figures)
        print(f"Figures 5–7 written to {args.figures}")


if __name__ == "__main__":
    main()
