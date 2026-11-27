# 1. Optical evaluation

The acquisition-grid experiment tests an optical decision rule against always restoring or keeping the raw chart. A separate fixed-chart experiment compares restoration outputs against a known undegraded image. Their reference images, estimators, and experimental units differ.

## 1.1 Reproduce the reported results

Run these commands from the extracted package root, using a Python environment with NumPy, SciPy, Pillow, and OpenCV.

```bash
python -m pip install numpy scipy pillow opencv-python-headless
python -B optics/gate/refit.py
python -B optics/chart/verify.py
```

The first command after installation refits 150 held-condition folds and verifies 120 summary rows for each model. The second rebuilds the common degraded input and rescores all 12 fixed-chart rows for MTF50, recovery relative to the source, absolute MTF50 error, PSNR, and flat-region intensity SD. It also checks both 16-condition sweep summaries and the archived byte and decoded-array SHA-256 hashes. Both scripts exit with an error on a mismatch. Run them with the normal Python interpreter (assertions enabled).

With the corresponding optional packages and AlexNet weights available, verify LPIPS and regenerate Figures 5–7 in a separate output directory.

```bash
python -m pip install torch torchvision lpips matplotlib
python -B optics/chart/verify.py --lpips --figures regenerated-figures
```

## 1.2 Acquisition grid and optical gate

The 479 controlled cam1 frames contribute 1,737 proposed ROIs across 50 acquisition conditions. For each model, 1,731 ROIs have valid values for every candidate and seed. Each JSON line in `gate/{final,precursor}/measurements.jsonl.gz` identifies one public frame and its ROI IDs. The aligned `raw` vector contains measured MTF50 values in cycles/pixel. Each `routes` entry contains the checkpoint seed, candidate index, frame-level safety-bypass flag, and candidate output MTF50 values on the same ROIs. A null value denotes an unavailable measurement.

The 4,953,697-parameter precursor and 17,111,907-parameter final model each have three seeds. `refit.py` forms the per-frame mean over matched ROIs, takes the condition median over frames, and compares with raw MTF50 of the 60 km/h, ISO 100 reference at the same distance. It fits 16 coefficient settings and eight threshold choices for each held condition and seed. For each of five held reference conditions, it excludes all ten conditions from the same distance during fitting. Each of the other 45 folds uses the other 49 conditions.

`folds.csv` contains the selected coefficients, margins, decisions, and held errors. The `always` policy keeps the selected coefficients and the frame safety bypass, `unguarded` additionally disables the safety bypass, and `always_tuned` fits its own coefficients. `conditions.csv` averages the three held predictions for each condition. `coverage.csv` gives the matched and available ROI/frame counts. `summary.csv` contains condition means and descriptive percentile intervals using 10,000 resamples with the held predictions fixed. The final-model gated-minus-always difference is −0.002561 cycles/pixel (95% interval −0.003752 to −0.001466). This is a post hoc internal validation on the acquisition grid used during method development. The verifier starts from ROI measurements for each candidate and seed.

## 1.3 Controlled fixed chart and paired input

The 292 × 492 author-controlled chart is a pixel-exact subregion of the public cam1 crop for `60km_2.5m_ISO100/frame_000102.png`. Its coordinates in that public crop are x = 402–693, y = 16–507, inclusive. To check the relationship using an extracted cam1 archive, pass the path to its union crop with `--public-crop`. `chart/provenance.json` records coordinate bounds, file and decoded-array hashes, image modes, and array shapes. [Public cam1 source crops](https://github.com/chuligod22-ux/MERIT/releases/tag/revision-cam1-optical-crops-v2)

All 12 rows in `chart/fixed_chart.csv` use a common 8-bit degraded input. It uses Gaussian blur with horizontal and vertical SDs of 2.6 and 2.4 pixels and Gaussian noise of 2.5 DN (NumPy RandomState seed 5), followed by clipping and truncation to 8 bits. The three PNG color channels match. The central 196 × 396 pixels are scored against the clean reference (MTF50 = 0.07731303 cycles/pixel). `chart/scoring_rois.npz` preserves every method’s central scoring array before visualization, including floating-point values outside the display range. The arrays use the exact method names from the CSV. Figure 5 uses the common 2nd–98th percentile range of the reference, 22–62 DN. The fixed chart yields 34.61 dB for SD-x4, 38.67 dB for DiffBIR v2.1, and 44.21 dB for MERIT. `chart/figures/` contains the three manuscript PDFs.

`chart/sweep16.csv` describes the separate floating-point input sweep for NAFNet-big and MERIT. Each model has four horizontal blur levels and four noise seeds. The vertical blur SD is 0.9 times the horizontal SD, and the noise SD is 2.5 DN. The verifier recomputes the means and population SDs from the stored rows.

## 1.4 Model output and optional inference

`chart/common_degraded.png` is the exact input supplied to SD-x4 and DiffBIR v2.1. The two corresponding PNGs are outputs produced on the authors’ controlled chart. `chart/sdx4_run.json` records the fixed SD-x4 Hub revision, empty prompt, seed 5, DDIM 50 steps, guidance scale 0, noise level 20, and conversion of the 4× output back to the input size. To repeat SD-x4 inference on CUDA, obtain the fixed model snapshot separately and pass its local revision directory to the script. The recorded run used PyTorch 2.11.0+cu128, diffusers 0.37.1 and transformers 5.9.0. The external model provider supplies the weights and access terms. [Pinned SD-x4 model snapshot](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler/tree/572c99286543a273bfd17fac263db5a77be12c4c)

```bash
python -B optics/chart/run_sdx4.py --snapshot /path/to/snapshots/572c99286543a273bfd17fac263db5a77be12c4c --output-dir regenerated-sdx4
```

The packaged verification uses the archived author-produced arrays and requires no diffusion model download. The chart audit also records the two author checkpoint hashes used for the fixed-chart network rows. The optical-gate experiment above evaluates the two model capacities separately.
