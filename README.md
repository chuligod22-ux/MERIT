# MERIT — Measurement-Fidelity Restoration for Inspection of Tunnels

Reproducibility package for the paper *"MERIT: Measurement-Fidelity Restoration for
High-Speed Tunnel-Lining Inspection — Recovering Chart-Measured Resolution and Crack
Detectability without Hallucination."*

MERIT is a measurement-fidelity restoration **framework** (not a new network) built on the
Measurement–Perception–Distortion (MPD) principle: a measurement-calibrated degradation model,
a measurement-prompt–conditioned restorer (NAFNet backbone, interchangeable), a fidelity +
frozen-segmenter-in-loop task loss, and an ISO 12233 chart-MTF50 hallucination test.

This package reproduces **all tables and figures** in the paper from the released code and
processed data — no raw images required.

## Repository layout

```
.
├── src/                 # model / degradation / training / evaluation / figure code
├── data/
│   ├── processed/       # CSV + JSON results that back every table and figure
│   └── samples/         # a few representative frames (added on release)
├── third_party/FETCH.md # how to obtain the segmenter / DiffBIR / SD-x4 (not redistributed)
├── requirements.txt     # Python dependencies
├── CITATION.cff
├── ANONYMIZATION_LOG.md
├── LICENSE-CODE         # MIT (code)
└── LICENSE-DATA         # CC-BY 4.0 (data)
```
Trained weights (MERIT + the eight backbone-transfer checkpoints + ablations, ~180 MB) are
distributed as **GitHub Release assets**, not in the git tree.

## Install

```bash
# Python 3.10+; a CUDA GPU is needed for training and for the diffusion baselines.
pip install -r requirements.txt        # or: uv pip install -r requirements.txt
```

## Configuration (paths)

All paths default to locations inside this repository (resolved from each script's location) and
can be overridden with environment variables — no machine-specific absolute paths are hard-coded:

| Variable | Default | Holds |
|----------|---------|-------|
| `MERIT_DATA` | `data/processed/` | processed CSV/JSON inputs (back every table/figure) |
| `MERIT_FIGS` | `figs/` | generated figures (output) |
| `MERIT_WEIGHTS` | `weights/` | trained `.pt` checkpoints (downloaded from GitHub Releases) |
| `MERIT_NRIQA` | `data/raw/` | raw cam1/cam2 frames + crack segmenter (available on request) |
| `MERIT_TMP` | `tmp/` | scratch directory (e.g. DiffBIR outputs) |

Figure scripts (`make_*.py`) reproduce all figures from `MERIT_DATA` alone. Training and
evaluation scripts additionally require the weights, the raw frames, the segmenter, and a GPU.

## Reproduce the results

Numbers/figures are produced from `data/processed/` (already included). To regenerate from
scratch, run training then evaluation (a GPU is required):

```bash
# 1. backbone-transfer models (8 = 4 backbones x {ON, OFF})   -> data/processed/backbone_axes.csv
python src/train_backbone.py        # writes weights/restorer_<backbone>_<on|off>.pt
python src/eval_backbones.py

# 2. classical / generative battery (Table 2)                  -> mpd_axes.csv
python src/eval_classical.py
python src/eval_diffusion.py         # SD-x4 (diffusers); DiffBIR via third_party/FETCH.md

# 3. detection-fitness recovery (Table 3, Figs)                -> full_battery_downstream.csv
python src/eval_detection.py
python src/eval_falsecrack.py        # hallucination / false-crack test

# 4. ablation (Table 4)                                        -> eval_disentangled.py outputs
python src/eval_disentangled.py

# 5. all figures
python src/make_figures.py           # + the other make_*.py figure scripts
```

See the per-script header comments for inputs/outputs. The reference degradation is
σ_h ≈ 2.6 / σ_v ≈ 2.4 px blur with σ_n = 2.5 DN noise; the clean chart MTF50 is 0.077 cy/px.

## Data

- **Processed data** (`data/processed/`, CC-BY 4.0): all per-condition / per-method results
  (MTF50, PSNR, LPIPS, noise, detection recovery, false-crack), which back every table and figure.
- **Representative frames** (`data/samples/`): a small anonymized subset for quick inspection.
- **Full raw archive** (cam1 ISO 12233 chart frames, cam2 tunnel-lining crack frames):
  available from the corresponding author on reasonable request.

## Third-party components (not redistributed)

The crack segmenter (UNet-VGG16, DeepCrack family), DiffBIR, and the Stable-Diffusion ×4
upscaler are used as baselines but **not** included here. See `third_party/FETCH.md` for
download-and-run instructions.

## Reproducibility notes

- SD-x4 is seeded (`torch.Generator(...).manual_seed(0)`); DiffBIR is run via its own CLI.
- Korean file paths: use a Unicode-aware image reader (`imread` + `imdecode`).

## License

- **Code** — MIT (`LICENSE-CODE`).
- **Data** — CC-BY 4.0 (`LICENSE-DATA`).

## Citation

See `CITATION.cff`. Please cite the paper if you use this code or data.
