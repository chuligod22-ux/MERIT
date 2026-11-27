# MERIT — Measurement-Fidelity Restoration for High-Speed Tunnel-Lining Crack Inspection

The second manuscript revision is archived as **v2.1.0-round2**, dated 28 November 2026. The new analyses are in `round2/`. The October revision's numerical tables, figures, acquisition protocols and large controlled-input assets remain available for comparison. [Second-revision release](https://github.com/chuligod22-ux/MERIT/releases/tag/v2.1.0-round2) · [October revision](https://github.com/chuligod22-ux/MERIT/releases/tag/v2.0.0-revision)

## 1. Reproduce the second-revision analyses

| Analysis | Public evidence |
|---|---|
| Optical preservation | Matched ROI measurements and code for held-condition gate refits. The controlled fixed-chart comparison includes the common input, twelve scoring arrays, comparator outputs and the metrics used to score them. [Optical evaluation](round2/optics/README.md) |
| Detection and restoration | Paired results for the fixed and selected routes, task-loss ablations, degradation severities and robustness checks. [Detection analysis](round2/detection/README.md) |
| Framework and backbone effects | Confirmation records and statistical contrasts for the factorial comparison. [Factorial analysis](round2/factorial/) |
| Field apparent width and sampling | Anonymous measurements for all 1,057 regions, the 741 paired outcomes, exact span resampling counts and the sampling design for 14 spans. [Apparent-width and sampling analysis](round2/field_width/README.md) |

The round-two analysis guides describe what each verifier reads and which computations it repeats. The field results can be recomputed from the derived numerical tables. Processing the 1,703 original field image/mask pairs requires separate access. Full retraining archives and third-party model weights are outside the package. [Request field inputs](ACCESS_REQUEST.md)

Chart MTF50, PSNR, LPIPS, crack-detection AUPRC, matched-sensitivity false-positive rate and expert-mask-referenced apparent width measure distinct endpoints.

## 2. Repository layout

```text
.
├── round2/               # second-revision numerical evidence and replay scripts
├── src/modules/          # October revision analysis and inference source
├── protocols/            # October revision protocols and hashes
├── data/
│   ├── processed/        # October revision numerical evidence
│   └── manifests/        # file hashes, public scope, and release inventory
├── figures/              # October revision Figures 1–15 and provenance
├── environment/          # environment notes
├── third_party/FETCH.md  # official acquisition instructions; no redistribution
├── tools/                # package assembly and verification
├── DATASET_CARD.md
├── ACCESS_REQUEST.md
├── ANONYMIZATION_LOG.md
├── LICENSE-CODE
└── LICENSE-DATA
```

## 3. October revision inputs and analyses

| Evidence stream | Public material | Scale |
|---|---|---:|
| cam1 controlled optical analysis | row-level results; exact context crops in release assets | 50 conditions, 479 frames, 1,737 ROIs |
| cam2 controlled detector-response comparison | row-level results; fixed crops in release assets | 50 conditions, 737 frames |
| four-tunnel field evaluation | anonymized results and code; pixels and masks by request | 4 tunnels, 14 spans, 1,703 tiles |
| printed-target width analysis | ROI ledger and traces; 50 inputs in release assets | 500 ROIs |
| perturbed-degradation audit | protocols, summaries, and code | 6 families × 3 severities |
| framework/backbone analyses | factorial rows, protocols, and code | CNN, Transformer, and state-space configurations |

## 4. October revision release assets

1. [revision-analysis-v2](https://github.com/chuligod22-ux/MERIT/releases/tag/revision-analysis-v2)
2. [revision-cam2-crops-v2](https://github.com/chuligod22-ux/MERIT/releases/tag/revision-cam2-crops-v2)
3. [revision-cam1-optical-crops-v2](https://github.com/chuligod22-ux/MERIT/releases/tag/revision-cam1-optical-crops-v2)
4. [revision-width-inputs-v2](https://github.com/chuligod22-ux/MERIT/releases/tag/revision-width-inputs-v2)
5. [revision-models-essential-v2](https://github.com/chuligod22-ux/MERIT/releases/tag/revision-models-essential-v2)
6. [revision-samples-v2](https://github.com/chuligod22-ux/MERIT/releases/tag/revision-samples-v2)

Each release asset is accompanied by a manifest and SHA-256 checksums. The
`revision-width-inputs-v2` payload is divided into three ZIP assets to remain below the
GitHub per-asset size limit. The four-tunnel images and masks are intentionally absent
from this list.

## 5. Install and verify

Python 3.10 or later is recommended. A CUDA-capable GPU is required for model inference;
the package-integrity and processed-result checks run on CPU.

```bash
python -m pip install -r requirements.txt
python tools/verify_public_package.py
```

Release maintainers can build and verify the large controlled-input assets outside the
Git tree with `tools/build_release_assets.py` and `tools/verify_release_assets.py`.
The model release contains the six restoration checkpoints and one frozen detector
needed for controlled cam1/cam2 inference; field-study pixels remain request-only.

The October revision assembly script accepts the internal revision root
as an explicit argument and never downloads restricted data:

```bash
python tools/assemble_revision_v2.py --source-root /path/to/revision/root
```

## 6. Restricted four-tunnel data

See [ACCESS_REQUEST.md](ACCESS_REQUEST.md). Public cohort manifests use anonymous tunnel
labels and request-only placeholders; they contain no tunnel pixels or mask pixels.

## 7. Third-party components

Comparator repositories and pretrained weights are not redistributed. See
[third_party/FETCH.md](third_party/FETCH.md) for official sources, recorded model IDs,
and provenance limitations.

## 8. License and citation

- Code: MIT; see [LICENSE-CODE](LICENSE-CODE).
- Public data and figures: CC-BY 4.0, subject to the exclusions in
  [LICENSE-DATA](LICENSE-DATA).
- Four-tunnel images and expert masks: not covered by the public CC-BY license and
  available only through the request process.

See [CITATION.cff](CITATION.cff) for citation metadata.
