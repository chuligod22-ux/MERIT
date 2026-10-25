# MERIT — Measurement-Fidelity Restoration for High-Speed Tunnel-Lining Crack Inspection

This repository is the revision reproducibility package for MERIT. It replaces the
superseded v1 public package with the analyses, protocols, and evidence used in the
revised manuscript.

## Reproducibility scope

The public package uses a mixed-access design.

- **Public R1 statistical reproduction:** anonymized row-level results, frozen
  protocols, analysis code, and provenance support recalculation of the reported
  tables, figures, confidence intervals, and sensitivity analyses.
- **Public R2 inference reproduction for controlled acquisitions:** release assets
  provide the exact cam1 optical context crops, cam2 fixed analysis crops, printed-width
  inputs, and essential author-trained checkpoints used by the revised analyses.
- **Request-based R2 reproduction for the four-tunnel field study:** the 1,703 image
  tiles and expert masks are not publicly redistributed. They are available from the
  corresponding author on reasonable request, subject to the applicable data-use
  conditions. Their anonymized result tables and analysis code are public.
- **Full retraining is outside this release:** unused indoor data, the full detector
  training archive, the full cousin-tunnel archive, intermediate checkpoints, failed
  development candidates, caches, and third-party weights are excluded.

The controlled and field endpoints answer different questions. Chart MTF50, PSNR,
LPIPS, crack-detection AUPRC, matched-sensitivity false-positive rate, and
expert-mask-referenced apparent width should not be combined into a single score.

## Repository layout

```text
.
├── src/modules/          # revision analysis and inference source
├── protocols/            # frozen JSON/CSV protocols and hashes
├── data/
│   ├── processed/        # row-level public evidence; no field pixels or masks
│   └── manifests/        # file hashes, public scope, and release inventory
├── figures/              # final Figures 1–15 and provenance
├── environment/          # environment notes
├── third_party/FETCH.md  # official acquisition instructions; no redistribution
├── tools/                # package assembly and verification
├── DATASET_CARD.md
├── ACCESS_REQUEST.md
├── ANONYMIZATION_LOG.md
├── LICENSE-CODE
└── LICENSE-DATA
```

## Key revision evidence

| Evidence stream | Public material | Scale |
|---|---|---:|
| cam1 controlled optical analysis | row-level results; exact context crops in release assets | 50 conditions, 479 frames, 1,737 ROIs |
| cam2 controlled detector-response comparison | row-level results; fixed crops in release assets | 50 conditions, 737 frames |
| four-tunnel field evaluation | anonymized results and code; pixels and masks by request | 4 tunnels, 14 spans, 1,703 tiles |
| printed-target width analysis | ROI ledger and traces; 50 inputs in release assets | 500 ROIs |
| perturbed-degradation audit | protocols, summaries, and code | 6 families × 3 severities |
| framework/backbone analyses | factorial rows, protocols, and code | CNN, Transformer, and state-space configurations |

## Revision-v2 release set

1. `revision-analysis-v2`
2. `revision-cam2-crops-v2`
3. `revision-cam1-optical-crops-v2`
4. `revision-width-inputs-v2`
5. `revision-models-essential-v2`
6. `revision-samples-v2`

Each release asset is accompanied by a manifest and SHA-256 checksums. The
`revision-width-inputs-v2` payload is divided into three ZIP assets to remain below the
GitHub per-asset size limit. The four-tunnel images and masks are intentionally absent
from this list.

## Install and verify

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

The assembly script is retained for auditability. It accepts the internal revision root
as an explicit argument and never downloads restricted data:

```bash
python tools/assemble_revision_v2.py --source-root /path/to/revision/root
```

## Restricted four-tunnel data

See [ACCESS_REQUEST.md](ACCESS_REQUEST.md). Public cohort manifests use anonymous tunnel
labels and request-only placeholders; they contain no tunnel pixels or mask pixels.

## Third-party components

Comparator repositories and pretrained weights are not redistributed. See
[third_party/FETCH.md](third_party/FETCH.md) for official sources, recorded model IDs,
and provenance limitations.

## License and citation

- Code: MIT; see [LICENSE-CODE](LICENSE-CODE).
- Public data and figures: CC-BY 4.0, subject to the exclusions in
  [LICENSE-DATA](LICENSE-DATA).
- Four-tunnel images and expert masks: not covered by the public CC-BY license and
  available only through the request process.

See [CITATION.cff](CITATION.cff) for citation metadata.
