# MERIT dataset card for v2.1.0-round2

## 1. Purpose

The second-revision measurements, protocols and replay scripts are in `round2/`. The earlier acquisition inputs remain in the linked revision-v2 releases. The numerical field results are public. The four-tunnel images and expert masks require approved access. [Second-revision release](https://github.com/chuligod22-ux/MERIT/releases/tag/v2.1.0-round2)

## 2. Public controlled-acquisition data

### 2.1 Cam1 optical stream

The analysis covers 479 chart frames, 1,737 frozen slanted-edge ROIs, and 50 combinations
of speed, stand-off distance, and ISO. Public release assets contain the exact per-frame
union-context crops used by restoration: the union of all frozen ROIs plus a 64-pixel
context margin. Source-image SHA-256 values and adjusted ROI coordinates preserve lineage.

### 2.2 Cam2 controlled stream

The full-dataset comparison covers 737 frames over the same 50-condition acquisition grid.
Public assets use the fixed 896 × 1344 analysis crop applied by the revised experiment,
rather than the 44.054-GiB full-frame archive.

### 2.3 Printed-width stream

Fifty controlled target images provide 500 nominal-width ROIs. The final ROI ledger, raw
traces, and corresponding input images are public revision assets. These inputs replace
the unrelated cam1 representatives in the superseded v1 release.

### 2.4 Second-revision chart and gate

The release supplies the controlled fixed-chart input, the scored arrays for twelve methods and the archived external-comparator outputs. These materials support direct rescoring of the controlled chart. The optical gate tables and code support refitting the held acquisition conditions from the frozen per-ROI candidate measurements. [Optical evaluation guide](round2/optics/README.md)

## 3. Restricted four-tunnel field data

The field study contains 1,703 image/mask pairs from 14 anonymous spans in four tunnels.
The images and expert masks are not publicly redistributed. Anonymized row-level outputs,
aggregation code, bootstrap settings, and detector-family summaries are public. Source
images and masks may be provided by the corresponding author on reasonable request,
subject to applicable data-use conditions. [Field data access](ACCESS_REQUEST.md)

The public second-revision tables retain the anonymous hierarchy needed to reproduce the reported comparisons. They include the measurements and missingness indicators for all 1,057 selected apparent-width regions, the mask-closing sensitivity results, and the sampling design for each of the 14 spans. The released values allow recomputation of the estimability, localisation and error summaries. Remeasuring the same chords on the field images requires access to the original pixels and masks. [Apparent-width analysis](round2/field_width/README.md)

## 4. Statistical units

- Controlled optical analysis: acquisition condition, with repeated frames within a cell.
- Controlled cam2 comparison: 45 prespecified non-development conditions for primary
  inference, with frames treated as repeated observations.
- Field analysis: tile → physical span → tunnel, with tunnel-aware aggregation.
- Width analysis: ROI with source-aware and tunnel-aware aggregation as specified by each
  frozen protocol.

The primary apparent-width result averages paired region changes within tiles, then tiles within spans, spans within tunnels, and the four tunnels equally. Width-band results use the paired regions within each band. In the post hoc sampling comparison, each span has equal weight within its sampling design.

## 5. Intended and out-of-scope uses

The package supports audit of restoration, optical response, detector response, crack
detection, matched-sensitivity false-positive rate, and apparent-width results. It is not
a safety certification dataset, a population-complete tunnel survey, or a physical crack
width metrology reference. Chart response is not a substitute for natural-crack truth.

## 6. Licensing

Public tracked data and figures use CC-BY 4.0. Four-tunnel images and masks, third-party
datasets, software, and weights are excluded from that license.
