# 1. Acquired-field apparent-width and sampling data

Run `python analyze.py` from this directory with Python 3.8 or later. The script uses the standard library, reads the three CSV files relative to its own location and prints a JSON report. Widths are in image pixels. A negative change in absolute error means a smaller difference from the expert-mask chord.

## 1.1 Region table

`regions.csv` contains one row for each of the 1,057 frozen expert-mask cross-sections. The anonymous tunnel, span, tile and region identifiers preserve the nesting required for the statistical analysis. The table provides numerical measurements derived from the restricted field images and annotations.

| Column | Definition |
|---|---|
| `tunnel`, `span`, `tile`, `region` | Anonymous identifiers. Regions nest in tiles, tiles in spans and spans in tunnels A–D. |
| `reference_width_px` | Length of the contiguous central interval on the original binary expert mask, using interpolated crossings at 0.5 on the frozen chord. |
| `closed_mask_width_px` | Length of the central interval after one binary morphological closing with a 3 × 3 square kernel, evaluated along the same original chord. Blank means the interval could not be measured. |
| `raw_width_px` | Full width at half maximum of the raw image's selected dark peak along the chord. Blank means no usable estimate. |
| `processed_seed0_px`, `processed_seed1_px`, `processed_seed2_px` | Direct processed widths for the three frozen runs before the measurement-preservation fallback. Blank denotes an unusable estimate. |
| `raw_profile_status` | Outcome of the raw estimator, recorded as pass, low SNR or low contrast. |
| `raw_peak_location` | Location of an estimable raw peak along the original chord. Values are the central interval, another on-mask interval, or outside the annotation. Blank means unmeasurable. An on-mask interval elsewhere on the chord can belong to the same annotated structure. |
| `raw_peak_distance_to_central_interval_px` | Along-chord distance from the estimable raw peak to the central mask interval, zero for peaks inside it. |
| `amplitude_dn`, `snr` | Raw dark-peak amplitude on the 8-bit intensity scale and the estimator's dimensionless signal-to-noise ratio. |
| `background_dn`, `mask_contrast_dn` | Background intensity at the chord midpoint and mean background-minus-image intensity within the central mask interval, in 8-bit digital numbers. |

For each seed, retain a usable processed estimate. If it is missing, use the raw estimate when available. Take the mean across usable seeds when at least two are available. The direct estimate applies the same two-seed minimum before fallback. A pair requires both a usable raw estimate and a usable collapsed estimate. The measurement-preservation fallback uses estimability and does not compare estimates with the mask reference.

For each pair, subtract raw absolute error from policy absolute error. For the primary estimate, average the differences within each tile, then average tiles within span, spans within tunnel and the four tunnels equally. Width-band summaries give each paired region equal weight. `bootstrap_span_counts.csv` holds the multiplicities of spans within each of the 10,000 original resamples. Each row is one replicate, with spans resampled independently within each tunnel using the original number of spans. The draw sequence used NumPy's default PCG64 generator, seed 20261113, and tunnel and span order as in the column header. The script reweights the empirical span effects using these stored draws and takes the linearly interpolated 2.5th and 97.5th percentiles. Both the reference width and the hierarchy used for the primary analysis are fixed across replicates.

## 1.2 Width results and interpretation

The script recovers 741 paired measurements from 1,057 regions, including 677 direct pairs. The hierarchical change is −0.3072 px (95% bootstrap interval −0.4147 to −0.2192). In the original ≥5 px band, the mean change across the 174 paired regions among 244 is +0.7486 px. These estimates use different denominators and weights. Paired and unpaired raw SNR medians are 9.26 and 3.33. The conditional width result consequently describes the measurable subset.

Of 567 paired regions originally below 5 px, only two had the raw dark peak in the annotated central interval. A further 511 peaks landed in another on-mask interval and 54 lay off mask. Closing reassigned 542 to widths of at least 5 px, left two below 5 px and produced 23 unmeasurable intervals. These operations assess the sensitivity of image-based apparent widths to the reference geometry. They do not supply a field measurement of physical crack aperture. Image-level remeasurement and alternative mask processing require access to the restricted source images and annotations.

## 1.3 Field detection and sampling

`sampling_spans.csv` gives one row per anonymous span, for 14 spans across four tunnels. Each row contains its sampling design, tile count, positive and total sampled pixels, and mean area under the precision–recall curve (AUPRC) for raw, fixed base, fixed full and selected routes. The four route values were matched on span, seed (1 or 2) and detector (five detectors) before taking the mean of the ten scores in each span. The public counts have the original sampling denominators, including any tile overlap.

The overall four-route comparison averages spans within each tunnel, then gives each tunnel equal weight. The fixed-base, fixed-full and selected changes from raw are +0.00020, −0.00064 and +0.00829 AUPRC. The sampling comparison gives each span equal weight within its sampling design. The two full-grid spans comprise 1,344 tiles. The twelve positive-enriched, overlapping spans comprise 359. Their respective mean raw AUPRCs are 0.3433 and 0.7076. The design groups cover different spans and tunnels, so these contrasts are descriptive.
