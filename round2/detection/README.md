# 1. Tunnel detection, objective ablation and degradation severity

These author-produced tables contain detection and fidelity measurements used in the revised MERIT manuscript. The four tunnel labels A–D identify study groups. The 14 anonymous span identifiers link the acquired-image and added-degradation evaluations and the companion width analysis. Field estimates give each observed tunnel equal weight after averaging its spans. All field intervals are conditional on these four tunnels.

## 2. Reproduce the results

Run these commands from this directory. The first uses Python 3.10 or later and the standard library. It recomputes the point estimates and paired contrasts from the score records and compares them with the archived summaries. The second uses NumPy to repeat the original 10,000 bootstrap draws and check every reported lower and upper endpoint in the accompanying detection summaries.

```sh
python verify.py
python verify_intervals.py
```

The scripts locate the CSV files relative to their own directory. A successful run ends with `All archived point estimates passed.` or `All archived bootstrap endpoints passed.`

## 3. Acquired-image and added-degradation routing

`routing_scores.csv` contains 140 acquired-image records (`native`) and 336 added-degradation records (`synthetic`). Each row gives span-level area under the precision–recall curve (AUPRC) for the input bypass (`raw`), common rule (`fixed`) and selected rule (`selected`) on the same seed and detector. The acquired panel uses five detectors, one profile family recorded as `v24`, and confirmation seeds 1 and 2. Its common rule uses a 0.125 residual blend with the fidelity-only NAFNet. The synthetic panel uses four restoration backbones and three detectors. Its common rule gives equal weights to the degraded-input and two restored-input detector probability maps. Its conditioned restorer receives the known added blur and noise strengths.

`routing_summary.csv` reports changes by backbone and detector, by tunnel and across tunnels. The `without_vgg` rows omit the VGG16 U-Net. The overall AUPRC changes for selected and fixed rules are +0.008293 and +0.000201 on acquired inputs, and +0.091408 and +0.088194 on degraded inputs. The accompanying intervals use joint resampling of spans for the three routes within each observed tunnel. They belong to this paired route comparison.

## 4. Training-objective ablation

`ablation_controlled.csv` contains 450 controlled samples from 33 physical source images, scored for three seeds and three arms. `condition` records the low, middle or high in-distribution degradation. `psnr` is peak signal-to-noise ratio in decibels and `lpips` is AlexNet-based perceptual distance. `ablation_field.csv` contains acquired-image AUPRC for five detectors, the same 14 spans, three seeds and four arms. The arms are the full model, zero-prompt model (`no_prompt`), model trained without the task loss (`no_task`), and the acquired-input bypass (`raw`, field only).

`ablation_summary.csv` reports paired contrasts and seed-specific and pooled estimates. Controlled estimates average within physical sources before averaging across sources. Field estimates average within spans and then across tunnels. The seed-matched comparison between the routing and objective experiments uses seeds 1 and 2 and checks all 140 shared raw records. Their common-profile AUPRC values are 0.545143 for the full model, 0.545785 for raw and 0.545986 for the fidelity-only routing model.

## 5. Degradation severity and robustness

| Measurements | Unit and interpretation |
| --- | --- |
| `dose_scores.csv`, `dose_summary.csv` | NAFNet and three CNN detectors over 14 spans, two seeds and three degradation strengths. Horizontal blur, vertical blur and noise are (0.7 px, 0.6 px, 0.75 DN) at `low`, (1.6 px, 1.3 px, 1.5 DN) at `medium`, and (2.6 px, 2.4 px, 2.5 DN) at `table5`. Each summary reports paired AUPRC changes. |
| `span_detail.csv`, `detail_bands.csv`, `baseline_association.csv` | The span's median normalized gradient strength defines its input-detail band. Detail-band estimates give equal weight to represented tunnels. Baseline associations use 14 span means, with both Spearman rank correlation and Pearson correlation after subtracting the corresponding tunnel mean. |
| `margin_conditions.csv`, `margin_summary.csv` | Forty-five primary chart conditions and 667 frames. The margin is in cycles/pixel. Raw and restored values are mean absolute detector-probability errors relative to the source. Each condition has equal weight. Recovery is 100 times one minus the ratio of mean restored error to mean raw error. |
| `backbone_source_means.csv`, `backbone_summary.csv` | Source-level changes in PSNR and LPIPS for three degradation levels. Each backbone uses its own source set. Source IDs are local to each backbone. Severity contrasts pair each source with its low-level estimate. |
| `inversion_scores.csv`, `inversion_summary.csv` | Three optical perturbation families, three severities, four inverse settings and 98 crops from 68 source images. `exact` uses the applied strength, `unknown_medium` uses the middle strength at every severity, and `under20` and `over20` scale the departure from neutral by 0.8 and 1.2. PSNR and LPIPS deltas are restored minus degraded. `plateau` is the fraction of degraded pixels at the upper intensity limit within a tolerance of 0.000001. Estimates average crops within source and then across sources. |

An empty confidence-limit cell denotes an estimate for which the archived analysis reported a point value only. Sample and source identifiers in the controlled, backbone and inversion tables are local to their respective experiments.
