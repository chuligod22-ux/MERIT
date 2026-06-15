# SP03 (MERIT) — Tables (E4)

> 수치 출처: `eval_classical.py` / `eval_stats.py` / `eval_disentangled.py` / `eval_parfusion.py` / `mpd_axes.csv`. held-out cam1 차트 ROI(MTF) + cam2 균열 프레임(검출). 열화 σ2.6/noise2.5 기준.

## Table 1 — Dataset and calibrated degradation model
| 항목 | 값 |
|------|----|
| cam1 차트 (MTF50 GT) | 50 조건 / 3,704 프레임 |
| cam2 균열 라이닝 (검출 GT) | 50 조건 / 737 프레임 |
| 측정 열화 파라미터 | MTF_h/v(거리), σ_motion(속도), σ_n(ISO) — `cam1_analysis.csv` + SP02 |
| 보정 검증 | 열화 연산자가 지정 MTF50 적중 (오차 ≤ 0.002) |
| pseudo-clean 기준 | 60km / 2.5m / ISO100 (운영상 최선) |

## Table 2 — Classical/generative battery vs MERIT (held-out chart, clean MTF50 0.077)
| Method | MTF50 | vs sharp | PSNR (dB) | LPIPS | flat-noise |
|--------|------:|---------:|----------:|------:|-----------:|
| degraded | 0.039* | 50% | 39.0 | 0.111 | 2.5 |
| Wiener (oracle PSF) | 0.074 | 96% | 38.2 | 0.456 | 2.9 |
| Richardson-Lucy (oracle) | 0.075 | 97% | 42.4 | 0.272 | 1.6 |
| unsupervised Wiener (oracle) | 0.017 | 22% | 28.3 | — | 9.6 |
| Wiener (wrong/generic PSF) | 0.027 | 35% | 37.9 | — | 2.7 |
| TV denoise | 0.041 | 53% | 43.5 | 0.253 | 0.3 |
| NL-means | 0.043 | 56% | 43.9 | 0.251 | 0.4 |
| unsharp mask | 0.027 | 35% | 32.3 | 0.391 | 6.1 |
| SD-x4 diffusion | 0.050** | 65% | 33.9 | 0.159 | — |
| DiffBIR v2.1 (diffusion) | 0.038*** | 49% | 38.7 | **0.129** | — |
| **MERIT (ours)** | **0.076** | **98%** | **44.7** | 0.150 | **0.4** |

*degraded는 열화 후 측정값. **SD-x4 확산 MTF는 crop-불안정(0.050~0.080) — 왜곡(저PSNR)이 주 약점.
***DiffBIR v2.1(복원 전용 강 확산 prior, ECCV'24)은 **전 방법 중 최저 LPIPS(0.129=최고 지각)**이나 MTF50은 clean의 49%(≈degraded 수준) — **지각 최적·계측 실패**. 두 확산계열(SD-x4=과추정·불안정, DiffBIR=과평활·미회복)이 서로 다른 방식으로 계측축 실패 → 생성-prior 범주의 측정-비충실 입증 강화(리뷰어 공격 D 차단).
→ **MERIT만 MTF·PSNR·노이즈 모두 최상권 + 경쟁력 있는 LPIPS + PSF 불요.**

## Table 3 — Crack-detection-fitness recovery (held-out cam2, N=12 frames)
| Method | Recovery of lost fitness (mean ± std) | NAFNet>Wiener |
|--------|---:|:---:|
| Wiener (oracle PSF) | +9% ± 10 | — |
| NAFNet-big (fidelity only) | +23% ± 24 | 9/12 |
| **MERIT (task loss, ours)** | **+65% ± 26** | **12/12** |

## Table 4 — Ablation (held-out, single backbone vs explicit disentanglement; matched/fewer params)
| Variant | params | MTF50 (% clean) | Detection (N=12) |
|---------|------:|------:|------:|
| MERIT (single + prompt + task) | 4.95M | **0.076 (98%)** | **+65% ± 26** |
| B: disentangled cascade (noise→optics) | 4.73M | 0.060 (77%) | +49% ± 32 |
| B: disentangled parallel+fusion | 3.93M | 0.059 (76%) | +49% ± 33 |
| (NAFNet-big = no task loss) | 4.95M | 0.083 (107%) | +23% ± 24 |

→ **명시적 물리-분리 아키텍처(2설계) 모두 단일 프롬프트 백본보다 나쁨** = 측정 프롬프트가 노이즈/광학 분해를 암묵 처리(B negative, 조건화 강화). task loss가 검출 +23%→+65%.

## Table 5 — Backbone transferability (동일 프레임워크를 아키텍처가 다른 백본 4종(3패러다임)에 적용; ON=프레임워크 / OFF=plain-L1 대조, 모두 from-scratch 4000스텝 동일 학습)
| Backbone | Paradigm | Framework | params | MTF50 (% clean) | Detection recovery | false-crack max |
|----------|----------|:---------:|------:|:---------------:|:------------------:|:---------------:|
| NAFNet | simplified-attn CNN | **ON** | 4.95M | 74% | **+77%** | 0.114 |
| NAFNet | | OFF (plain L1) | 4.95M | 72% | +5% | 0.109 |
| PlainUNet | pure-conv CNN | **ON** | 4.38M | 76% | **+41%** | 0.106 |
| PlainUNet | | OFF | 4.38M | 68% | +6% | 0.124 |
| RestormerLite | transformer (MDTA) | **ON** | 2.49M | 73% | **+72%** | 0.122 |
| RestormerLite | | OFF | 2.49M | 70% | +5% | 0.107 |
| MambaLite | state-space (selective scan) | **ON** | 1.64M | 79% | **+39%** | 0.109 |
| MambaLite | | OFF | 1.64M | 71% | +6% | 0.106 |

(clean MTF50 0.077; clean false-crack max 0.092; degraded detection = 1% of clean. 단일 영역 검출 프로토콜 — 헤드라인 N=12 통계와 별개. MambaLite=순수 PyTorch selective-scan SSM(커스텀 CUDA 불요, sm_120 동작), 공식 MambaIRv2/MaIR 아님.)
→ **프레임워크 인과(ON≫OFF)**: 검출 회복 ON +39~77% vs OFF +5~6%(~10×, Mamba 포함), MTF도 ON>OFF 일관 → 백본 아닌 **프레임워크**가 이득의 원천. **백본 무관**: CNN 2종 + transformer + **state-space(Mamba)** 모두 ON에서 MTF 73~79%·큰 검출 양수·false-crack ≪0.5(<DiffBIR 0.173) 일관 → **CNN·Transformer·SSM 3패러다임 횡단 전이**(최소 1.64M Mamba가 MTF 79% 최고). (양식 통제용 단기 from-scratch; production NAFNet은 warm-start로 MTF 98%/+65% N=12.)

> **E4 figure 완료 (F1~F8 전체)**: F1 MPD 삼각 / F2 파이프라인 도식 / F3 MTF sweep(σ별) / F4 검출 회복 막대 / F5 정성 비교 패널(crack, PSNR+det 주석) / F6 false-crack 시각화(세그멘터 heat, mean 0.0001) / F7 ablation 막대 / F8 고전 배터리 산점. 생성 스크립트: `make_figures.py`(F1~F4,F7,F8) + `sweep_for_fig.py`(F3 데이터) + `make_qualitative.py`(F5,F6).
