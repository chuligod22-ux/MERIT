# C018 MERIT v2.5 열화 강건성

## 1. 목적

C009에서 실패한 6개 OOD 열화군×3단계를 개선하고, 개발 표본과 부모 영상이 겹치지 않는 기존 C010 98-crop bank에서 동결 후보를 한 번 확인한다.

## 2. 최종 상태

- 개발 평가: 2,700 records, 33 parents, 18/18 PASS
- 확인 평가: 5,292 records, 68 parents, 18/18 PASS
- 열화 입력 비교 감사: 15/18 실제 입력 개선, 모션 3/18 exact preservation, 혼합·악화 0/18
- 열화군: 조명 경사·비네팅, 선형 모션 PSF, Poisson-Gaussian, 포화·클리핑, 공간가변 블러, 베일링·산란
- exact identity: 평균·p99 절대 변화량 `0.0`
- 확인 freeze: `protocol/confirmatory_freeze_v1.json`
- 확인 결과: `runs/confirmation_v1/RUN_COMPLETE.json`

## 3. 주요 산출물

| 경로 | 역할 |
|---|---|
| `protocol/development_v1.json` | 개발 범위와 과학적 경계 |
| `runs/development_v1/physics_scan.json` | 물리 경로 gain scan |
| `runs/development_v1/poisson_nafnet_seed0/final.pt` | Poisson-Gaussian fine-tuned checkpoint |
| `runs/development_v1/poisson_mild_fusion_scan.json` | 경미한 Poisson 결합 선택 |
| `runs/development_v1/c009_development_evaluation.json` | 최종 개발 평가 |
| `protocol/confirmatory_freeze_v1.json` | 확인 전 code·data·model·gate 동결 |
| `runs/confirmation_v1/RUN_COMPLETE.json` | one-shot 확인 결과 |
| `runs/input_reference_audit_v1/RUN_COMPLETE.json` | 최종 경로 대 열화 입력 사후 비교 감사 |
| `runs/input_reference_audit_v1/cell_summary.csv` | 18개 셀 입력 기준 요약 |
| `runs/figure15_v1/fig15_perturbed_degradation.png` | 18셀 열지도와 6계열 severity-2 정성 비교 |
| `runs/figure15_v1/fig15_perturbed_degradation.pdf` | Figure 15 벡터 원본 |
| `runs/figure15_v1/fig15_provenance.json` | 원본-only 예시 선택·입력·출력 hash 계보 |
| `scripts/c018_audit_input_reference.py` | 동결 확인 출력 재현·입력 비교 감사 |
| `scripts/plot_c018_perturbed_degradation.py` | Figure 15 재생성 및 원고 자산 동기화 |
| `scripts/test_c018.py` | 단위시험 |
| `scripts/test_c018_input_reference_audit.py` | 감사 산출물 회귀시험 |

## 4. 해석 경계

결과는 열화 종류와 파라미터가 주어진 retrospective parent-disjoint synthetic OOD 확인이다. blind degradation estimation, 신규 촬영, actual sensor calibration, cross-site field restoration 또는 physical-width(mm) 정확도 근거가 아니다. 모션 PSF 경로는 deblurring이 아니라 기존 base의 추가 손상을 막는 안전 우회다.

입력 비교 감사는 이미 완료된 동결 확인 출력에 열화 입력 comparator를 추가한 사후 기술 분석이다. 원래 candidate-versus-base 확인 판정을 변경하거나 두 번째 독립 확인으로 해석하지 않는다. 이 감사에서 비모션 15개 셀은 열화 입력보다 PSNR·LPIPS·gradient 방향이 모두 개선됐고, 모션 3개 셀은 열화 입력과 픽셀 단위로 완전히 동일했으며, 혼합 또는 악화 셀은 없었다. 조명·포화·베일링의 큰 PSNR 개선은 알려진 합성 파라미터의 결정론적 역변환 결과이므로 실제 현장 기대 효과로 사용하지 않는다.
