# 1. C061 선형 모션 디블러링 개발

## 1.1 목적과 범위

C018의 선형 모션 PSF exact bypass를 실제 입력 기준 개선 경로로 대체했다. 합성 시 true kernel 길이와 각도를 경로에 제공한 비맹목 복원 시험이며, 실제 터널 영상의 미지 PSF 추정이나 현장 디블러링 효능으로 확대하지 않는다.

## 1.2 개발과 동결 확인

- 개발군: C006 `synthetic_holdout`, 50 crops, 33 physical parents
- 확인군: C010, 98 crops, 68 physical parents
- 개발·확인 parent overlap: 0
- 후보: 강도별 128개 Wiener, Landweber, Richardson--Lucy 및 입력 혼합 조합
- 최종 경로: 3·7 px PSF는 Wiener(`K=0.003`), 13 px PSF는 Landweber(32회, step 1.5)
- 동결: 경로·코드·source bank·통계 gate를 확인 결과 열람 전에 고정

## 1.3 확인 결과

| 모션 강도 | 고정 경로 | ΔPSNR vs input (dB) | 95% CI | ΔLPIPS vs input | 95% CI |
|---|---|---:|---:|---:|---:|
| 1, 3 px | Wiener | +10.006906 | [+9.942788, +10.062365] | -0.066200 | [-0.067171, -0.065253] |
| 2, 7 px | Wiener | +6.702691 | [+6.438496, +6.930634] | -0.179685 | [-0.183141, -0.176116] |
| 3, 13 px | Landweber | +3.618613 | [+3.527309, +3.719351] | -0.143313 | [-0.145932, -0.140727] |

세 강도 모두 PSNR·LPIPS·그래디언트 오차·엣지 오버슈트 gate를 통과했다. 기존 비모션 15개 C018 셀과 결합한 최종 결과는 **18/18 input-reference improvement, 0 mixed/worse**다.

## 1.4 재현 자산

- 개발 동결: `protocol/development_v1a.json`
- 확인 동결: `protocol/confirmation_freeze_v1.json`
- 개발 결과: `runs/development_v1/`
- 확인 결과: `runs/confirmation_v1/`
- R1-1 통합 결과: `runs/integrated_r1_1_v1/`
- 실행·집계 코드: `scripts/`

## 1.5 해석 한계

이 결과는 제공된 합성 PSF에 대한 known-operator recovery를 입증한다. blind kernel estimation, 실제 터널 진동·공간가변 모션의 제거, 센서 보정 현장 회복 및 같은 크기의 현장 이득은 입증하지 않는다. 68-parent 확인군은 이 모션 경로에 대한 parent-disjoint held-out confirmation이며 새로 촬영한 현장 코호트는 아니다.
