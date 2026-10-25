# MERIT C051 프레임워크-백본 지배성 검증

## 1. 목적

C050은 합성 광학 PSNR에서 MERIT의 평균 효과가 양수였지만 백본 주변효과보다 크지 않았음을 보였다. C051은 논문의 실제 검사 목적에 맞춰 **균열 검출 AUPRC에서 MERIT 프레임워크 효과가 복원 백본 선택 효과보다 큰지**를 직접 검정한다.

## 2. 설계

- 복원 백본: NAFNet, PlainUNet, RestormerLite, MambaLite
- 실험 요인: MERIT `on`/`off`
- 반복: paired model seeds 0, 1, 2
- 학습: 동일 clean patch bank, 동일 4,000-step degradation stream, 동일 optimizer schedule
- ON: measured blur/noise prompt + L1/gradient/frequency fidelity loss + frozen segmenter task loss
- OFF: degraded image + plain L1
- 현장 평가: 초기 투고 후 추가된 4개 터널, 14 physical parents, 1,703 fixed tiles
- 검출기: tunnel-held-out ResNet34-UNet, independently trained VGG16-UNet, DeepLabV3-ResNet50
- 주평가변수: pixel AUPRC
- 추론 단위: physical parent; tile은 계산 단위일 뿐 독립 표본으로 세지 않는다.

## 3. 주 판정

각 bootstrap 반복에서 다음을 다시 계산한다.

1. `framework_main = mean(AUPRC_ON - AUPRC_OFF)`
2. `max_backbone = max(abs(backbone marginal mean pairwise contrast))`
3. `dominance = framework_main - max_backbone`

`dominance`의 parent-stratified bootstrap 95% CI 하한이 0보다 클 때만 **검출 효능에서 프레임워크 영향이 백본 선택보다 크다**고 판정한다. C050의 PSNR 결과까지 포함한 모든 지표에 대한 무조건적 지배성 주장은 허용하지 않는다.

## 4. 증거 지위

네 터널은 초기 투고본에 없던 신규 외부 현장 자료다. 다만 수정 개발 과정에서 이미 반복 사용됐으므로 C051은 새로 동결한 matched retraining experiment이며, 결과 미열람 prospective 신규 현장 확증으로 표현하지 않는다.

## 5. 실행 순서

1. `scripts/c051_prepare.py`
2. `scripts/c051_train.py` 또는 `scripts/c051_queue.py`
3. 최초 `scripts/c051_evaluate.py` 평가는 detector 입력 단위 오류로 무효화
4. `scripts/c051_evaluate_v2.py` 또는 `scripts/c051_evaluation_queue_v2.py`
5. `scripts/c051_analyze_v2.py`
6. `scripts/c051_audit_v2.py`

## 6. 최종 결과

최초 평가 v1은 `[0,1]` float 복원 출력이 detector에서 다시 255로 나뉘는 이중 정규화 오류 때문에 전면 무효화했다. 학습 checkpoint와 기존 C007 계열 결과는 영향을 받지 않는다. 복원 출력을 `uint8 [0,255]`로 변환하는 단일 교정만 허용한 v2 addendum `a27386b8…ff85767`을 결과 전에 동결했다.

보정 v2는 24/24 평가 cell, 96/96 fold, 1,008/1,008 factorial row를 완료했다. Framework ON−OFF AUPRC는 `−0.012855` CI `[−0.014423,−0.011132]`, 최대 backbone 주변대비는 `0.123199`, dominance는 `−0.136055` CI `[−0.148244,−0.124571]`였다. 네 backbone 중 PlainUNet만 양수였고 detector 3/3과 tunnel 4/4의 평균 효과가 음수여서 모든 사전동결 gate가 실패했다. 독립 감사 `ef6a97b1…4b27cb5`는 issues 0으로 PASS했다.

최종 claim status는 `FRAMEWORK_EFFECT_NOT_SUPPORTED`다. 상세 처분은 `../d5960_merit_r1_4_downstream_framework_backbone_factorial.md`에 기록했다.
