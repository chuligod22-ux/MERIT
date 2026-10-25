# 1. C050 framework–backbone matched factorial

## 1.1 목적

C040의 동일한 302개 물리 부모·450개 크롭·low/mid/high 열화 입력에서 NAFNet, MPRNet, Restormer의 base와 최종 MERIT 경로를 3개 model seed로 비교한다.

## 1.2 판정

물리 부모 paired bootstrap으로 MERIT 주효과와 백본 최대 주변효과를 비교한다. 모든 백본×열화 셀의 MERIT ΔPSNR CI 하한이 양수이고, `MERIT 주효과 − 최대 백본 효과` CI 하한이 양수이며 framework/backbone factor SS ratio CI 하한이 1보다 클 때만 제한된 강한 주장을 허용한다.

## 1.3 해석 경계

Restormer C040 결과는 C050 분석 동결 전에 이미 공개됐으므로 이 실험은 retrospective matched factorial이다. 세 복원 백본과 통제 합성 열화 범위를 넘어 백본 교환 가능성, 모든 구조에 대한 우위, 실제 현장 검출 효능 또는 자연 균열 물리폭 정확도를 주장하지 않는다.

## 1.4 결과

9개 GPU cell과 12,150 records를 완료했다. MERIT framework 주효과는 `+0.168158 dB`였지만 최대 backbone 주변대비가 `0.293827 dB`였고, dominance margin은 `−0.125668 dB`로 strong gate가 FAIL했다. NAFNet-mid와 MPRNet-high도 음수였다. 독립 감사는 9개 cell·bootstrap 재계산·C049 Restormer 재현을 최대 차이 `0.0`으로 확인했다.

기존 Table 5 detection-recovery의 ON−OFF 평균 `51.75%p`가 backbone 최대 주변대비 `18.5%p`보다 크다는 점추정 패턴은 남지만, 단일 detector-region의 cell당 한 값이라 반복 CI가 없다. 따라서 descriptive contrast로만 사용한다.
