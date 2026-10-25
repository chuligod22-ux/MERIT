# 1. C063 737프레임 복원 방법 전수 비교

## 1.1 목적

R3-3의 기존 12개 검출기 선택 크롭 비교를 cam2 전체 737프레임으로 확장한다. 모든 프레임에서 상단 40% 라이닝 띠 안에 수평·수직으로 중앙 정렬된 고정 `896×1344` 영역을 사용하며 프레임·크롭을 검출기 반응으로 선별하지 않는다. 이 크기는 기존 표 4의 크롭과 같지만 위치 결정은 detector-blind이다. 동일한 결정론적 합성 열화를 각 운용 원본에 적용하고 고전 복원기, 충실도 전용 NAFNet 및 MERIT을 같은 입력에서 대응 비교한다.

## 1.2 통계 단위

737장은 모두 평가하고 이미지별 결과를 공개한다. 다만 동일 촬영 조건과 연속 취득에 중첩되어 있으므로 737개의 독립 표본으로 주장하지 않는다. 주 추론은 C006 직접 개발 5개 조건을 제외한 사전 지정 45개 촬영 조건 셀을 군집 단위로 사용하고, 조건 내부 프레임 중앙값을 계산한다. C058 adapter는 동일 50조건 자산에서 개발됐으므로 이 45조건 결과도 외부 독립 확증이 아니라 노출을 줄인 개발 분석이다. 이미지 단위 분석은 보조 민감도 분석으로만 제공한다.

## 1.3 결과 해석 경계

운용 원본 프레임에도 잔여 열화가 있으므로 source는 깨끗한 물리 참값이 아니다. cam2에 독립 사람 마스크가 없기 때문에 이 실험은 합성 열화로 손실된 외부 검출기 응답을 각 복원 방법이 얼마나 source 쪽으로 되돌리는지 비교한다. 실제 균열 검출 정확도·민감도·위양성률은 외부 4터널 전문가 마스크 분석과 분리한다.

## 1.4 실행

```powershell
<LOCAL_PATH_REDACTED>
<LOCAL_PATH_REDACTED>
<LOCAL_PATH_REDACTED>
```

## 1.5 사전 고정 비교

- degraded input
- Wiener, oracle PSF
- Richardson–Lucy, oracle PSF
- unsupervised Wiener, oracle PSF
- Wiener, wrong/generic PSF
- TV denoise
- NL-means
- unsharp mask
- fidelity-only NAFNet
- MERIT core with the frozen safe adapter
- final v3i routed MERIT

DiffBIR와 SD-x4는 기존 코드에서도 12프레임 전체 battery와 별도 실행된 생성형 사례였다. 현재 8K 전체 코호트에 대응하는 재현 가능한 실행 자산이 없으므로 C063의 주 전수 비교에 포함하지 않으며, 이 사실을 표 caption과 한계에 공개한다.

## 1.6 실행 및 검산 결과

본 실행은 737/737프레임·50/50조건을 완료했고, 사전 지정 45조건·667프레임을 주 집계에 사용했다. 모든 방법과 지표는 finite이며 `status.json` 상태는 `COMPLETE_CLAIM_BOUNDED`다.

최초 `independent_verification.json` v1은 recovery percentage의 CSV round-trip 차이 최대 `2.20×10^-10`에 공통 absolute tolerance `5×10^-12`를 적용해 21/355 검사를 실패시켰다. 원 결과와 v1은 감사용으로 보존한다. `independent_verify_c063_v2.py`는 raw 통계 tolerance를 `5×10^-12`로 유지하고, near-zero detector-response error로 나눈 recovery percentage에만 `5×10^-10` absolute tolerance를 적용한다. 결과값·CI·p값을 바꾸지 않고 `independent_verification_v2.json`에서 355/355 PASS했다.

## 1.7 과학적 해석

정식 해석은 `../d6180_merit_c063_full737_interpretation.md`에 있다. MERIT 두 경로만 mean-response recovery 점추정치가 양수였지만 CI는 0을 포함했고, 한 고영향 조건 제외와 프레임 직접평균에서는 방향이 음수로 바뀌었다. MERIT core는 8개 비교 복원기보다 평균오차가 낮았고 4개 직접 비교가 Holm 보정 후 유의했다. 따라서 C063은 비교 복원기 대비 response distortion·tail-risk 감소를 지지하지만 degraded 대비 보편적 회복이나 검출 정확도 향상을 지지하지 않는다.

해석 그림과 provenance는 `artifacts/full737_battery_v1/c063_interpretation_forest.{png,pdf}` 및 `c063_interpretation_forest_provenance.json`에 있다.
