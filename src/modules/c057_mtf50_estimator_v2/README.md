# 1. MERIT C057 MTF50 estimator v2

## 1.1 목적

기존 cam1 50조건 MTF50 측정기의 장거리·고ISO 경계 위치 실패를 개선하고, RAW와 OFF/ON 3-seed 복원 결과를 같은 측정법과 동일 ROI로 재평가한다.

v2는 `sigma=0.75 px` Gaussian을 signed-gradient 경계 위치 추정에만 사용한다. ESF·LSF·MTF50 계산은 원본 픽셀을 그대로 사용하고 실패 ROI에는 값을 채우지 않는다.

## 1.2 RAW 검증

```powershell
uv run --project <LOCAL_PATH_REDACTED>
uv run --project <LOCAL_PATH_REDACTED>
uv run --project <LOCAL_PATH_REDACTED>
```

- 개발 ISO100: 333/335 유효, ROI Spearman 0.93618, PASS
- 공식 holdout ISO200–1600: 1,398/1,402 유효, ROI Spearman 0.98373, frame Spearman 0.98931, PASS
- 실행 동결: `protocol/mtf50_v2_execution_freeze.json`

## 1.3 복원 재평가

```powershell
uv run --project <LOCAL_PATH_REDACTED>
uv run --project <LOCAL_PATH_REDACTED>
uv run --project <LOCAL_PATH_REDACTED>
uv run --project <LOCAL_PATH_REDACTED>
```

`inference`는 479개 frame progress를 원자적으로 저장하며 중단 뒤 같은 명령으로 재개한다. 모든 실행은 복원 동결본, 원본 protocol/cohort, 모델 코드, 6개 checkpoint, RAW holdout PASS의 SHA-256을 먼저 검증한다.

최종 결과는 1,731/1,737 동일 7-run complete case에서 OFF `+0.0033489 cy/px`, ON `+0.0013072 cy/px`로 둘 다 RAW보다 악화했다. 상세는 `../d5990_merit_r2_1_mtf50_v2_result.md`와 `artifacts/restoration_v2/restoration_v2_final.json`에 있다.

공식 독립 검산 생성에는 아래 명령을 사용했다. 공식 결과를 다시 생성하지 않고 동결 자산·CSV·479개 progress 파일을 읽어 핵심 통계와 해시를 재계산한다.

```powershell
uv run --project <LOCAL_PATH_REDACTED>
```

검산 결과는 `artifacts/restoration_v2/restoration_v2_independent_verification.json`에 저장되며 현재 상태는 `PASS`다. 증거 파일의 우발적 덮어쓰기를 막기 위해 같은 경로에 결과가 있으면 검산기는 중단한다.

## 1.4 검증

```powershell
uv run --project <LOCAL_PATH_REDACTED>
```

전체 16개 시험이 통과했다.

## 1.5 주장 한계

본 결과는 회고적 측정법 개발 및 기존 50조건 재평가다. ISO 12233 준수, 현장 광학 교정, 자연 균열 검출 정확도 또는 물리 폭 효능의 증거가 아니다. 기존 v1 86.82% 실패와 음성 효능 결과를 함께 보존한다.
