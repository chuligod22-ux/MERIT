# 1. MERIT C058 optical-fidelity adapter

## 1.1 목적

C058은 동결된 C006 task-ON 복원 영상에 작은 결정론적 안전 어댑터를 적용해 실제 chart MTF50의 비감소 가능성을 개발한다. C006 모델, 체크포인트, C057 MTF50 v2 추정기와 기존 결과는 수정하지 않는다.

어댑터 식은 다음과 같다.

```text
clip(
  RAW
  + alpha * Gaussian(clip(ON - RAW, ±cap_DN), sigma)
  + sharp_gain * (RAW - Gaussian(RAW, 1.0)),
  0,
  1
)
```

`alpha < 0`은 `RAW + |alpha| * (RAW - ON)` 방향의 inverse-residual 경로다. C057 task-ON 평균이 RAW보다 악화된 결과를 반영해 음수 alpha를 사전 명시했다. `alpha = 0`인 raw-only unsharp 후보는 MERIT 잔차의 기여와 고전적 sharpening 효과를 분리하는 ablation이다.

## 1.2 동결된 개발 설계

- 개발 데이터: C057 cohort의 ISO100 10조건, 92 frames, 335 ROI identities
- 효능 endpoint: 80 km/h ISO100의 5개 거리 조건
- 안전 reference: 같은 거리의 60 km/h ISO100 5조건
- reference 처리: 후보와 시드에 관계없이 bit-exact RAW bypass
- 모델 출력: 동결된 task-ON seed 0, 1, 2를 모두 사용
- 선택 통계: 각 조건에서 3개 시드를 평균하며 favorable-seed 선택 금지
- 불확실성: 5개 조건을 cluster로 한 deterministic percentile bootstrap
- 후보: identity, forward/inverse residual, band-limited inverse residual, inverse residual + low unsharp, raw-only unsharp를 합친 24개 고정 grid

개발 PASS에는 평균 restored-minus-RAW absolute MTF50 error `< 0`, condition-bootstrap 95% CI 상한 `< 0`, 5조건 중 최소 4조건 개선, 3개 seed 평균 모두 `<= 0`, valid coverage와 clipping/overshoot/change safety gate, reference exact bypass가 모두 필요하다. 최종 선택 후보는 추가로 `alpha != 0`이어야 한다.

## 1.3 실행 순서

모든 Python 명령은 지정된 tunnelscanning uv 환경에서 실행한다.

```powershell
uv run --project <LOCAL_PATH_REDACTED>
  python 05_MERIT_revision/40_수정본/c058_optical_fidelity_adapter/scripts/c058_development.py create-contract

uv run --project <LOCAL_PATH_REDACTED>
  python 05_MERIT_revision/40_수정본/c058_optical_fidelity_adapter/scripts/c058_development.py check-contract

uv run --project <LOCAL_PATH_REDACTED>
  python 05_MERIT_revision/40_수정본/c058_optical_fidelity_adapter/scripts/c058_development.py smoke

uv run --project <LOCAL_PATH_REDACTED>
  python 05_MERIT_revision/40_수정본/c058_optical_fidelity_adapter/scripts/c058_development.py run-development

uv run --project <LOCAL_PATH_REDACTED>
  python 05_MERIT_revision/40_수정본/c058_optical_fidelity_adapter/scripts/c058_development.py aggregate-development
```

개발 결과가 PASS일 때만 선택 경로를 동결한다.

```powershell
uv run --project <LOCAL_PATH_REDACTED>
  python 05_MERIT_revision/40_수정본/c058_optical_fidelity_adapter/scripts/c058_development.py freeze-selection
```

동결 후 기존 나머지 40조건을 한 번 확인한다. inference는 frame별 atomic progress라 중단 후 같은 명령으로 재개할 수 있다.

```powershell
uv run --project <LOCAL_PATH_REDACTED>
  python 05_MERIT_revision/40_수정본/c058_optical_fidelity_adapter/scripts/c058_development.py confirm-existing-40 `
  --output 05_MERIT_revision/40_수정본/c058_optical_fidelity_adapter/artifacts/confirmation_existing40_v1 `
  --stage inference

uv run --project <LOCAL_PATH_REDACTED>
  python 05_MERIT_revision/40_수정본/c058_optical_fidelity_adapter/scripts/c058_development.py confirm-existing-40 `
  --output 05_MERIT_revision/40_수정본/c058_optical_fidelity_adapter/artifacts/confirmation_existing40_v1 `
  --stage aggregate
```

## 1.4 증거 해석 제한

ISO200/400/800/1600의 40조건은 adapter 후보 선택에는 사용하지 않지만 C057에서 모델 출력이 이미 관찰된 조건이다. 따라서 결과 표기 문구는 항상 **“adapter-level held-out conditions, not independent data; C057 model outputs were previously observed”**를 사용한다. 이 결과는 독립 자료 또는 prospective confirmation으로 주장할 수 없다.

후속 논문 근거에는 동결된 선택 경로의 외부 4개 터널 균열 검출 AUPRC 및 폭 측정 noninferiority가 별도로 필요하다.

## 1.5 테스트

```powershell
uv run --project <LOCAL_PATH_REDACTED>
  python -m pytest -q -p no:cacheprovider `
  05_MERIT_revision/40_수정본/c058_optical_fidelity_adapter/tests
```
