# C007 D0 cam1·cam2 50조건 실제 열화 평가

## 1. 목적

기존 초판 데이터의 cam1 `거리 5 × ISO 5 × 속도 2 = 50` 조건을 실제 촬영 열화 입력으로 사용해 C006 zero-prompt task OFF/ON 3-seed의 광학 응답을 평가한다. 합성 열화를 다시 가하지 않는다.

이 분석은 D3 결과를 확인한 뒤 추가된 사후 분석이다. D3 endpoint를 대체하거나 불리한 현장 결과를 구제하는 근거로 사용하지 않는다.

## 2. 고정 코호트와 분석 단위

- 코호트: 결과를 보기 전에 존재한 Imatest `*_Y_multi.csv`와 대응 추출 PNG가 모두 있는 479프레임 전부
- 조건: 50개 전부, 조건당 5–13프레임
- ROI: 각 프레임에서 기존 Imatest가 검출해 저장한 multi-ROI 전부. 자동검출이 4개 미만을 찾은 프레임도 있어 ROI 수는 가변이며, 신규 선별·제외하지 않는다.
- 독립 분석 단위: 50개 조건 셀
- 프레임과 ROI: 조건 내부 반복측정
- primary: development 5셀과 거리별 pseudo-clean reference 5셀의 합집합을 제외한 42셀

기존 `tunnel_v2_cam1.csv` 398프레임은 초판 분석 계보 확인용으로만 보존하고 신규 코호트 선별에는 쓰지 않는다. 원본 Imatest 값은 raw estimator validation에만 사용한다. 복원영상은 동일 좌표에서 `in-house slanted-edge MTF50 surrogate`로 Raw/OFF/ON을 같은 방식으로 다시 측정한다.

## 3. 실행

모든 Python 실행은 저장소 루트에서 `uv run`을 사용한다.

```powershell
$env:UV_CACHE_DIR='<LOCAL_PATH_REDACTED>'
$env:PYTHONDONTWRITEBYTECODE='1'

uv run --project <LOCAL_PATH_REDACTED>
  05_MERIT_revision\40_수정본\c007_d0_condition_eval\d0_cam1_eval.py `
  --protocol 05_MERIT_revision\40_수정본\c007_d0_condition_eval\d0_cam1_protocol.json `
  prepare --output-root 05_MERIT_revision\40_수정본\c007_d0_condition_eval\artifacts\cam1_v1

uv run --project <LOCAL_PATH_REDACTED>
  05_MERIT_revision\40_수정본\c007_d0_condition_eval\d0_cam1_eval.py `
  --protocol 05_MERIT_revision\40_수정본\c007_d0_condition_eval\d0_cam1_protocol.json `
  self-test --output-root 05_MERIT_revision\40_수정본\c007_d0_condition_eval\artifacts\cam1_v1

uv run --project <LOCAL_PATH_REDACTED>
  05_MERIT_revision\40_수정본\c007_d0_condition_eval\d0_cam1_eval.py `
  --protocol 05_MERIT_revision\40_수정본\c007_d0_condition_eval\d0_cam1_protocol.json `
  run --output-root 05_MERIT_revision\40_수정본\c007_d0_condition_eval\artifacts\cam1_v1

uv run --project <LOCAL_PATH_REDACTED>
  05_MERIT_revision\40_수정본\c007_d0_condition_eval\d0_cam1_eval.py `
  --protocol 05_MERIT_revision\40_수정본\c007_d0_condition_eval\d0_cam1_protocol.json `
  aggregate --output-root 05_MERIT_revision\40_수정본\c007_d0_condition_eval\artifacts\cam1_v1
```

## 4. 해석 제한

- MTF 증가는 자동으로 복원 성공이 아니다. 거리별 60 km/h·ISO 100 raw reference와의 절대오차 감소를 본다.
- 398프레임을 독립 `N`으로 세지 않는다.
- 5개 development 조건은 primary에서 제외한다.
- 이 결과는 한 장비·한 통제 취득계열의 조건 강건성에 한정한다.
- cam2 task 결과와 D3 현장 결과를 대신하지 않는다.

## 5. cam2 독립 검출기 반응 감사

cam2에는 독립 사람 mask나 등록된 물리 균열 GT가 없으므로 정확도 평가는 수행하지 않는다. 50개 조건에서 파일명 하위 중앙 프레임을 내용 비의존적으로 1장씩 고정하고, 외부 전문가 라벨의 네 개 다른 터널로 별도 학습한 D2 ResNet34-UNet 반응만 RAW/OFF/ON ×3에서 비교한다.

```powershell
uv run --project <LOCAL_PATH_REDACTED>
  05_MERIT_revision\40_수정본\c007_d0_condition_eval\d0_cam2_response_eval.py build `
  --protocol 05_MERIT_revision\40_수정본\c007_d0_condition_eval\d0_cam2_protocol.json `
  --output-root 05_MERIT_revision\40_수정본\c007_d0_condition_eval\artifacts\cam2_v1

uv run --project <LOCAL_PATH_REDACTED>
  05_MERIT_revision\40_수정본\c007_d0_condition_eval\d0_cam2_response_eval.py self-test `
  --protocol 05_MERIT_revision\40_수정본\c007_d0_condition_eval\d0_cam2_protocol.json `
  --output-root 05_MERIT_revision\40_수정본\c007_d0_condition_eval\artifacts\cam2_v1

uv run --project <LOCAL_PATH_REDACTED>
  05_MERIT_revision\40_수정본\c007_d0_condition_eval\d0_cam2_response_eval.py run `
  --protocol 05_MERIT_revision\40_수정본\c007_d0_condition_eval\d0_cam2_protocol.json `
  --output-root 05_MERIT_revision\40_수정본\c007_d0_condition_eval\artifacts\cam2_v1

uv run --project <LOCAL_PATH_REDACTED>
  05_MERIT_revision\40_수정본\c007_d0_condition_eval\d0_cam2_aggregate.py `
  --protocol 05_MERIT_revision\40_수정본\c007_d0_condition_eval\d0_cam2_protocol.json `
  --output-root 05_MERIT_revision\40_수정본\c007_d0_condition_eval\artifacts\cam2_v1
```

## 6. 2026-09-23 결과

- cam1: 479프레임·1,737 ROI·50조건·42개 1차 조건을 완료했다. RAW surrogate의 Imatest 순위상관은 높았지만 유효 ROI가 86.82%로 사전 gate 95%를 통과하지 못했다. 동일 ROI 1,508개 보조분석에서 OFF는 MTF50 기준오차 +0.003429 cy/px, ON은 +0.001182 cy/px로 모두 악화했다.
- cam2: 50조건·RAW/OFF/ON ×3의 350개 독립 D2 response를 완료했다. 모든 arm-level 기준오차 bootstrap CI가 0을 포함했고 사람 GT가 없어 정확도 향상은 판정할 수 없다.
- 통합 판정: 실제 50조건에서 MTF 복원 또는 균열 검출 향상 주장은 **NO-GO**다. 상세 처분은 `../d5580_merit_d0_cam1_cam2_50condition_result.md`에 기록했다.
