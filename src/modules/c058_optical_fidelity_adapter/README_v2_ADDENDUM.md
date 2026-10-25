# 1. C058 v2 교정 부록

## 1.1 v1 중단과 보존

C058 v1 preflight는 92개 개발 frame 중 30개에서 중단했다. 정렬 순서상 이 30개는 모두 60 km/h ISO100 reference였고 exact RAW bypass만 실행됐다. 80 km/h ISO100 endpoint 출력은 0개였으므로 효능 결과는 생성되지 않았다.

v1 코드, 계약, status와 30개 progress는 감사용으로 보존한다. 중단 사실과 SHA-256 binding은 `artifacts/development_v1/PREFLIGHT_NO_GO.json` 및 sidecar에 기록했다. v1은 재개하거나 효능 근거로 사용하지 않는다.

## 1.2 v2 핵심 교정

C058 v2는 `scripts/c058_development_v2.py`와 별도 `development_v2` 출력만 사용한다.

- 개발 집계는 RAW와 24개 후보 × 3개 seed를 모두 포함하는 하나의 공통 complete-case ROI 집합을 고정한다.
- 전체 공통 complete-case 비율은 95% 이상, 각 거리별 비율은 90% 이상이어야 한다.
- 기존 40조건 확인은 RAW와 동결된 선택 후보 × 3개 seed의 공통 complete-case만 사용한다.
- progress 검증은 contract·C057·selection hash, 후보 전체 사양, 진단값의 유한성·범위, ROI identity·orientation을 확인한다.
- unsharp가 포함된 MERIT 후보는 같은 gain의 raw-only unsharp보다 평균 absolute-error delta가 작아야 하며, 5개 조건 paired bootstrap 95% CI 상한도 0보다 작아야 한다.
- pure residual 후보에는 위 incremental gate가 적용되지 않으며 N/A PASS로 기록한다.
- selection freeze는 progress에서 ROI와 diagnostics를 다시 만들고 complete-case, condition, summary, ranking, gate, 선택 route, reference, counts, raw-only ablation을 끝까지 재계산한다.
- `development_final.json`과 selection JSON은 각각 SHA-256 sidecar가 있어야 한다.
- `confirm-existing-40`의 기본 출력은 `artifacts/confirmation_existing40_v2`이다.

## 1.3 v2 실행 순서

```powershell
uv run --project <LOCAL_PATH_REDACTED>
  python 05_MERIT_revision/40_수정본/c058_optical_fidelity_adapter/scripts/c058_development_v2.py create-contract

uv run --project <LOCAL_PATH_REDACTED>
  python 05_MERIT_revision/40_수정본/c058_optical_fidelity_adapter/scripts/c058_development_v2.py check-contract

uv run --project <LOCAL_PATH_REDACTED>
  python 05_MERIT_revision/40_수정본/c058_optical_fidelity_adapter/scripts/c058_development_v2.py smoke

uv run --project <LOCAL_PATH_REDACTED>
  python 05_MERIT_revision/40_수정본/c058_optical_fidelity_adapter/scripts/c058_development_v2.py run-development

uv run --project <LOCAL_PATH_REDACTED>
  python 05_MERIT_revision/40_수정본/c058_optical_fidelity_adapter/scripts/c058_development_v2.py aggregate-development

uv run --project <LOCAL_PATH_REDACTED>
  python 05_MERIT_revision/40_수정본/c058_optical_fidelity_adapter/scripts/c058_development_v2.py freeze-selection
```

개발 gate가 PASS이고 release 재계산도 PASS인 경우에만 `selected_route_v2.json`이 생성된다. 이후 `confirm-existing-40`을 실행할 수 있다. 이 40조건은 **adapter-level held-out conditions, not independent data; C057 model outputs were previously observed**로만 기술한다.

## 1.4 v2 검증 범위

```powershell
uv run --project <LOCAL_PATH_REDACTED>
  python -m pytest -q -p no:cacheprovider `
  05_MERIT_revision/40_수정본/c058_optical_fidelity_adapter/tests/test_c058_adapter.py `
  05_MERIT_revision/40_수정본/c058_optical_fidelity_adapter/tests/test_c058_development_v2.py
```

회귀시험은 공통 complete-case 고정, coverage 실패, 후보 및 해시 변조, 진단값 범위, ROI 방향, incremental contribution, progress에서 release까지의 end-to-end 재계산, final sidecar, condition·ranking 변조, confirmation 기본 출력 분리를 포함한다.


## 1.5 v3g 전체 RAW 광학 검증

v3g는 공개 137프레임의 전체 8K RAW에서 로케이터 137/137, ROI 548/548, 활성 평균 Δ절대오차 `-0.005563 cy/px`, 95% CI `[-0.008703,-0.002493]`, 6/6 조건 개선으로 개발 gate를 통과했다. 이후 동결된 280프레임 one-shot 홀드아웃은 SHA 중복 0과 비활성 23조건 exact bypass를 통과했으나 최종 **FAIL**이다.

독립 결과는 정상 프레임 205/280, 공통 ROI coverage 41.96%, 활성 평균 `-0.002242 cy/px`, 95% CI `[-0.007117,+0.003114]`, 조건 개선 3/6이었다. locator 실패 64건, tile-size 미달 11건, 최대 preclip overshoot 9.5266%, introduced clipping 0.9702%가 확인됐다. 따라서 v3g는 실제 광학 개선의 독립 근거로 사용할 수 없다. 상세 판정은 `artifacts/safety_router_v3g/HOLDOUT_INTERPRETATION.md`와 `HOLDOUT_NO_GO.json`을 따른다.

## 1.6 v3h 차트 적격성 교정과 개발 선택

v3h 감사에서 v3g 280장 목록에 사분원 MTF 차트가 아닌 균열판·벽면 프레임이 혼입됐음이 확인됐다. 전체 차트 적격은 8장, 비차트는 259장, 부분 차트는 13장이었다. v3g의 137장 공개 100% coverage도 broad-scale template의 소형 오검출과 border replication 영향을 받아 v3h 증거로 승계하지 않는다.

v3h는 `c058_raw_locator_v2.py`, `c058_mtf_measure_v3h.py`, `c058_inference_v3h.py`, `c058_adapter_v2.py`를 사용한다. 실제 센서 픽셀이 없는 ROI는 측정하지 않고, 작은 crop은 448px까지 반사 패딩하며, 어댑터는 uint8 양자화 이후 신규 0/255 clipping을 만들지 않도록 픽셀별 headroom을 적용한다.

개발자료 84장·활성 공통 ROI 120개에서 task-ON seed 1, `alpha=-0.75`, `sigma=0.75`, cap 8 DN 단일 경로를 선택했다. 평균 Δ절대오차는 `-0.0039258513 cy/px`, 95% CI `[-0.0048862479,-0.0028210066]`, 개선 조건 6/6이며 RAW 평균 절대오차를 68.86% 줄였다. 최대 overshoot와 신규 clipping은 0, 최대 변화는 4 DN이다. 상세 결과는 `artifacts/safety_router_v3h_development/V3H_DEVELOPMENT_REPORT.md`, 선택 동결은 `protocol/selected_safety_router_v3h_development.json`을 따른다. 이 선택은 개발 결과이며 독립 확증이 아니다.

## 1.7 v3h 공개 50조건 전수 감사

`scripts/c058_v3h_full50_audit.py`는 선택 동결을 바꾸지 않고 공개 479장·50조건 전체를 재집계한다. 활성 6조건은 유효 차트 38장·공통 ROI 118개에서 모두 개선됐고, 나머지 44조건·428장은 `raw_exact_bypass`로 입력과 픽셀 단위로 동일하다. 따라서 전수 판정은 6개 개선·44개 동일·0개 악화다. 전체 50조건 평균 Δ절대오차는 `-0.0005115797 cy/px`, 95% CI는 `[-0.0009752507,-0.0001466555]`이며 활성 6조건 평균은 `-0.0042631641 cy/px`다.

상세 전수표는 `artifacts/safety_router_v3h_development/full50_condition_audit/condition_results.csv`, 요약과 보고서는 같은 폴더의 `summary.json`, `FULL50_CONDITION_REPORT.md`에 있다. 우회 조건은 개선으로 계산하지 않고 정확한 동률로 표시하며, 공개 개발자료 결과이므로 독립 확증으로 기술하지 않는다.

## 1.8 v3h 50조건 전수 효능 분석

§1.7의 `6개 개선·44개 동일·0개 악화`는 동결 라우터가 비활성 44조건을 조건 단위로 미리 RAW exact bypass한 **안전성 감사**다. 44조건에서 복원 효능을 측정한 결과가 아니므로 전체 50조건 효능의 근거로 사용하지 않는다.

추가 분석은 같은 v3h 단일 경로를 조건 단위 사전 우회 없이 공개 479장·50조건 모두에 적용했다. 479/479프레임, 50/50조건, paired ROI 1,731/1,737개가 집계됐고 조건 판정은 **32개 개선·4개 동률·14개 악화**였다. 전체 평균 Δ절대오차는 `+0.0000379473 cy/px`, 조건 bootstrap 95% CI는 `[-0.0014045226,+0.0015702932]`로 전체 50조건 평균 개선은 입증되지 않았다.

80 km/h 25조건에서는 20개 개선·2개 동률·3개 악화, 평균 `-0.0022088450 cy/px`, 95% CI `[-0.0035208060,-0.0008161242]`로 개선됐다. 반면 60 km/h 25조건은 12개 개선·2개 동률·11개 악화였고, 거리별 기준을 정의하는 60 km/h·ISO100 5조건은 모두 악화했다. 기준조건의 역할은 개선이 아니라 RAW 보존이므로 후속 경로는 이 5조건을 exact bypass하는 보존 규칙이 필요하다.

상세 결과는 `artifacts/safety_router_v3h_development/all50_efficacy_v1/ALL50_EFFICACY_REPORT.md`와 `condition_results.csv`에 있다. 독립 재계산은 50개 고유 조건, 479개 프레임, paired ROI 1,731개, 분류 합계와 전체 평균·CI를 원시 CSV에서 재현해 `independent_recalculation.json`의 모든 검사를 통과했다. C058 전체 시험은 97/97 PASS했다.

## 1.9 v3i RAW 광학 여유 게이트

v3h 무우회 전수 분석의 악화는 RAW MTF50이 이미 거리별 기준에 가깝거나 기준보다 높은 조건에서 복원이 기준을 넘어간 데서 발생했다. v3i는 모델 구조와 v3h 어댑터를 유지하고, 거리별 60 km/h·ISO100 RAW 기준에서 현재 조건 RAW MTF50을 뺀 광학 여유가 `0.001 cy/px`보다 클 때만 v3h를 활성화한다. 기준조건과 나머지 조건은 모델을 호출하지 않고 byte-exact RAW를 반환한다.

479장·50조건 전수 개발 재평가 결과는 32개 개선·18개 동률·0개 악화다. 전체 평균 Δ절대오차는 `-0.0021273594 cy/px`, 95% CI는 `[-0.0027722948,-0.0015091932]`다. 기준 5조건은 5/5 exact preservation이고, 80 km/h는 20개 개선·5개 동률·0개 악화, 평균 `-0.0028337062`, CI `[-0.0037458677,-0.0019392213]`다. 신규 clipping은 0이다.

보고서와 전수표는 `artifacts/safety_router_v3i_development/all50_margin_gated_v1/`, 실행 동결은 `protocol/v3i_margin_router_development_freeze_v1.json`에 있다. 별도 재계산기는 모든 목표 gate를 원시 CSV에서 재현했고 C058 전체 시험은 102/102 PASS했다. threshold는 v3h 50조건 결과 열람 뒤 정했으므로 v3i는 개발 결과이며 같은 50조건을 새로운 미노출 확증으로 부르지 않는다.
