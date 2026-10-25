# MERIT v2.4 개발·배포 패키지

## 1. 목적

사용자가 제공한 기존 A–D 네 터널 자료로 MERIT의 균열 검출과 자연 균열 apparent-width 측정을 개선한다. 네 터널은 모두 반복 개발 자료이며, seed 0은 cycle 5 경로 선택에 사용하고 seed 1·2는 선택 후 확률적 재현 검증에 사용했다. 이 결과는 신규 시설의 독립 외부 확증으로 해석하지 않는다.

## 2. 최종 배포 구성

| endpoint | 영상 프로파일 | 후처리 | 결정 |
|---|---|---|---|
| ResNet34–U-Net 균열 검출 | `sharp0025` | 보정 확률 100% | cycle 5 채택 |
| VGG16–U-Net 균열 검출 | `inv1250` | raw 25% + 보정 75% | cycle 5 채택 |
| DeepLabV3–ResNet50 균열 검출 | `inv0125` | raw 25% + 보정 75% | cycle 5 채택 |
| 자연 균열 apparent width | `inv1250` | reference-blind safe fallback | v2.4 채택 |

검출 확률 결합은 정답 마스크나 기준 폭을 조회하지 않는다. 폭 측정 fallback은 raw가 `ESTIMABLE`이고 보정 영상만 `NON_ESTIMABLE`일 때 raw 측정값을 사용한다.

## 3. 핵심 결과

### 3.1 균열 검출

평가 범위는 1,703 tiles, 14 physical parents, 4 tunnels, 3 seeds이다. 타일은 독립 표본으로 세지 않고 parent와 tunnel을 계층 통계단위로 사용했다.

| 분석 | 패널 ΔAUPRC | 95% CI | 터널 방향 | matched ΔFPR | 판정 |
|---|---:|---:|---:|---:|---|
| 전체 seed 0–2 | `+0.005965` | `[+0.005183, +0.006761]` | `4/4` 양수 | `−0.0000506` | PASS |
| 보류 seed 1–2 | `+0.005620` | `[+0.004930, +0.006340]` | `4/4` 양수 | `−0.0000670` | PASS |
| 전체 seed, cycle 5 − cycle 3 | `+0.004641` | `[+0.004077, +0.005266]` | `4/4` 우세 | — | PASS |
| 보류 seed, cycle 5 − cycle 3 | `+0.004110` | `[+0.003542, +0.004668]` | `4/4` 우세 | — | PASS |

전체 seed 패널 ΔF1은 `+0.003024`, 95% CI `[+0.002159, +0.003917]`이다. 검출기별 전체 seed ΔAUPRC는 ResNet `+0.000013`, VGG `+0.016295`, DeepLab `+0.001586`이다. 주 개선은 VGG에서 발생했으며 ResNet의 효과는 작다.

### 3.2 자연 균열 apparent width

평가 범위는 1,057 ROI, 382 tiles, 14 parents, 4 tunnels이다.

| 지표 | 결과 | 판정 |
|---|---:|---|
| 계층 Δ절대오차 | `−0.307222 px` | 개선 |
| 95% CI | `[−0.414666, −0.219153]` | 전체 `<0` |
| 터널별 개선 | `4/4` | PASS |
| paired 측정 가능률 | `741/1,057 = 70.104%` | 사전 기준 `≥70%` PASS |
| raw 비추정률 | `29.896%` | 기준 |
| effective 비추정률 | `27.436%` | 감소 |

반환 단위와 주장 명칭은 각각 `pixel`, `expert-mask-compatible natural crack apparent width`이다.

## 4. 사용 방법

검출에서는 원시 영상과 보정 영상을 같은 검출기에 통과시킨 뒤 확률을 결합한다.

```python
from c016_release import detection_candidate, fuse_detection_probabilities

profile_tensor = detection_candidate(inputs, base_model, detector_name)
raw_probability = detector(raw_image)
profile_probability = detector(profile_image)
final_probability = fuse_detection_probabilities(
    raw_probability,
    profile_probability,
    detector_name,
)
```

폭 측정은 `width_candidate()`로 보정 영상을 만들고 `measure_apparent_width()`로 raw·보정 결과와 safe fallback을 함께 얻는다. `route(..., task="crack_detection")`는 검출의 영상 단계만 반환하므로 확률 결합 함수를 이어서 호출해야 한다.

## 5. 재현과 감사

프로젝트 루트에서 다음 명령을 실행한다.

```powershell
$env:UV_CACHE_DIR = (Join-Path (Get-Location) '.uv-cache')
uv run --project '<LOCAL_PATH_REDACTED>' --no-sync pytest -q '05_MERIT_revision/40_수정본/c016_merit_v24_development/scripts/test_c016.py' '05_MERIT_revision/40_수정본/c016_merit_v24_development/scripts/test_c016_release.py' --basetemp '.pytest_tmp_c016_release'
uv run --project '<LOCAL_PATH_REDACTED>' --no-sync python '05_MERIT_revision/40_수정본/c016_merit_v24_development/scripts/c016_audit.py'
```

현재 검증 결과는 단위시험 10/10 PASS, Ruff PASS, Python compile PASS, 진행 결과 102개 hash·schema 감사 PASS이다.

## 6. 주요 산출물

- `protocol/v24_detection_cycle5_protocol_v1.json`: cycle 5 결과 전 동결 계약
- `protocol/v24_detection_cycle5_selection_v1.json`: seed 0 선택 후 seed 1·2 전 동결 경로
- `runs/detection_cycle5/summary.json`: cycle 5 전체·보류 seed 결과
- `runs/c016_release_v24_summary.json`: 최종 endpoint별 배포 결정
- `scripts/c016_release.py`: 배포 인터페이스
- `scripts/c016_audit.py`: protocol·selection·결과·release 무결성 감사

## 7. 주장 경계

v2.4는 제공된 네 터널에서 균열 검출 보조와 전문가 마스크 호환 apparent-width(px) 분석에 사용할 수 있다. seed 1·2는 확률적 보류 반복이며 독립 시설이 아니다. 자연 균열의 traceable physical width(mm), 신규 시설 일반화, 완전 자동 안전 판정은 이 패키지가 입증하지 않는다.
