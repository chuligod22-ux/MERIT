# D2 다중 균열 검출기 합성 열화 검증

## 1. 실행 목적

동일한 1,703타일 paired cohort에서 각 검출기 내부의 `task-ON − degraded` 변화를 계산하고, 세 독립 D2 검출기에서 방향이 재현되는지 평가한다. Legacy DeepCrack은 task loss 결합 참고축으로만 보고하며 독립 panel gate에서 제외한다.

## 2. 실행 순서

```powershell
$env:UV_CACHE_DIR='<LOCAL_PATH_REDACTED>'
$project='<LOCAL_PATH_REDACTED>'
$root='<LOCAL_PATH_REDACTED>'
$out="$root\artifacts\multidetector_v1"

uv run --project $project --no-sync python -m py_compile "$root\d2_multidetector.py"
uv run --project $project --no-sync python -m pytest -q "$root\tests"
uv run --project $project --no-sync python "$root\d2_multidetector.py" audit --output-dir $out
uv run --project $project --no-sync python "$root\d2_multidetector.py" smoke --output-dir $out
uv run --project $project --no-sync python "$root\d2_multidetector.py" train --output-dir $out
uv run --project $project --no-sync python "$root\d2_multidetector.py" evaluate --output-dir $out
uv run --project $project --no-sync python "$root\d2_multidetector.py" summarize --output-dir $out
```

기존 최종 파일은 덮어쓰지 않는다. 중단된 개별 학습 run은 불완전 디렉터리를 자동 재사용하지 않으며, 완료된 run만 checkpoint hash를 확인한 뒤 건너뛴다.

## 3. 판정 주의

- 모델 간 절대 AUPRC 차이는 MERIT 효능 근거가 아니다.
- 세 독립 모델의 paired delta와 방향 일관성이 primary다.
- Legacy DeepCrack만 양수이면 일반 효능이 아니라 task-training/evaluation coupling 가능성으로 해석한다.
- 이 분석은 기존 음성 결과 확인 뒤 설계된 post-hoc sensitivity이며 confirmatory validation이 아니다.
