# D2 외부 터널 합성 열화 복원 검증

## 1. 목적

D2의 전문가 픽셀 마스크와 4개 터널 leave-one-tunnel-out 검출기를 이용해 고정 joint blur–noise 합성 열화에서 C006 MERIT task-OFF/ON의 검출 성능 회복을 평가한다.

## 2. 실행 순서

모든 Python 명령은 검증된 CUDA 환경을 재사용한다.

```powershell
$env:UV_CACHE_DIR='<LOCAL_PATH_REDACTED>'
$env:PYTHONPYCACHEPREFIX='<LOCAL_PATH_REDACTED>'
$project='<LOCAL_PATH_REDACTED>'
$root='<LOCAL_PATH_REDACTED>'
$out="$root\artifacts\external_v1"

uv run --project $project --no-sync python -m py_compile "$root\d2_external_synthetic.py"
uv run --project $project --no-sync python -m pytest -q "$root\tests"
uv run --project $project --no-sync python "$root\d2_external_synthetic.py" audit --output-dir $out
uv run --project $project --no-sync python "$root\d2_external_synthetic.py" smoke --output-dir "$root\artifacts\smoke_v1"
uv run --project $project --no-sync python "$root\d2_external_synthetic.py" run --output-dir $out
uv run --project $project --no-sync python "$root\d2_external_synthetic.py" summarize --output-dir $out
uv run --project $project --no-sync python "$root\independent_verify.py" --artifact-dir $out
```

기존 최종 파일이 있으면 덮어쓰지 않는다. 재실행은 새 버전 디렉터리를 사용한다.

## 3. 해석 제한

이 실험은 외부 터널 콘텐츠에 추가한 **합성** 열화의 복원 검증이다. 다른 장비에서 자연 발생한 열화의 실제 복원 또는 hallucination 부재를 검증하지 않는다.

## 4. 최종 결과

- 1,703타일·14개 연속구간·4개 터널, 4 folds × 3 chains 완료.
- `task-ON − degraded` 동등 터널 AUPRC `−0.016640`, 95% CI `[-0.026526, -0.005874]`, 양수 터널 1/4.
- F1@0.5 차이 `−0.028096`, 95% CI `[-0.040274, -0.015935]`, 양수 터널 0/4.
- primary gate와 task-loss increment gate 모두 실패: `NOT_SUPPORTED`.
- 독립 재계산은 모든 집계·CI·gate를 재현해 `PASS`.

해석과 원고 처분은 `../d5650_merit_d2_external_synthetic_result.md`를 따른다.
