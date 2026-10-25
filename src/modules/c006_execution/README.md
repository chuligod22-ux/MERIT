# MERIT C006 실행 패키지

## 1. 목적

G09에서 승인한 zero-prompt task-loss OFF/ON ×3 matched 실험을 실행한다. D0 bank 선정에는 모델 출력을 사용하지 않으며 D3 결과는 6개 final checkpoint와 운영 E2E gate가 모두 닫힌 뒤 한 번만 계산한다.

## 2. 동결 자산

- protocol: `c006_protocol.json`
- bank: `artifacts/d0_bank_448.npy`
- 행별 계보: `artifacts/d0_bank_manifest.csv`
- bank 원장: `artifacts/d0_bank_ledger.json`
- 상세 기준: `../d5510_merit_c006_d0_bank_protocol_freeze.md`

`crack/wall role`은 pixel-level GT가 아니라 취득 조건·고정 공간 규칙의 bank 역할이다. 이를 검출 정확도 평가에 사용하지 않는다.

## 3. 검증·실행 명령

모든 Python 명령은 저장소 루트에서 `uv run`으로 실행한다.

```powershell
$env:UV_CACHE_DIR='<LOCAL_PATH_REDACTED>'
$env:PYTHONDONTWRITEBYTECODE='1'
uv run --project . --no-sync python -m pytest -q 05_MERIT_revision\40_수정본\c006_execution\tests\test_c006_protocol.py
uv run --project <LOCAL_PATH_REDACTED>
```

smoke가 통과한 뒤에만 다음 6개를 실행한다.

```powershell
uv run --project <LOCAL_PATH_REDACTED>
uv run --project <LOCAL_PATH_REDACTED>
uv run --project <LOCAL_PATH_REDACTED>
uv run --project <LOCAL_PATH_REDACTED>
uv run --project <LOCAL_PATH_REDACTED>
uv run --project <LOCAL_PATH_REDACTED>
```

각 run은 기존 `final.pt`를 기본적으로 덮어쓰지 않는다. 오류는 해당 run 폴더의 `failure.json`에 보존한다.

작업공간 `.venv`는 CPU-only PyTorch이므로 GPU smoke·본학습에는 사용하지 않는다. 위 명령은 기존 MERIT CUDA 환경을 읽기 전용 `--no-sync`로 사용하며 외부 저장소에 파일을 만들지 않는다.

## 4. 현재 실행 결과

- 2026-09-19에 task-loss OFF/ON × seeds `{0,1,2}`의 8,000-step final checkpoint 6개가 모두 완료됐다.
- `c006_e2e_protocol.json`과 `c006_operational_inference.py`에 8K full-frame 복원·D2 detector 추론 규약을 고정했다.
- `artifacts/g09_e2e_operational_v4/g09_e2e_aggregate.json`의 보수적 6,916-map 투영은 **60.5389시간**, 최대 GPU memory 증가는 **22,370 MB**, OOM 없음이다.
- G09의 12.0시간 상한을 초과했으므로 판정은 **FAIL**이고 D3 one-shot은 차단 상태다.
- 상세 실행·처분은 `../d5520_merit_c006_six_run_g09_e2e_result.md`를 따른다.

## 5. 금지사항

- D1·D2·D3·DQ1 또는 legacy `crack_bank.npy`를 restorer bank에 추가하지 않는다.
- DeepCrack score/map으로 crop을 선택·제외하거나 role을 바꾸지 않는다.
- 중간 checkpoint나 seed를 결과에 따라 선택하지 않는다.
- 일부 run만 성공한 상태에서 D3를 평가하지 않는다.
- frozen D3를 smoke·튜닝·threshold 선택에 사용하지 않는다.
