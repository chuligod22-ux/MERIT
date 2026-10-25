# MERIT v2.3 field package

## 1. 상태

C012의 detector-conditioned crack-detection profile과 printed nominal-width profile은 동결된 내부 holdout gate를 통과했다. 신규 외부 현장에 대한 prospective confirmation과 자연 균열 physical-width 검증은 아직 완료되지 않았다.

현재 논문 경로에서는 사용자가 이미 제공한 4터널을 추가 촬영 없이 C014 retrospective 분석에 사용했다. C/D 검출 holdout은 PASS했고 자연 균열 expert-mask apparent-width는 측정 가능 ROI에서 개선됐으나 paired coverage gate는 실패했다. 상세 결과는 `../d5860_merit_v23_existing_four_tunnel_field_analysis.md`에 있다.

## 2. 주요 파일

- `scripts/c012_infer.py`: 단일 이미지 추론 CLI
- `scripts/c012_profiles.py`: 동결된 작업·검출기 조건부 프로필
- `protocol/holdout_freeze_v1.json`: holdout 전 동결 계약
- `runs/development_v2/summary.json`: 최종 개발 결과
- `runs/holdout_v1/summary.json`: 내부 holdout 결과
- `runs/c012_audit.json`: 무결성 감사 결과
- `../d5840_merit_v23_internal_holdout.md`: 상세 보고서

## 3. 빠른 실행

```powershell
$python = "<LOCAL_PATH_REDACTED>"
$runner = "05_MERIT_revision/40_수정본/c012_field_model/scripts/c012_infer.py"
& $python -X utf8 $runner --input input.png --output output.png --task printed_width
```

균열 검출은 `--task crack_detection`과 지원 detector ID를 함께 지정한다. 아무 보정도 적용하지 않는 안전 경로는 `--task identity`다.
