# 1. C060 R2-3 Table 5 재구성

## 1.1 목적

Reviewer 2 Comment 3에 대응해 Table 5를 R1-2의 최종 NAFNet·MPRNet·Restormer 확인 경로와 일치시키고, paired PSNR·AlexNet LPIPS 및 parent-bootstrap 95% 신뢰구간을 보고한다.

## 1.2 산출물

| 파일 | 내용 |
|---|---|
| `scripts/build_table5.py` | C010 NAFNet LPIPS 추가 산출, C039·C049 결과 결합, provenance 검사 |
| `artifacts/table5_final_v1/nafnet_summary.json` | C010 3-seed·3조건 PSNR 재현 및 LPIPS 요약 |
| `artifacts/table5_final_v1/table5_final.json` | 최종 9행 Table 5 SSOT |
| `artifacts/table5_final_v1/table5_final.csv` | 원고 표용 평탄화 결과 |
| `REPORT.md` | 결과·해석·검증 보고서 |

## 1.3 실행

```powershell
$env:UV_CACHE_DIR='.tmp/uv-cache'
uv run --project '<LOCAL_PATH_REDACTED>' --no-sync python `
  '05_MERIT_revision/40_수정본/c060_r2_3_table5/scripts/build_table5.py' all

uv run --project '<LOCAL_PATH_REDACTED>' --no-sync python -m pytest -q `
  --basetemp '.tmp/pytest-c060' `
  '05_MERIT_revision/40_수정본/c060_r2_3_table5/tests'
```

## 1.4 해석 경계

- 양의 ΔPSNR은 동일 backbone baseline보다 operational source reference에 가까움을 뜻한다.
- 음의 ΔLPIPS는 AlexNet feature distance 감소를 뜻한다.
- 세 backbone은 서로 다른 parent-disjoint 확인은행과 backbone별 고정 route를 사용하므로 효과 크기로 backbone 순위를 매기지 않는다.
- 실제 4터널 현장은 registered paired source reference가 없어 PSNR·LPIPS가 N/A다.
- LPIPS는 물리적 진실성이나 fabricated detail 부재의 인증 지표가 아니다.
