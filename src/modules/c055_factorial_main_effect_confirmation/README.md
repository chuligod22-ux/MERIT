# 1. C055 matched factorial main-effect confirmation

C054 보존 결과는 framework 주효과와 framework-ON 상태의 최대 조건부 backbone 범위를 비교했다. C055는 C051과 동일하게 2×4 factorial의 주변평균으로 두 주효과를 맞춰 비교한다. Seed 0은 추정량 개발에만 사용하며, 정책·추정량·gate·코드를 동결한 뒤 seed 1·2를 확인에 사용한다.

## 2. 최종 결과

Freeze `f195ea49…5730e` 뒤 seed 1·2의 32/32 GPU 단위를 완료했다. Framework 주효과는 `+0.091408` CI `[+0.082036,+0.100814]`, 최대 backbone 주효과는 `0.049899`, dominance는 `+0.041509` CI `[+0.033898,+0.049350]`로 6개 gate가 모두 PASS했다. 결과 payload는 `73cf5c63…64607a13`이다. ON-only 최대 조건부 범위를 사용하는 민감도 dominance는 `−0.008390` CI `[-0.019096,+0.002223]`로 불확정이며 제한사항으로 유지한다. 상세는 `../d5970_merit_r1_4_factorial_framework_dominance_confirmation.md`를 따른다.

## 3. R3-3 계층 Supplement

Reviewer 3 Comment 3의 대체 주효능 근거를 추적할 수 있도록 `runs/confirmation_rows.csv` 336행을 14 physical parent와 4 tunnel cluster로 집계했다.

- `runs/r3_3_parent_summary.csv`: 각 parent에서 seed 2개·detector 3개·backbone 4개인 24행 평균
- `runs/r3_3_tunnel_summary.csv`: parent 평균을 tunnel 안에서 평균한 4개 최상위 cluster
- 생성기: `scripts/build_r3_3_supplement.py`
- 검증: 14/14 parent와 4/4 tunnel의 ΔAUPRC 양수, tunnel A--D `+0.121423/+0.027569/+0.198637/+0.018002`

```powershell
uv run python 05_MERIT_revision/40_수정본/c055_factorial_main_effect_confirmation/scripts/build_r3_3_supplement.py
```

1,703개 tile은 반복 관측이며 독립 N으로 사용하지 않는다. Primary uncertainty는 seed·detector 반복을 parent 안에서 평균하고 parent를 tunnel 안에서 재표집한 뒤 tunnel을 동일 가중하는 원 C055의 10,000회 hierarchical bootstrap을 따른다.
