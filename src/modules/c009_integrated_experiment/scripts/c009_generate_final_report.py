"""Render the completed frozen C009 summary as an auditable Korean report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import c009_evaluation_common as common


C009 = Path(__file__).resolve().parent.parent
REVISION = C009.parent.parent
DEFAULT_SUMMARY = C009 / "evaluation" / "final_summary" / "c009_final_summary_v3.json"
DEFAULT_OUTPUT = REVISION / "40_수정본" / "d5800_merit_c009_full_experiment_results.md"


def number(value: Any, digits: int = 5) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def ci(record: dict[str, Any], digits: int = 5) -> str:
    return (
        f"{number(record.get('mean'), digits)} "
        f"[{number(record.get('ci95_low'), digits)}, {number(record.get('ci95_high'), digits)}]"
    )


def token(value: bool) -> str:
    return "PASS" if value else "FAIL"


def render(summary: dict[str, Any], summary_path: Path) -> str:
    optical = summary["optical"]
    prompt = summary["prompt"]
    architecture = summary["architecture"]
    ood = summary["ood"]
    crack = summary["external_four_tunnel_crack"]
    width = summary["printed_width"]
    d3 = summary["different_tunnel_same_system"]
    lines = [
        "# d5800 MERIT C009 전체 통합 실험 결과",
        "",
        "## 1. 최종 판정",
        "",
        f"- **GLOBAL VERDICT: {summary['global_verdict']}**",
        f"- 분석 상태: `{summary['status']}`",
        f"- 결과 JSON: `{summary_path}`",
        f"- 결과 JSON SHA-256: `{common.sha256_file(summary_path)}`",
        f"- 사전동결 계약 SHA-256: `{summary['contract_sha256']}`",
        f"- G12 gate SHA-256: `{summary['g12_sha256']}`",
        "",
        "이 판정은 고정 최종 checkpoint, 전체 사전지정 seed·backbone·detector·터널·severity·폭 조건을 사용한 결과다. 일부 양성 subgroup 또는 threshold 결과로 실패 gate를 구제하지 않는다.",
        "",
        "## 2. 핵심 gate",
        "",
        "| Gate | 판정 |",
        "|---|---:|",
        f"| Optical macro | {token(optical['optical_macro_pass'])} |",
        f"| Restoration efficacy floor | {token(optical['restoration_efficacy_floor_pass'])} |",
        f"| Backbone portability | {token(optical['backbone_portability_pass'])} |",
        f"| Identity safety | {token(optical['identity_safety_pass'])} |",
        f"| Oversharpening safety | {token(optical['oversharpening_safety_pass'])} |",
        f"| Prompt reliance | {token(prompt['prompt_reliance_pass'])} |",
        f"| External four-tunnel crack | {token(crack['external_crack_gate_pass'])} |",
        f"| Printed nominal width | {token(width['printed_width_gate_pass'])} |",
        "",
        "## 3. 광학 충실도",
        "",
        "| 지표 | 평균 [95% CI] |",
        "|---|---:|",
    ]
    for name, record in optical["metrics"].items():
        if isinstance(record, dict):
            lines.append(f"| {name} | {ci(record)} |")
        else:
            lines.append(f"| {name} | {number(record)} |")
    lines.extend(["", "### 3.1 Backbone별 ΔPSNR", "", "| Backbone | ΔPSNR [95% CI] |", "|---|---:|"])
    for family, record in optical["backbone_portability"].items():
        lines.append(
            f"| {family} | {ci(record)}; Holm p={number(record['holm_adjusted_one_sided_p'])} |"
        )
    lines.extend(["", "### 3.2 Identity safety", "", "| Backbone | mean change | p99 change | saturation Δ | preclip OOR | 판정 |", "|---|---:|---:|---:|---:|---:|"])
    for family, record in optical["identity_by_family"].items():
        lines.append(
            f"| {family} | {number(record['mean_absolute_output_change'])} | "
            f"{number(record['p99_absolute_output_change'])} | "
            f"{number(record['saturation_fraction_increase'])} | "
            f"{number(record['preclip_out_of_range_fraction'])} | {token(record['pass'])} |"
        )
    lines.extend(["", "## 4. Prompt 의존성", "", f"- Prompt gate: **{token(prompt['prompt_reliance_pass'])}**", f"- Dose-response rho: {ci(prompt['dose_response'])}", f"- Informative vs zero-trained ΔPSNR: {ci(prompt['informative_minus_zero_trained'])}", "", "| 비교 | correct−perturbation ΔPSNR [95% CI] |", "|---|---:|"])
    for name, record in prompt["perturbation_comparisons"].items():
        lines.append(f"| {name} | {ci(record)} |")
    lines.extend(["", "## 5. 구조 비교", "", "| 구조 | ΔPSNR [95% CI] | ΔLPIPS [95% CI] | 판정 |", "|---|---:|---:|---:|"])
    for name, record in architecture.items():
        lines.append(f"| {name} | {ci(record['delta_psnr'])} | {ci(record['delta_lpips'])} | {token(record['pass'])} |")
    lines.extend(["", "## 6. OOD robustness", "", f"- 6-family broad claim: **{token(ood['broad_six_family_claim_pass'])}**", "", "| Family | S1 | S2 | S3 |", "|---|---:|---:|---:|"])
    for family, severities in ood["by_family_severity"].items():
        lines.append(f"| {family} | {token(severities['1']['pass'])} | {token(severities['2']['pass'])} | {token(severities['3']['pass'])} |")
    lines.extend(["", "## 7. 외부 4터널 균열 endpoint", "", f"- Macro ΔAUPRC: {ci(crack['macro'])}", f"- 독립 detector panel gate: **{token(crack['external_crack_gate_pass'])}**", "", "| Detector | ΔAUPRC [95% CI] | positive tunnels | 판정 |", "|---|---:|---:|---:|"])
    for detector, record in crack["detectors"].items():
        lines.append(
            f"| {detector} | {ci(record)}; Holm p={number(record['holm_adjusted_one_sided_p'])} | "
            f"{record['positive_tunnels']}/4 | {token(record['pass'])} |"
        )
    lines.extend(
        [
            "",
            f"- Matched-sensitivity false-positive pixel burden: {ci(crack['matched_sensitivity_false_positive_pixel_burden'])}",
            f"- Matched-sensitivity safety: **{token(crack['matched_sensitivity_safety_pass'])}**",
        ]
    )
    lines.extend(["", "## 8. Printed nominal apparent width", "", f"- 절대오차 Δ: {ci(width['primary_delta_absolute_error'])}", f"- 상대 MAE 감소율: {number(width['relative_mean_absolute_error_reduction'])}", f"- Base 대비 MERIT 절대오차 Δ: {ci(width['increment_over_base_delta_absolute_error'])}", f"- 개선 폭: {width['improved_estimable_nominal_widths']}/{width['required_improved_estimable_nominal_widths']}", f"- restored/degraded non-estimable rate: {number(width['restored_non_estimable_rate'])}/{number(width['degraded_non_estimable_rate'])}", f"- Width gate: **{token(width['printed_width_gate_pass'])}**", "", "## 9. 다른 터널·동일 시스템 974장", "", f"- model당 평가 frame: {d3['frames_per_model']}", f"- 해석 한계: {d3['claim_boundary']}", "", "model별 image-visible detector response는 최종 JSON에 전량 보존한다. clean reference가 없으므로 이 결과를 실제 복원 정확도 또는 물리적 false-positive rate로 표현하지 않는다.", "", "## 10. 필수 보고 한계", "", f"- {summary['mandatory_context']}", "- D0 prompt 결과는 synthetic oracle 범위다.", "- 외부 4터널은 다른 장비 자료이며 controlled synthetic degradation endpoint다.", "- 사촌터널은 다른 터널·동일 촬영시스템이며, 현장타설 콘크리트 밖의 material generalization을 입증하지 않는다.", "- printed chart는 명목 apparent width 보존이며 자연 균열의 물리 폭 정확도가 아니다.", "- 최종 원고·답변서의 효능 문구는 위 GLOBAL VERDICT보다 강하게 쓰지 않는다.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary_path = args.summary.resolve()
    summary = common.load_json(summary_path)
    if summary.get("status") != "PASS_COMPLETE_PREDECLARED_ANALYSIS":
        raise common.EvaluationError("Final summary is not complete")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite final report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(summary, summary_path), encoding="utf-8", newline="\n")
    print(json.dumps({"status": "PASS", "path": str(output), "sha256": common.sha256_file(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
