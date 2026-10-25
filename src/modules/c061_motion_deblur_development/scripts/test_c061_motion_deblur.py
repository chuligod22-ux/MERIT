"""C061 motion-route evidence and manuscript consistency tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path


C061 = Path(__file__).resolve().parents[1]
REVISION = C061.parent
CONFIRMATION = C061 / "runs" / "confirmation_v1" / "RUN_COMPLETE.json"
INTEGRATED = C061 / "runs" / "integrated_r1_1_v1" / "RUN_COMPLETE.json"
CELL_SUMMARY = C061 / "runs" / "integrated_r1_1_v1" / "cell_summary.csv"
FREEZE = C061 / "protocol" / "confirmation_freeze_v1.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_confirmation_passes_all_three_known_kernel_conditions() -> None:
    result = load_json(CONFIRMATION)
    freeze = load_json(FREEZE)
    assert result["status"] == "PASS"
    assert result["all_three_severities_pass"] is True
    assert result["crops"] == 98
    assert result["parents"] == 68
    assert result["records"] == 294
    assert result["development_confirmation_parent_overlap"] == 0
    assert result["identity_route"] == {
        "maximum_absolute_change": 0.0,
        "pass": True,
    }
    assert freeze["scientific_boundary"]["operator"].startswith(
        "The true synthetic PSF"
    )
    expected_psnr = [10.0069057042776, 6.702691206287211, 3.6186127328955147]
    expected_lpips = [
        -0.06620040560579475,
        -0.1796850759316893,
        -0.1433127522468567,
    ]
    for severity, (psnr, lpips) in enumerate(
        zip(expected_psnr, expected_lpips), start=1
    ):
        cell = result["cells"][str(severity)]
        assert cell["pass"] is True
        assert cell["summaries"]["delta_psnr_db"]["mean"] == psnr
        assert cell["summaries"]["delta_psnr_db"]["ci95_low"] > 0.0
        assert cell["summaries"]["delta_lpips"]["mean"] == lpips
        assert cell["summaries"]["delta_lpips"]["ci95_high"] < 0.0
        assert cell["summaries"]["delta_gradient"]["ci95_high"] <= 0.0
        assert cell["summaries"]["delta_overshoot"]["ci95_high"] <= 0.01


def test_integrated_r1_1_has_eighteen_improvements_and_no_worse_cell() -> None:
    result = load_json(INTEGRATED)
    assert result["status"] == "PASS"
    assert result["input_reference_improvement_cells"] == 18
    assert result["preservation_cells"] == 0
    assert result["mixed_or_worse_cells"] == 0
    with CELL_SUMMARY.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 18
    assert {row["classification"] for row in rows} == {
        "INPUT_REFERENCE_IMPROVEMENT"
    }
    motion = [row for row in rows if row["condition"] == "linear_motion_psf"]
    assert len(motion) == 3
    assert all(float(row["delta_psnr_vs_input_ci95_low"]) > 0.0 for row in motion)
    assert all(float(row["delta_lpips_vs_input_ci95_high"]) < 0.0 for row in motion)


def test_active_manuscripts_report_result_and_limit_without_old_bypass_claim() -> None:
    files = [
        REVISION / "manuscript_r3_6" / "main.tex",
        REVISION / "manuscript_r3_6" / "main_readable_pre_r2_4.md",
        REVISION / "manuscript_r3_6" / "main_readable_pre_r2_4_ko.md",
        REVISION / "manuscript_r3_6" / "main_reviewer_highlight.tex",
        REVISION / "manuscript_r3_6" / "main_reviewer_highlight.md",
        REVISION / "manuscript_r3_6" / "main_reviewer_highlight_ko.md",
        REVISION / "manuscript_r2_4" / "main.tex",
        REVISION / "manuscript_r2_4" / "main_readable.md",
        REVISION / "manuscript_r2_4" / "main_readable_ko.md",
    ]
    for path in files:
        text = path.read_text(encoding="utf-8")
        if "manuscript_r3_6" in str(path):
            assert "3.619" in text
            assert "10.007" in text
        else:
            assert "3.618613" in text
            assert "10.006906" in text
        assert "Wiener" in text and "Landweber" in text
        assert (
            "blind field deblurring" in text
            or "unknown-kernel field motion" in text
            or "현장 맹목 디블러링" in text
            or "미지 커널 현장 모션" in text
            or "미지 커널의 현장 모션" in text
        )
        assert "zero-gain exact bypass" not in text
        assert "영이득 정확 우회" not in text
        assert "15 of 18" not in text
        assert "15개 셀은 열화 입력보다 개선" not in text


def test_figure15_uses_integrated_metrics_and_current_motion_output() -> None:
    provenance = load_json(
        REVISION
        / "c018_merit_v25_degradation_robustness"
        / "runs"
        / "figure15_v1"
        / "fig15_provenance.json"
    )
    integrated = load_json(INTEGRATED)
    confirmation = load_json(CONFIRMATION)
    assert provenance["cell_summary_sha256"] == integrated["cell_summary_sha256"]
    assert (
        provenance["c061_confirmation_sha256"]
        == integrated["source_c061_confirmation_sha256"]
    )
    assert confirmation["status"] == "PASS"
    for directory in ("manuscript_r3_6", "manuscript_r2_4"):
        assert (REVISION / directory / "fig15.png").is_file()
        assert (REVISION / directory / "fig15.pdf").is_file()
