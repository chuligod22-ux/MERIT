from __future__ import annotations

import csv
import json
import re
from pathlib import Path


C060 = Path(__file__).resolve().parents[1]
OUTPUT = C060 / "artifacts" / "table5_final_v1"
MANUSCRIPT = C060.parent / "manuscript_r2_3" / "main.tex"


def test_final_table_contract() -> None:
    result = json.loads((OUTPUT / "table5_final.json").read_text(encoding="utf-8"))
    assert result["status"] == "PASS"
    assert len(result["rows"]) == 9
    assert {row["backbone"] for row in result["rows"]} == {
        "NAFNet",
        "MPRNet",
        "Restormer",
    }
    assert {row["condition"] for row in result["rows"]} == {"low", "mid", "high"}
    assert all(row["delta_psnr_ci95_low_db"] > 0.0 for row in result["rows"])
    by_key = {(row["backbone"], row["condition"]): row for row in result["rows"]}
    assert all(
        row["delta_lpips_alex_ci95_high"] < 0.0
        for row in result["rows"]
        if row["backbone"] in {"MPRNet", "Restormer"}
    )
    assert by_key[("NAFNet", "low")]["delta_lpips_alex_ci95_high"] < 0.0
    assert by_key[("NAFNet", "mid")]["delta_lpips_alex_ci95_low"] > 0.0
    assert by_key[("NAFNet", "high")]["delta_lpips_alex_ci95_low"] > 0.0
    assert "N/A" in result["interpretation"]["actual_field"]


def test_csv_matches_json() -> None:
    result = json.loads((OUTPUT / "table5_final.json").read_text(encoding="utf-8"))
    with (OUTPUT / "table5_final.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(result["rows"])
    assert [(row["backbone"], row["condition"]) for row in rows] == [
        (row["backbone"], row["condition"]) for row in result["rows"]
    ]


def test_nafnet_reproduces_frozen_psnr() -> None:
    summary = json.loads((OUTPUT / "nafnet_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "PASS"
    assert summary["records"] == 2646
    assert summary["parents"] == 68
    assert summary["c010_psnr_reproduction_max_abs_error"] <= 1e-8


def test_manuscript_table_matches_numeric_ssot() -> None:
    result = json.loads((OUTPUT / "table5_final.json").read_text(encoding="utf-8"))
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")

    def signed(value: float) -> str:
        return f"{value:+.6f}"

    for row in result["rows"]:
        psnr = (
            f"${signed(row['delta_psnr_mean_db'])}$ "
            f"[${signed(row['delta_psnr_ci95_low_db'])}$, "
            f"${signed(row['delta_psnr_ci95_high_db'])}$]"
        )
        perceptual = (
            f"${signed(row['delta_lpips_alex_mean'])}$ "
            f"[${signed(row['delta_lpips_alex_ci95_low'])}$, "
            f"${signed(row['delta_lpips_alex_ci95_high'])}$]"
        )
        assert psnr in manuscript
        assert perceptual in manuscript
    assert "no registered paired source-reference image exists" in manuscript
    assert "joint-optimal interior" not in manuscript
    assert "MambaLite & state-space (selective scan)" not in manuscript
    assert "the benefit follows the framework, not the network" not in manuscript
    assert "any future backbone operating under the MERIT framework" not in manuscript
    assert "the network still matters" in manuscript
    assert "universal improvement in any single metric" in manuscript


def test_manuscript_static_structure() -> None:
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    assert not MANUSCRIPT.read_bytes().startswith(b"\xef\xbb\xbf")
    assert "\ufffd" not in manuscript
    assert "<<<<<<<" not in manuscript and ">>>>>>>" not in manuscript
    assert all(line == line.rstrip() for line in manuscript.splitlines())
    assert manuscript.count("{") == manuscript.count("}")

    tokens = re.findall(r"\\(begin|end)\{([^}]+)\}", manuscript)
    stack: list[str] = []
    for action, environment in tokens:
        if action == "begin":
            stack.append(environment)
        else:
            assert stack and stack.pop() == environment
    assert not stack

    labels = re.findall(r"\\label\{([^}]+)\}", manuscript)
    refs = re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", manuscript)
    assert len(labels) == len(set(labels))
    assert set(refs) <= set(labels)

    citations = {
        key.strip()
        for group in re.findall(r"\\cite[tp]?\{([^}]+)\}", manuscript)
        for key in group.split(",")
    }
    bibliography = set(re.findall(r"\\bibitem(?:\[[^]]+\])?\{([^}]+)\}", manuscript))
    assert citations <= bibliography

    figures = re.findall(r"\\includegraphics(?:\[[^]]+\])?\{([^}]+)\}", manuscript)
    assert len(figures) == 14
    for figure in figures:
        assert any((MANUSCRIPT.parent / f"{figure}{suffix}").exists() for suffix in ("", ".pdf", ".png"))
