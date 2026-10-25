"""R3-2 prompt-response and manuscript consistency checks."""

from __future__ import annotations

import json
import re
from pathlib import Path


C009 = Path(__file__).resolve().parents[1]
REVISION = C009.parent
SUMMARY = C009 / "evaluation" / "final_summary" / "c009_final_summary_v3.json"
MANUSCRIPT = REVISION / "manuscript_r3_2" / "main.tex"
RESPONSE = REVISION / "d6040_merit_r3_2_final_response.md"


def test_prompt_evidence_matches_reported_values() -> None:
    prompt = json.loads(SUMMARY.read_text(encoding="utf-8"))["prompt"]
    comparisons = prompt["perturbation_comparisons"]
    assert prompt["prompt_reliance_pass"] is False
    assert prompt["dose_response"]["mean"] == 0.39637118408354116
    assert prompt["informative_minus_zero_trained"]["mean"] == 0.07163293703346502
    assert comparisons["correct_minus_zero"]["mean"] == 1.9727718806523844
    assert comparisons["correct_minus_mismatch_0.5"]["mean"] < 0
    assert comparisons["correct_minus_mismatch_0.75"]["mean"] < 0
    assert comparisons["correct_minus_mismatch_1.25"]["mean"] > 0
    assert comparisons["correct_minus_mismatch_1.5"]["mean"] > 0


def test_frozen_prompt_record_count() -> None:
    counts = []
    for seed in (0, 1, 2):
        marker = C009 / "evaluation" / "wp2_prompt" / f"wp1__nafnet__merit_full__seed{seed}" / "RUN_COMPLETE.json"
        payload = json.loads(marker.read_text(encoding="utf-8"))
        output = Path(payload["output_path"])
        counts.append(sum(1 for _ in output.open(encoding="utf-8")))
    assert counts == [4050, 4050, 4050]
    assert sum(counts) == 12150


def test_response_and_manuscript_preserve_interpretive_boundary() -> None:
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    response = RESPONSE.read_text(encoding="utf-8")
    for text in (manuscript, response):
        assert "12,150" in text
        assert "1.97277" in text
        assert "0.07163" in text
        assert "0.39637" in text
        assert "operational per-frame prompt estimator" in text
        normalized = text.replace("−", "-")
        for value in (
            "+1.97277",
            "+0.18668",
            "+0.07023",
            "+0.01254",
            "-0.02079",
            "-0.09981",
            "+0.17078",
            "+0.25525",
        ):
            assert value in normalized
    assert "The prespecified prompt-reliance gate therefore failed" in manuscript
    assert "prompt channel functional sensitivity: **SUPPORTED**" in response
    assert "exact-severity monotone calibration: **NOT SUPPORTED**" in response
    assert "Ground-truth masks, reference widths, tunnel identity, and evaluation outcomes are not inference inputs" in manuscript


def test_manuscript_static_structure() -> None:
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    assert not MANUSCRIPT.read_bytes().startswith(b"\xef\xbb\xbf")
    assert "\ufffd" not in manuscript
    assert "<" * 7 not in manuscript and ">" * 7 not in manuscript
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
        assert any(
            (MANUSCRIPT.parent / f"{figure}{suffix}").exists()
            for suffix in ("", ".pdf", ".png")
        )
