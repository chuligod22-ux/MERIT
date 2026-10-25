"""R3-1 response and manuscript consistency checks."""

from __future__ import annotations

import json
import re
from pathlib import Path


C018 = Path(__file__).resolve().parents[1]
REVISION = C018.parent
AUDIT = C018 / "runs" / "input_reference_audit_v1" / "RUN_COMPLETE.json"
MANUSCRIPT = REVISION / "manuscript_r3_1" / "main.tex"
RESPONSE = REVISION / "d6030_merit_r3_1_final_response.md"


def test_r3_1_numeric_contract_is_present() -> None:
    result = json.loads(AUDIT.read_text(encoding="utf-8"))
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    response = RESPONSE.read_text(encoding="utf-8")
    assert result["classification_counts"] == {
        "EXACT_SAFE_PRESERVATION": 3,
        "INPUT_REFERENCE_IMPROVEMENT": 15,
    }
    assert "5,292" in manuscript
    assert "68 physical parents" in manuscript
    assert "15" in manuscript and "3" in manuscript
    assert "3.788776" in manuscript
    assert "0.005641" in manuscript

    assert "18/18" in response
    assert "98 crops" in response and "68 physical parents" in response
    assert "3.618613" in response
    assert "0.005641" in response


def test_motion_is_reported_as_preservation_not_deblurring() -> None:
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    response = RESPONSE.read_text(encoding="utf-8")
    assert "exact bypass" in manuscript
    assert "did not remove the motion blur" in manuscript
    assert "motion deblurring efficacy" not in manuscript
    assert "successful motion deblurring" not in manuscript

    assert "known-kernel" in response
    assert "does not establish unknown-kernel estimation" in response
    assert "successful motion deblurring" not in response


def test_manuscript_states_post_confirmation_scope() -> None:
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    assert "post-confirmation descriptive audit" in manuscript
    assert "did not estimate an unknown degradation blindly" in manuscript
    assert "not a second independent confirmation" in manuscript
    assert "15 of 18" in manuscript
    assert "no audited cell worsened" in manuscript


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
        assert any(
            (MANUSCRIPT.parent / f"{figure}{suffix}").exists()
            for suffix in ("", ".pdf", ".png")
        )
