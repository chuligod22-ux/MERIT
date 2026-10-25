from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import c058_development_v2 as dev
from c058_adapter import AdapterCandidate


SAFETY = {
    "minimum_overall_common_complete_case_fraction": 0.95,
    "minimum_each_distance_common_complete_case_fraction": 0.90,
    "maximum_introduced_clipping_fraction": 0.005,
    "maximum_preclip_overshoot_fraction": 0.01,
    "maximum_abs_change_dn": 12.0,
}


def _condition_rows(
    route_specs: list[tuple[str, float, float, float]],
) -> pd.DataFrame:
    rows = []
    for route_id, alpha, sharp_gain, restored_error in route_specs:
        for seed in (0, 1, 2):
            for distance in (2.5, 3.5, 4.5, 5.5, 6.5):
                rows.append(
                    {
                        "route_id": route_id,
                        "alpha": alpha,
                        "sigma": 1.5 if sharp_gain else 0.0,
                        "sharp_gain": sharp_gain,
                        "cap_dn": 4.0,
                        "seed": seed,
                        "condition_cell": f"80km_{distance:.1f}m_ISO100",
                        "distance_m": distance,
                        "abs_error": restored_error,
                        "raw_abs_error": 0.10,
                        "common_complete_case_overall_fraction": 1.0,
                        "common_complete_case_min_distance_fraction": 1.0,
                        "reference_exact": True,
                        "introduced_clipping_fraction": 0.0,
                        "preclip_overshoot_fraction": 0.0,
                        "max_abs_change_dn": 4.0,
                    }
                )
    return pd.DataFrame(rows)


def test_common_complete_case_uses_one_identical_roi_set_for_every_run() -> None:
    rows = []
    run_keys = ["raw_none", "route_0", "route_1", "route_2"]
    for roi_id, distance in enumerate((2.5, 3.5, 4.5, 5.5, 6.5), start=1):
        for run_key in run_keys:
            valid = not (roi_id == 5 and run_key == "route_2")
            rows.append(
                {
                    "frame_id": f"f{roi_id}",
                    "condition_cell": f"80km_{distance:.1f}m_ISO100",
                    "speed_kmh": 80,
                    "distance_m": distance,
                    "iso": 100,
                    "roi_id": roi_id,
                    "orientation": "vertical_edge",
                    "run_key": run_key,
                    "mtf50": 0.1 if valid else np.nan,
                    "valid": valid,
                }
            )
    locked, report = dev.build_common_complete_case(
        pd.DataFrame(rows),
        run_keys,
        minimum_overall_fraction=0.95,
        minimum_each_distance_fraction=0.90,
    )
    assert report["common_complete_case_roi_identities"] == 4
    assert report["gate"] == "FAIL"
    identities = {
        run_key: set(group["roi_id"])
        for run_key, group in locked.groupby("run_key")
    }
    assert len({frozenset(value) for value in identities.values()}) == 1
    assert set(locked["roi_id"]) == {1, 2, 3, 4}


def test_complete_case_rejects_missing_run_and_duplicates() -> None:
    base = pd.DataFrame(
        [
            {
                "frame_id": "f",
                "condition_cell": "c",
                "speed_kmh": 80,
                "distance_m": 2.5,
                "iso": 100,
                "roi_id": 1,
                "orientation": "vertical_edge",
                "run_key": "raw_none",
                "mtf50": 0.1,
                "valid": True,
            }
        ]
    )
    with pytest.raises(dev.ProtocolError, match="Run-key coverage"):
        dev.build_common_complete_case(
            base,
            ["raw_none", "route_0"],
            minimum_overall_fraction=0.95,
            minimum_each_distance_fraction=0.90,
        )
    duplicated = pd.concat([base, base], ignore_index=True)
    with pytest.raises(dev.ProtocolError, match="Duplicate ROI"):
        dev.build_common_complete_case(
            duplicated,
            ["raw_none"],
            minimum_overall_fraction=0.95,
            minimum_each_distance_fraction=0.90,
        )


def test_unsharp_combination_requires_incremental_merit_contribution() -> None:
    conditions = _condition_rows(
        [
            ("pure_inverse", -0.25, 0.0, 0.08),
            ("raw_only", 0.0, 0.025, 0.07),
            ("combo_pass", -0.25, 0.025, 0.05),
            ("combo_fail", -0.125, 0.025, 0.075),
        ]
    )
    summaries = dev.summarize_candidate_metrics(
        conditions,
        bootstrap_count=2000,
        bootstrap_seed=58,
        safety=SAFETY,
    )
    by_route = {row["route_id"]: row for row in summaries}
    assert by_route["pure_inverse"]["incremental_contribution"]["pass"]
    assert not by_route["pure_inverse"]["incremental_contribution"]["applicable"]
    assert by_route["combo_pass"]["incremental_contribution"]["pass"]
    assert by_route["combo_pass"]["incremental_contribution"][
        "paired_condition_bootstrap_95pct_ci"
    ][1] < 0.0
    assert not by_route["combo_fail"]["incremental_contribution"]["pass"]
    assert by_route["combo_fail"]["technical_gate"] == "FAIL"


def _progress_payload(candidate: AdapterCandidate) -> tuple[pd.DataFrame, dict]:
    frame = pd.DataFrame(
        [
            {
                "frame_id": "80km_2.5m_ISO100/frame.png",
                "condition": "80km_2.5m_ISO100",
                "speed_kmh": 80,
                "distance_m": 2.5,
                "iso": 100,
                "frame": "frame.png",
                "roi_id": 1,
                "orientation": "vertical_edge",
                "image_sha256": "image-hash",
            },
            {
                "frame_id": "80km_2.5m_ISO100/frame.png",
                "condition": "80km_2.5m_ISO100",
                "speed_kmh": 80,
                "distance_m": 2.5,
                "iso": 100,
                "frame": "frame.png",
                "roi_id": 2,
                "orientation": "horizontal_edge",
                "image_sha256": "image-hash",
            },
        ]
    )
    rois = [
        {"roi_id": 1, "orientation": "vertical_edge", "mtf50": 0.1, "valid": True},
        {"roi_id": 2, "orientation": "horizontal_edge", "mtf50": 0.2, "valid": True},
    ]
    diagnostics = {
        "exact_identity": False,
        "reference_bypass": False,
        "residual_cap_fraction": 0.0,
        "uncapped_residual_mean_abs_dn": 1.0,
        "capped_residual_mean_abs_dn": 1.0,
        "preclip_overshoot_fraction": 0.0,
        "preclip_below_zero_fraction": 0.0,
        "preclip_above_one_fraction": 0.0,
        "introduced_clipping_fraction": 0.0,
        "introduced_low_clipping_fraction": 0.0,
        "introduced_high_clipping_fraction": 0.0,
        "mean_abs_change_dn": 0.5,
        "max_abs_change_dn": 1.0,
    }
    events = [{"kind": "raw", "route_id": "raw", "seed": "none", "rois": rois}]
    for seed in (0, 1, 2):
        events.append(
            {
                "kind": "adapter",
                "route_id": candidate.route_id,
                "seed": str(seed),
                "candidate": candidate.to_dict(),
                "diagnostics": diagnostics,
                "rois": rois,
            }
        )
    return frame, {
        "schema_version": dev.PROGRESS_SCHEMA,
        "study": "development_iso100",
        "frame_id": frame.iloc[0]["frame_id"],
        "image_sha256": "image-hash",
        "contract_sha256": "contract-hash",
        "selection_freeze_sha256": None,
        "c057_restoration_freeze_sha256": "c057-hash",
        "events": events,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(c057_restoration_freeze_sha256="bad"), "C057-freeze"),
        (lambda payload: payload["events"][1]["candidate"].update(alpha=0.5), "candidate spec"),
        (
            lambda payload: payload["events"][1]["diagnostics"].update(
                introduced_clipping_fraction=1.1
            ),
            "diagnostic fraction",
        ),
        (
            lambda payload: payload["events"][1]["rois"][0].update(
                orientation="horizontal_edge"
            ),
            "ROI identity/orientation",
        ),
        (lambda payload: payload.update(selection_freeze_sha256="unexpected"), "selection-freeze"),
    ],
)
def test_progress_validation_fails_closed_on_tampering(
    tmp_path: Path, mutation, message: str
) -> None:
    candidate = AdapterCandidate("route", -0.25, 0.0, 0.0, 4.0)
    frame, payload = _progress_payload(candidate)
    mutation(payload)
    path = tmp_path / "progress.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(dev.ProtocolError, match=message):
        dev._validate_progress(
            path,
            frame,
            contract_hash="contract-hash",
            candidates=[candidate],
            c057_freeze_hash="c057-hash",
            study="development_iso100",
            selection_hash=None,
        )


def test_progress_validation_accepts_fully_bound_payload(tmp_path: Path) -> None:
    candidate = AdapterCandidate("route", -0.25, 0.0, 0.0, 4.0)
    frame, payload = _progress_payload(candidate)
    path = tmp_path / "progress.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    dev._validate_progress(
        path,
        frame,
        contract_hash="contract-hash",
        candidates=[candidate],
        c057_freeze_hash="c057-hash",
        study="development_iso100",
        selection_hash=None,
    )


def test_confirmation_default_output_is_separate() -> None:
    assert (
        dev.resolve_output_path("confirm-existing-40", None)
        == dev.DEFAULT_CONFIRMATION_OUTPUT
    )
    assert dev.DEFAULT_CONFIRMATION_OUTPUT != dev.DEFAULT_DEVELOPMENT_OUTPUT
    requested = Path("custom")
    assert dev.resolve_output_path("confirm-existing-40", requested) == requested


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_sidecar(path: Path) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(f"{dev.sha256_file(path)}  {path.name}\n", encoding="ascii")


def _synthetic_release(tmp_path: Path, monkeypatch):
    candidates = [
        AdapterCandidate("pure_inverse", -0.25, 0.0, 0.0, 4.0),
        AdapterCandidate("combo", -0.25, 1.5, 0.025, 4.0),
        AdapterCandidate("raw_only", 0.0, 0.0, 0.025, 8.0),
    ]
    contract = {
        "candidate_grid": [candidate.to_dict() for candidate in candidates],
        "bootstrap": {"count": 500, "seed": 58},
        "safety": SAFETY,
    }
    cohort_rows = []
    for speed in (60, 80):
        for distance in (2.5, 3.5, 4.5, 5.5, 6.5):
            cohort_rows.append(
                {
                    "frame_id": f"{speed}km_{distance:.1f}m_ISO100/frame.png",
                    "condition": f"{speed}km_{distance:.1f}m_ISO100",
                    "speed_kmh": speed,
                    "distance_m": distance,
                    "iso": 100,
                    "roi_id": 1,
                    "orientation": "vertical_edge",
                }
            )
    cohort = pd.DataFrame(cohort_rows)

    class FakeD0:
        @staticmethod
        def frame_records(frame):
            return list(frame.groupby("frame_id", sort=True))

    module = SimpleNamespace(D0=FakeD0)
    monkeypatch.setattr(dev, "load_contract", lambda _: (contract, "contract-hash"))
    monkeypatch.setattr(
        dev,
        "load_c057_environment",
        lambda: (module, None, None, cohort, "c057-hash"),
    )
    monkeypatch.setattr(dev, "_development_cohort", lambda frame: frame)
    monkeypatch.setattr(dev, "_validate_progress", lambda *args, **kwargs: None)

    roi_rows = []
    mtf_by_route = {"pure_inverse": 0.08, "combo": 0.09, "raw_only": 0.07}
    for source in cohort.itertuples(index=False):
        raw_value = 0.10 if source.speed_kmh == 60 else 0.06
        common = {
            "frame_id": source.frame_id,
            "condition_cell": source.condition,
            "speed_kmh": source.speed_kmh,
            "distance_m": source.distance_m,
            "iso": source.iso,
            "roi_id": source.roi_id,
            "orientation": source.orientation,
        }
        roi_rows.append(
            {
                **common,
                "route_id": "raw",
                "seed": "none",
                "run_key": "raw_none",
                "mtf50": raw_value,
                "valid": True,
            }
        )
        for seed in (0, 1, 2):
            for candidate in candidates:
                value = 0.10 if source.speed_kmh == 60 else mtf_by_route[candidate.route_id]
                roi_rows.append(
                    {
                        **common,
                        "route_id": candidate.route_id,
                        "seed": seed,
                        "run_key": f"{candidate.route_id}_{seed}",
                        "mtf50": value,
                        "valid": True,
                    }
                )
    roi = pd.DataFrame(roi_rows)
    expected_keys = ["raw_none"] + [
        f"{candidate.route_id}_{seed}" for seed in (0, 1, 2) for candidate in candidates
    ]
    complete, coverage = dev.build_common_complete_case(
        roi,
        expected_keys,
        minimum_overall_fraction=0.95,
        minimum_each_distance_fraction=0.90,
    )
    diagnostics_rows = []
    for source in cohort.itertuples(index=False):
        for seed in (0, 1, 2):
            for candidate in candidates:
                diagnostics_rows.append(
                    {
                        "frame_id": source.frame_id,
                        "condition_cell": source.condition,
                        "speed_kmh": source.speed_kmh,
                        "distance_m": source.distance_m,
                        "iso": source.iso,
                        "route_id": candidate.route_id,
                        "seed": seed,
                        **candidate.to_dict(),
                        "exact_identity": source.speed_kmh == 60,
                        "reference_bypass": source.speed_kmh == 60,
                        "introduced_clipping_fraction": 0.0,
                        "preclip_overshoot_fraction": 0.0,
                        "max_abs_change_dn": 4.0,
                    }
                )
    diagnostics = pd.DataFrame(diagnostics_rows)
    conditions, references = dev._condition_table(
        complete,
        diagnostics,
        candidates,
        endpoint_filter=lambda frame: (frame["speed_kmh"] == 80)
        & (frame["iso"] == 100),
        coverage=coverage,
    )
    summaries = dev.summarize_candidate_metrics(
        conditions,
        bootstrap_count=500,
        bootstrap_seed=58,
        safety=SAFETY,
    )
    selection = dev.rank_and_select_candidates(summaries)
    selected = next(row for row in summaries if row["route_id"] == selection["selected_route_id"])
    raw_only = next(
        row
        for row in summaries
        if row["alpha"] == 0.0 and row["sharp_gain"] == selected["sharp_gain"]
    ) if selected["sharp_gain"] > 0 else None

    paths = {
        "roi_results": tmp_path / "development_roi_results.csv",
        "common_complete_case_roi_results": tmp_path
        / "development_complete_case_roi_results.csv",
        "pixel_diagnostics": tmp_path / "development_pixel_diagnostics.csv",
        "condition_results": tmp_path / "development_condition_results.csv",
        "candidate_summaries": tmp_path / "development_candidate_summaries.json",
        "common_complete_case_report": tmp_path
        / "development_complete_case_report.json",
        "progress_manifest": tmp_path / "development_progress_manifest.json",
    }
    roi.to_csv(paths["roi_results"], index=False)
    complete.to_csv(paths["common_complete_case_roi_results"], index=False)
    diagnostics.to_csv(paths["pixel_diagnostics"], index=False)
    conditions.to_csv(paths["condition_results"], index=False)
    _write_json(paths["candidate_summaries"], summaries)
    _write_json(paths["common_complete_case_report"], coverage)
    manifest = []
    for index, (_, frame) in enumerate(FakeD0.frame_records(cohort), start=1):
        progress = tmp_path / "progress" / f"{index:04d}.json"
        progress.parent.mkdir(exist_ok=True)
        progress.write_text("{}\n", encoding="utf-8")
        manifest.append(
            {
                "frame_id": str(frame.iloc[0]["frame_id"]),
                "path": str(progress.relative_to(tmp_path)).replace("\\", "/"),
                "sha256": dev.sha256_file(progress),
            }
        )
    _write_json(paths["progress_manifest"], manifest)
    monkeypatch.setattr(
        dev,
        "_collect_rows",
        lambda *args, **kwargs: (
            roi.copy(),
            diagnostics.copy(),
            copy.deepcopy(manifest),
        ),
    )
    final = {
        "schema_version": dev.CONTRACT_SCHEMA,
        "state": "DEVELOPMENT_COMPLETE",
        "gate": selection["gate"],
        "selection": selection,
        "selected_candidate_summary": selected,
        "raw_only_unsharp_ablation": raw_only,
        "common_complete_case": coverage,
        "reference_mtf50_by_distance": {
            f"{distance:.1f}": value for distance, value in sorted(references.items())
        },
        "counts": {
            "frames": 10,
            "roi_identities": 10,
            "common_complete_case_roi_identities": 10,
            "condition_cells": 10,
            "candidate_routes": 3,
            "seeds": 3,
        },
        "inputs": {
            "contract_sha256": "contract-hash",
            "c057_restoration_freeze_sha256": "c057-hash",
        },
        "artifacts": {
            label: {"path": str(path), "sha256": dev.sha256_file(path)}
            for label, path in paths.items()
        },
    }
    final_path = tmp_path / "development_final.json"
    _write_json(final_path, final)
    _write_sidecar(final_path)
    return contract, final, final_path, paths


def _refresh_final_sidecar(final_path: Path) -> None:
    sidecar = final_path.with_suffix(final_path.suffix + ".sha256")
    sidecar.unlink(missing_ok=True)
    _write_sidecar(final_path)


def test_release_verifier_recomputes_complete_case_conditions_and_ranking(
    tmp_path: Path, monkeypatch
) -> None:
    _, _, _, _ = _synthetic_release(tmp_path, monkeypatch)
    _, verified, candidate = dev.verify_development_release(
        tmp_path / "contract.json", tmp_path
    )
    assert verified["gate"] == "PASS"
    assert candidate.route_id == verified["selection"]["selected_route_id"]


def test_release_verifier_rejects_condition_tamper_even_with_refreshed_hashes(
    tmp_path: Path, monkeypatch
) -> None:
    _, final, final_path, paths = _synthetic_release(tmp_path, monkeypatch)
    conditions = pd.read_csv(paths["condition_results"])
    conditions.loc[0, "abs_error"] += 0.01
    conditions.to_csv(paths["condition_results"], index=False)
    final["artifacts"]["condition_results"]["sha256"] = dev.sha256_file(
        paths["condition_results"]
    )
    _write_json(final_path, final)
    _refresh_final_sidecar(final_path)
    with pytest.raises(dev.ProtocolError, match="condition table"):
        dev.verify_development_release(tmp_path / "contract.json", tmp_path)


def test_release_verifier_rejects_roi_tamper_against_progress_recomputation(
    tmp_path: Path, monkeypatch
) -> None:
    _, final, final_path, paths = _synthetic_release(tmp_path, monkeypatch)
    roi = pd.read_csv(paths["roi_results"])
    roi.loc[0, "mtf50"] += 0.01
    roi.to_csv(paths["roi_results"], index=False)
    final["artifacts"]["roi_results"]["sha256"] = dev.sha256_file(
        paths["roi_results"]
    )
    _write_json(final_path, final)
    _refresh_final_sidecar(final_path)
    with pytest.raises(dev.ProtocolError, match="Progress-derived ROI"):
        dev.verify_development_release(tmp_path / "contract.json", tmp_path)


def test_release_verifier_rejects_diagnostics_tamper_against_progress(
    tmp_path: Path, monkeypatch
) -> None:
    _, final, final_path, paths = _synthetic_release(tmp_path, monkeypatch)
    diagnostics = pd.read_csv(paths["pixel_diagnostics"])
    diagnostics.loc[0, "max_abs_change_dn"] += 1.0
    diagnostics.to_csv(paths["pixel_diagnostics"], index=False)
    final["artifacts"]["pixel_diagnostics"]["sha256"] = dev.sha256_file(
        paths["pixel_diagnostics"]
    )
    _write_json(final_path, final)
    _refresh_final_sidecar(final_path)
    with pytest.raises(dev.ProtocolError, match="Progress-derived pixel diagnostics"):
        dev.verify_development_release(tmp_path / "contract.json", tmp_path)


def test_release_verifier_rejects_ranking_tamper_with_valid_sidecar(
    tmp_path: Path, monkeypatch
) -> None:
    _, final, final_path, _ = _synthetic_release(tmp_path, monkeypatch)
    final["selection"]["selected_route_id"] = "raw_only"
    _write_json(final_path, final)
    _refresh_final_sidecar(final_path)
    with pytest.raises(dev.ProtocolError, match="ranking/selection"):
        dev.verify_development_release(tmp_path / "contract.json", tmp_path)


def test_release_verifier_rejects_contract_or_c057_binding_tamper(
    tmp_path: Path, monkeypatch
) -> None:
    _, final, final_path, _ = _synthetic_release(tmp_path, monkeypatch)
    final["inputs"]["c057_restoration_freeze_sha256"] = "wrong"
    _write_json(final_path, final)
    _refresh_final_sidecar(final_path)
    with pytest.raises(dev.ProtocolError, match="contract/C057 bindings"):
        dev.verify_development_release(tmp_path / "contract.json", tmp_path)


def test_release_verifier_rejects_artifact_hash_mismatch(
    tmp_path: Path, monkeypatch
) -> None:
    _, _, _, paths = _synthetic_release(tmp_path, monkeypatch)
    with paths["condition_results"].open("a", encoding="utf-8") as handle:
        handle.write("tamper\n")
    with pytest.raises(dev.ProtocolError, match="artifact hash mismatch"):
        dev.verify_development_release(tmp_path / "contract.json", tmp_path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda final: final["counts"].update(frames=9), "counts"),
        (
            lambda final: final["reference_mtf50_by_distance"].update({"2.5": 0.2}),
            "distance references",
        ),
        (
            lambda final: final["common_complete_case"].update(
                common_complete_case_fraction=0.99
            ),
            "final common complete-case",
        ),
        (
            lambda final: final.update(raw_only_unsharp_ablation=None),
            "raw-only ablation",
        ),
    ],
)
def test_release_verifier_rejects_derived_final_claim_tamper(
    tmp_path: Path, monkeypatch, mutate, message: str
) -> None:
    _, final, final_path, _ = _synthetic_release(tmp_path, monkeypatch)
    mutate(final)
    _write_json(final_path, final)
    _refresh_final_sidecar(final_path)
    with pytest.raises(dev.ProtocolError, match=message):
        dev.verify_development_release(tmp_path / "contract.json", tmp_path)


def test_release_verifier_requires_final_sidecar(tmp_path: Path, monkeypatch) -> None:
    _synthetic_release(tmp_path, monkeypatch)
    final_path = tmp_path / "development_final.json"
    final_path.with_suffix(final_path.suffix + ".sha256").unlink()
    with pytest.raises(dev.ProtocolError, match="Missing frozen file or sidecar"):
        dev.verify_development_release(tmp_path / "contract.json", tmp_path)
