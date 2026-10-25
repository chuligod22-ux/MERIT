from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


BASE = Path(__file__).resolve().parents[1]
SCRIPT = BASE / "scripts" / "c058_safety_router_v3.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("c058_safety_router_v3_tested", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V3 = _load_module()


def test_router_is_active_only_for_frozen_speed_and_iso_strata() -> None:
    assert V3.router_active(80, 100)
    assert V3.router_active(80, 200)
    for speed in (60, 80):
        for iso in (100, 200, 400, 800, 1600):
            assert V3.router_active(speed, iso) is (
                speed == 80 and iso in {100, 200}
            )


def test_inactive_router_is_byte_exact_and_skips_model() -> None:
    raw = np.arange(64, dtype=np.uint8).reshape(8, 8)
    called = False

    def forbidden_model(_: np.ndarray, __: int) -> np.ndarray:
        nonlocal called
        called = True
        raise AssertionError("inactive route must not call the model")

    outputs = V3.route_native_crop(raw, 60, 800, forbidden_model)
    assert not called
    assert set(outputs) == {"raw_none", "raw_exact_bypass_0", "raw_exact_bypass_1", "raw_exact_bypass_2"}
    assert all(np.array_equal(value[0], raw) for value in outputs.values())
    assert all(value[1]["exact_identity"] for value in outputs.values())


def test_active_router_uses_all_three_seeds_and_fixed_candidate() -> None:
    raw = np.full((16, 16), 128, dtype=np.uint8)
    called: list[int] = []

    def model(_: np.ndarray, seed: int) -> np.ndarray:
        called.append(seed)
        return np.full((16, 16), 0.49 + seed * 0.001, dtype=np.float32)

    outputs = V3.route_native_crop(raw, 80, 100, model)
    assert called == [0, 1, 2]
    assert set(outputs) == {
        "raw_none",
        f"{V3.ROUTE_ID}_0",
        f"{V3.ROUTE_ID}_1",
        f"{V3.ROUTE_ID}_2",
    }
    assert all(outputs[f"{V3.ROUTE_ID}_{seed}"][1]["route_id"] == V3.ROUTE_ID for seed in range(3))


def test_native_context_contains_every_canonical_roi_with_fixed_margin() -> None:
    locator = type("Locator", (), {"center_x": 1000.0, "center_y": 500.0, "radius": 300.0})()
    x0, y0, x1, y1 = V3.native_context_bounds((1200, 2000), locator)
    assert (x0, y0, x1, y1) == (685, 185, 1315, 815)


def test_native_mtf_scale_is_explicit() -> None:
    assert V3.canonical_to_native_mtf(0.2, 640.0) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        V3.canonical_to_native_mtf(0.2, 0.0)


def test_bootstrap_is_condition_clustered_and_deterministic() -> None:
    values = np.array([-0.3, -0.2, -0.1, 0.0])
    first = V3.bootstrap_condition_mean_ci(values, count=2000, seed=123)
    second = V3.bootstrap_condition_mean_ci(values, count=2000, seed=123)
    assert first == second
    assert first[1] <= 0.0


def test_inventory_uses_names_only_and_rejects_exposed_or_bank_overlap(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    condition = raw / "80km_2.5m_ISO100"
    condition.mkdir(parents=True)
    for name in ("frame_000001.png", "frame_000002.png"):
        (condition / name).write_bytes(b"not-decoded")
    exposed = pd.DataFrame({"frame_id": ["80km_2.5m_ISO100/frame_000001.png"]})
    rows, audit = V3.discover_holdout_inventory(
        raw,
        exposed,
        bank_frame_ids=set(),
        expected_frames=1,
        expected_conditions=1,
        allowed_conditions={"80km_2.5m_ISO100"},
    )
    assert rows[0]["frame_id"].endswith("frame_000002.png")
    assert "sha256" not in rows[0] and "bytes" not in rows[0]
    assert audit["frame_id_overlap_with_exposed"] == 0
    with pytest.raises(V3.ProtocolError, match="bank"):
        V3.discover_holdout_inventory(
            raw,
            exposed,
            bank_frame_ids={"80km_2.5m_ISO100/frame_000002.png"},
            expected_frames=1,
            expected_conditions=1,
            allowed_conditions={"80km_2.5m_ISO100"},
        )


def test_one_shot_attempt_marker_is_fail_closed(tmp_path: Path) -> None:
    marker = tmp_path / "attempt_started.json"
    assert V3.claim_one_shot(marker, "freeze-hash") == "STARTED"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["execution_freeze_sha256"] == "freeze-hash"
    resumed: list[bool] = []
    assert V3.claim_one_shot(
        marker, "freeze-hash", progress_validator=lambda: resumed.append(True)
    ) == "RESUME"
    assert resumed == [True]
    with pytest.raises(V3.ProtocolError, match="already been claimed"):
        V3.claim_one_shot(marker, "different-freeze", progress_validator=lambda: None)
    (tmp_path / "attempt_complete.json").write_text("{}", encoding="utf-8")
    with pytest.raises(V3.ProtocolError, match="already complete"):
        V3.claim_one_shot(marker, "freeze-hash", progress_validator=lambda: None)


def test_gate_requires_active_seed_direction_and_inactive_identity() -> None:
    condition = pd.DataFrame(
        {
            "condition_cell": ["a", "a", "a", "b", "b", "b"],
            "active": [True, True, True, False, False, False],
            "seed": [0, 1, 2, 0, 1, 2],
            "delta_abs_error": [-0.1, -0.1, -0.1, 0.0, 0.0, 0.0],
        }
    )
    summary = V3.evaluate_gate(
        condition,
        ci=(-0.1, -0.01),
        coverage=0.99,
        locator_coverage=1.0,
        maximum_introduced_clipping_fraction=0.0,
        maximum_abs_change_dn=4.0,
        inactive_exact_identity=True,
    )
    assert summary["gate"] == "PASS"
    one_positive_seed = condition.copy()
    one_positive_seed.loc[one_positive_seed["seed"] == 2, "delta_abs_error"] = 0.01
    failed = V3.evaluate_gate(
        one_positive_seed,
        ci=(-0.1, -0.01),
        coverage=0.99,
        locator_coverage=1.0,
        maximum_introduced_clipping_fraction=0.0,
        maximum_abs_change_dn=4.0,
        inactive_exact_identity=True,
    )
    assert failed["gate"] == "FAIL"
    coverage_failed = V3.evaluate_gate(
        condition,
        ci=(-0.1, -0.01),
        coverage=0.94,
        locator_coverage=1.0,
        maximum_introduced_clipping_fraction=0.0,
        maximum_abs_change_dn=4.0,
        inactive_exact_identity=True,
    )
    assert coverage_failed["gate"] == "FAIL"


def test_progress_validation_rejects_relocalized_or_wrong_route() -> None:
    payload = {
        "locator_reused_for_all_outputs": True,
        "route_active": True,
        "events": [
            {"run_key": "raw_none"},
            {"run_key": f"{V3.ROUTE_ID}_0"},
            {"run_key": f"{V3.ROUTE_ID}_1"},
            {"run_key": f"{V3.ROUTE_ID}_2"},
        ],
    }
    V3.validate_event_coverage(payload)
    payload["locator_reused_for_all_outputs"] = False
    with pytest.raises(V3.ProtocolError, match="locator"):
        V3.validate_event_coverage(payload)
    payload["locator_reused_for_all_outputs"] = True
    payload["events"].pop()
    with pytest.raises(V3.ProtocolError, match="coverage"):
        V3.validate_event_coverage(payload)


def _valid_progress_payload() -> dict:
    identity = [
        (1, "vertical_edge"),
        (2, "vertical_edge"),
        (3, "horizontal_edge"),
        (4, "horizontal_edge"),
    ]
    diagnostics = V3.identity_diagnostics(route_id="raw", seed="none")
    events = []
    for run_key in ["raw_none", "raw_exact_bypass_0", "raw_exact_bypass_1", "raw_exact_bypass_2"]:
        record = dict(diagnostics)
        record["route_id"] = "raw" if run_key == "raw_none" else "raw_exact_bypass"
        events.append(
            {
                "run_key": run_key,
                "diagnostics": record,
                "rois": [
                    {
                        "roi_id": roi_id,
                        "orientation": orientation,
                        "valid": True,
                        "reason": "ok",
                        "mtf50_canonical_cy_px": 0.2,
                        "mtf50_native_cy_px": 0.1,
                    }
                    for roi_id, orientation in identity
                ],
            }
        )
    return {
        "schema_version": V3.PROGRESS_SCHEMA,
        "study": "prospective_holdout_403",
        "ordinal": 1,
        "frame_id": "60km_2.5m_ISO200/frame_000001.png",
        "condition_cell": "60km_2.5m_ISO200",
        "speed_kmh": 60,
        "distance_m": 2.5,
        "iso": 200,
        "route_active": False,
        "image_sha256": "a" * 64,
        "contract_sha256": "contract",
        "c057_restoration_freeze_sha256": "c057",
        "locator_execution_freeze_sha256": "locator",
        "execution_freeze_sha256": "execution",
        "inventory_sha256": "inventory",
        "selection_sha256": "selection",
        "locator_reused_for_all_outputs": True,
        "locator": {"locator_reused_for_all_outputs": True, "score": 0.8},
        "status": "ok",
        "reason": "ok",
        "events": events,
    }


def _validate_sample(payload: dict) -> None:
    V3.validate_progress_payload(
        payload,
        {
            "ordinal": 1,
            "frame_id": "60km_2.5m_ISO200/frame_000001.png",
            "condition_cell": "60km_2.5m_ISO200",
            "speed_kmh": 60,
            "distance_m": 2.5,
            "iso": 200,
            "route_active": False,
            "image_sha256": "a" * 64,
        },
        study="prospective_holdout_403",
        contract_hash="contract",
        c057_hash="c057",
        locator_freeze_hash="locator",
        execution_freeze_hash="execution",
        inventory_hash="inventory",
        selection_hash="selection",
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("contract_sha256", "tampered"),
        ("selection_sha256", "tampered"),
        ("locator_execution_freeze_sha256", "tampered"),
        ("image_sha256", "b" * 64),
    ],
)
def test_progress_binding_tamper_is_rejected(field: str, value: str) -> None:
    payload = _valid_progress_payload()
    _validate_sample(payload)
    payload[field] = value
    with pytest.raises(V3.ProtocolError):
        _validate_sample(payload)


def test_progress_roi_and_diagnostic_tamper_is_rejected() -> None:
    payload = _valid_progress_payload()
    payload["events"][0]["rois"][0]["orientation"] = "horizontal_edge"
    with pytest.raises(V3.ProtocolError, match="ROI identity"):
        _validate_sample(payload)
    payload = _valid_progress_payload()
    payload["events"][1]["diagnostics"]["introduced_clipping_fraction"] = 2.0
    with pytest.raises(V3.ProtocolError, match="diagnostic"):
        _validate_sample(payload)


def test_local_and_global_affine_coordinates_are_equivalent() -> None:
    locator = V3.locator_module()
    located = locator.LocatorResult(1000.0, 500.0, 300.0, 5.0, 0.9, 2000, 1200)
    x0, y0, _, _ = V3.native_context_bounds((1200, 2000), located)
    local = V3.local_locator(located, x0, y0)
    global_matrix = locator.normalization_matrix(located)
    local_matrix = locator.normalization_matrix(local)
    points_local = np.array([[50.0, 80.0, 1.0], [300.0, 400.0, 1.0]]).T
    points_global = points_local.copy()
    points_global[0] += x0
    points_global[1] += y0
    assert np.allclose(global_matrix @ points_global, local_matrix @ points_local, atol=1e-10)


def test_name_only_inventory_freezes_exact_403_45_and_active_count(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    conditions = sorted(V3.all_nonreference_conditions())
    created = 0
    for condition_index, condition in enumerate(conditions):
        directory = raw / condition
        directory.mkdir(parents=True)
        count = 8 if condition_index < 2 else 9
        for frame_index in range(count):
            (directory / f"frame_{frame_index:06d}.png").write_bytes(b"")
            created += 1
    assert created == 403
    rows, audit = V3.discover_holdout_inventory(
        raw,
        pd.DataFrame({"frame_id": []}),
        bank_frame_ids=set(),
        expected_frames=403,
        expected_conditions=45,
    )
    assert len(rows) == 403 and audit["condition_cells"] == 45
    assert audit["active_frames"] == sum(
        row["speed_kmh"] == 80 and row["iso"] in {100, 200} for row in rows
    )
    assert all("sha256" not in row for row in rows)


def test_holdout_decode_requires_exact_public_8k_shape() -> None:
    valid = np.zeros((5460, 8192), dtype=np.uint8)
    assert V3.validate_decoded_holdout_image(valid) == (8192, 5460, "uint8")
    with pytest.raises(V3.ProtocolError, match="shape changed"):
        V3.validate_decoded_holdout_image(np.zeros((5459, 8192), dtype=np.uint8))
    with pytest.raises(V3.ProtocolError, match="uint8"):
        V3.validate_decoded_holdout_image(np.zeros((1, 1), dtype=np.float32))


def test_execution_manifest_reconstruction_exposes_duplicate_and_prior_overlap() -> None:
    progress = [
        {
            "ordinal": 1,
            "frame_id": "a/x.png",
            "condition_cell": "a",
            "relative_path": "a/x.png",
            "image_sha256": "a" * 64,
            "image_width": 8192,
            "image_height": 5460,
            "image_dtype": "uint8",
            "status": "ok",
            "reason": "ok",
        },
        {
            "ordinal": 2,
            "frame_id": "b/y.png",
            "condition_cell": "b",
            "relative_path": "b/y.png",
            "image_sha256": "a" * 64,
            "image_width": 8192,
            "image_height": 5460,
            "image_dtype": "uint8",
            "status": "ok",
            "reason": "ok",
        },
    ]
    manifest = V3.expected_holdout_execution_manifest(
        progress,
        execution_hash="execution",
        inventory_hash="inventory",
        exposed_hashes={"a" * 64},
        bank_hashes=set(),
    )
    assert manifest["duplicate_sha256_within_holdout"] == 1
    assert manifest["sha256_overlap_with_c057_exposed"] == 1
    tampered = dict(manifest)
    tampered["frames"] = 3
    with pytest.raises(V3.ProtocolError, match="numeric mismatch"):
        V3.assert_nested_close(tampered, manifest, path="manifest")


def test_failed_progress_is_immutable_audit_row() -> None:
    payload = _valid_progress_payload()
    payload["status"] = "failed"
    payload["reason"] = "LocatorFailure: score below gate"
    payload["events"] = []
    _validate_sample(payload)
    payload["events"] = [{"run_key": "raw_none"}]
    with pytest.raises(V3.ProtocolError, match="failed progress"):
        _validate_sample(payload)


def test_nested_release_comparison_rejects_summary_tamper() -> None:
    expected = {"summary": {"gate": "PASS", "mean": -0.001}}
    V3.assert_nested_close(expected, expected)
    tampered = {"summary": {"gate": "PASS", "mean": 0.001}}
    with pytest.raises(V3.ProtocolError, match="numeric mismatch"):
        V3.assert_nested_close(tampered, expected)
