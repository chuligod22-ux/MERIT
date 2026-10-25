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


def test_c057_verified_environment_loader_returns_frozen_four_value_contract() -> None:
    """Integration smoke: exercise the real C057 loader without a mock or GPU."""

    module = V3.c057_module()
    loaded = module.load_verified_environment(module.DEFAULT_FREEZE)
    assert isinstance(loaded, tuple)
    assert len(loaded) == 4
    protocol, original_freeze, cohort, freeze_hash = loaded
    assert isinstance(protocol, dict)
    assert isinstance(original_freeze, dict)
    assert isinstance(cohort, pd.DataFrame) and not cohort.empty
    assert isinstance(freeze_hash, str) and len(freeze_hash) == 64


def test_model_callback_reads_amp_dtype_from_frozen_restoration_root() -> None:
    captured: dict[str, object] = {}

    def tiled(crop: np.ndarray, model: object, device: object, **kwargs: object) -> np.ndarray:
        captured.update(kwargs)
        assert model == "model" and device == "device"
        return crop.astype(np.float32) / 255.0

    protocol = {
        "restoration": {
            "amp_dtype": "float16",
            "tiling": {
                "tile_size": 256,
                "overlap": 32,
                "batch_size": 1,
                "positive_hann_floor": 0.01,
            },
        }
    }
    callback = V3._model_callback(
        object(), protocol, {("on", "0"): "model"}, tiled, "device"
    )
    output = callback(np.zeros((8, 8), dtype=np.uint8), 0)
    assert output.shape == (8, 8)
    assert captured["amp_dtype"] == "float16"


def test_public_runner_full_raw_path_reproduces_frozen_locator_result() -> None:
    """Regression: the public runner must localize the 8K RAW, not the C057 extract."""

    frame_id = "60km_2.5m_ISO100/frame_000102.png"
    expected = pd.read_csv(
        BASE / "artifacts" / "locator_development_v1" / "locator_frame_results.csv"
    )
    expected = expected[expected["frame_id"] == frame_id].iloc[0]
    locator = V3.locator_module()
    image = locator.read_gray(V3.RAW_ROOT / frame_id)
    located = locator.locate_chart(image)
    assert image.shape == (5460, 8192) and image.dtype == np.uint8
    assert located.score == pytest.approx(float(expected["score"]), abs=1e-12)
    assert located.center_x == pytest.approx(float(expected["center_x_raw"]), abs=1e-9)
    assert located.center_y == pytest.approx(float(expected["center_y_raw"]), abs=1e-9)
    assert located.radius == pytest.approx(float(expected["radius_raw"]), abs=1e-9)


def test_router_is_active_only_for_frozen_speed_and_iso_strata() -> None:
    assert V3.router_active(80, 100, 2.5)
    assert V3.router_active(80, 200, 4.5)
    for speed in (60, 80):
        for iso in (100, 200, 400, 800, 1600):
            for distance in (2.5, 3.5, 4.5, 5.5, 6.5):
                assert V3.router_active(speed, iso, distance) is (
                    speed == 80
                    and iso in {100, 200}
                    and distance in {2.5, 3.5, 4.5}
                )


def test_inactive_high_distance_router_is_byte_exact_and_skips_model() -> None:
    raw = np.arange(64, dtype=np.uint8).reshape(8, 8)
    called = False

    def forbidden_model(_: np.ndarray, __: int) -> np.ndarray:
        nonlocal called
        called = True
        raise AssertionError("inactive route must not call the model")

    outputs = V3.route_native_crop(raw, 80, 100, 5.5, forbidden_model)
    assert not called
    assert set(outputs) == {"raw_none", "raw_exact_bypass"}
    assert all(np.array_equal(value[0], raw) for value in outputs.values())
    assert all(value[1]["exact_identity"] for value in outputs.values())


def test_active_router_uses_frozen_conditioned_expert_and_strength() -> None:
    raw = np.full((16, 16), 128, dtype=np.uint8)
    called: list[int] = []

    def model(_: np.ndarray, seed: int) -> np.ndarray:
        called.append(seed)
        return np.full((16, 16), 0.49 + seed * 0.001, dtype=np.float32)

    outputs = V3.route_native_crop(raw, 80, 100, 2.5, model)
    assert called == [1]
    assert set(outputs) == {
        "raw_none",
        f"{V3.ROUTE_ID}_1",
    }
    assert outputs[f"{V3.ROUTE_ID}_1"][1]["route_id"] == V3.ROUTE_ID
    assert outputs[f"{V3.ROUTE_ID}_1"][1]["conditioned_alpha"] == pytest.approx(-0.2)
    assert V3.expert_configuration(4.5, 100) == (0, -0.9)


def test_condition_table_accepts_unnumbered_inactive_bypass() -> None:
    rows = []
    for run_key, value in (("raw_none", 0.10), ("raw_exact_bypass", 0.10)):
        rows.append(
            {
                "frame_id": "80km_5.5m_ISO100/frame.png",
                "condition_cell": "80km_5.5m_ISO100",
                "speed_kmh": 80,
                "distance_m": 5.5,
                "iso": 100,
                "route_active": False,
                "is_reference": False,
                "run_key": run_key,
                "roi_id": 1,
                "orientation": "vertical_edge",
                "mtf50_native_cy_px": value,
            }
        )
    table = V3._condition_table(
        pd.DataFrame(rows), {"5.5": 0.10}, include_reference=False
    )
    assert len(table) == 1
    assert int(table.iloc[0]["seed"]) == -1
    assert float(table.iloc[0]["delta_abs_error"]) == pytest.approx(0.0)


def test_native_context_contains_every_canonical_roi_with_fixed_margin() -> None:
    locator = type("Locator", (), {"center_x": 1000.0, "center_y": 500.0, "radius": 300.0})()
    x0, y0, x1, y1 = V3.native_context_bounds((1200, 2000), locator)
    assert (x0, y0, x1, y1) == (685, 185, 1315, 815)


def test_native_mtf_scale_is_explicit() -> None:
    assert V3.canonical_to_native_mtf(0.2, 640.0) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        V3.canonical_to_native_mtf(0.2, 0.0)


@pytest.mark.parametrize(
    ("canonical", "native", "expected_valid", "expected_reason"),
    [
        (0.2, 0.5, True, "ok"),
        (0.500001, 0.4, False, "canonical_nyquist_exceeded"),
        (0.2, 0.500001, False, "native_nyquist_exceeded"),
        (float("nan"), 0.2, False, "canonical_mtf50_nonfinite_or_nonpositive"),
    ],
)
def test_mtf50_nyquist_filter_is_fail_closed(
    canonical: float,
    native: float,
    expected_valid: bool,
    expected_reason: str,
) -> None:
    valid, reason, canonical_out, native_out = V3.enforce_mtf50_nyquist(
        estimator_valid=True,
        estimator_reason="ok",
        canonical_value=canonical,
        native_value=native,
    )
    assert valid is expected_valid and reason == expected_reason
    if expected_valid:
        assert canonical_out == canonical and native_out == native
    else:
        assert canonical_out is None and native_out is None


def test_measure_native_crop_nulls_native_nyquist_excess() -> None:
    class Locator:
        @staticmethod
        def normalize_chart(_crop: np.ndarray, _located: object):
            return np.zeros((2, 2), dtype=np.uint8), [
                {
                    "roi_id": 1,
                    "orientation": "vertical_edge",
                    "x1": 0,
                    "x2": 2,
                    "y1": 0,
                    "y2": 2,
                }
            ]

    class Estimator:
        @staticmethod
        def estimate_mtf50_v2(_patch: np.ndarray, _orientation: str, **_kwargs: object):
            return type(
                "Result",
                (),
                {"valid": True, "reason": "ok", "mtf50": 0.1473093151222759},
            )()

    located = type("Located", (), {"radius": 62.15669991296415})()
    measured = V3.measure_native_crop(
        np.zeros((2, 2), dtype=np.uint8), located, Locator(), Estimator()
    )
    assert measured == [
        {
            "roi_id": 1,
            "orientation": "vertical_edge",
            "valid": False,
            "reason": "native_nyquist_exceeded",
            "mtf50_canonical_cy_px": None,
            "mtf50_native_cy_px": None,
        }
    ]


@pytest.mark.parametrize(
    ("progress_name", "canonical", "native", "radius"),
    [
        (
            "0128_80km_6.5m_ISO200_frame_000068.png.json",
            0.1473093151222759,
            0.758389375644707,
            62.15669991296415,
        ),
        (
            "0132_80km_6.5m_ISO200_frame_000072.png.json",
            0.4915657262586504,
            1.2813798565188994,
            122.75909567527064,
        ),
    ],
)
def test_real_public_v3c_inactive_raw_diagnostics_are_native_nyquist_invalid(
    progress_name: str,
    canonical: float,
    native: float,
    radius: float,
) -> None:
    fixture = (
        BASE
        / "artifacts"
        / "safety_router_v3c"
        / "public_e2e_development"
        / "progress"
        / progress_name
    )
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    assert payload["status"] == "ok" and payload["route_active"] is False
    assert payload["locator"]["radius_raw"] == pytest.approx(radius)
    raw = next(event for event in payload["events"] if event["run_key"] == "raw_none")
    roi = next(item for item in raw["rois"] if item["roi_id"] == 1)
    assert roi["mtf50_canonical_cy_px"] == pytest.approx(canonical)
    assert roi["mtf50_native_cy_px"] == pytest.approx(native)
    checked = V3.enforce_mtf50_nyquist(
        estimator_valid=roi["valid"],
        estimator_reason=roi["reason"],
        canonical_value=roi["mtf50_canonical_cy_px"],
        native_value=roi["mtf50_native_cy_px"],
    )
    assert checked == (False, "native_nyquist_exceeded", None, None)


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
    for name in ("frame_000001.png", "frame_000002.png", "frame_000003.png"):
        (condition / name).write_bytes(b"not-decoded")
    exposed = pd.DataFrame({"frame_id": ["80km_2.5m_ISO100/frame_000001.png"]})
    rows, audit = V3.discover_holdout_inventory(
        raw,
        exposed,
        bank_frame_ids={"80km_2.5m_ISO100/frame_000002.png"},
        expected_frames=1,
        expected_conditions=1,
        allowed_conditions={"80km_2.5m_ISO100"},
    )
    assert rows[0]["frame_id"].endswith("frame_000003.png")
    assert "sha256" not in rows[0] and "bytes" not in rows[0]
    assert audit["frame_id_overlap_with_exposed"] == 0
    assert audit["excluded_c006_bank_parent_frame_ids"] == 1


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


def test_gate_requires_every_conditioned_expert_to_improve_and_inactive_identity() -> None:
    condition = pd.DataFrame(
        {
            "condition_cell": ["a", "b", "c"],
            "active": [True, True, False],
            "seed": [1, 0, 0],
            "delta_abs_error": [-0.1, -0.05, 0.0],
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
    one_positive_condition = condition.copy()
    one_positive_condition.loc[one_positive_condition["condition_cell"] == "b", "delta_abs_error"] = 0.01
    failed = V3.evaluate_gate(
        one_positive_condition,
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
    active_ci_failed = V3.evaluate_gate(
        condition,
        ci=(-0.1, -0.01),
        active_ci=(-0.1, 0.001),
        coverage=0.99,
        locator_coverage=1.0,
        maximum_introduced_clipping_fraction=0.0,
        maximum_abs_change_dn=4.0,
        inactive_exact_identity=True,
    )
    assert active_ci_failed["gate"] == "FAIL"


def test_progress_validation_rejects_relocalized_or_wrong_route() -> None:
    payload = {
        "locator_reused_for_all_outputs": True,
        "route_active": True,
        "events": [
            {"run_key": "raw_none"},
            {"run_key": f"{V3.ROUTE_ID}_1"},
        ],
        "distance_m": 2.5,
        "iso": 100,
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
    for run_key in ["raw_none", "raw_exact_bypass"]:
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
        "study": V3.HOLDOUT_STUDY,
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
        study=V3.HOLDOUT_STUDY,
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
    payload = _valid_progress_payload()
    payload["events"][0]["rois"][0]["mtf50_native_cy_px"] = 0.500001
    with pytest.raises(V3.ProtocolError, match="physical range"):
        _validate_sample(payload)
    payload = _valid_progress_payload()
    payload["events"][0]["rois"][0]["valid"] = False
    payload["events"][0]["rois"][0]["reason"] = "native_nyquist_exceeded"
    with pytest.raises(V3.ProtocolError, match="null MTF"):
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


def test_name_only_inventory_freezes_exact_280_29_and_active_domain(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    active_conditions = [
        f"80km_{distance:.1f}m_ISO{iso}"
        for distance in (2.5, 3.5, 4.5)
        for iso in (100, 200)
    ]
    inactive_pool = sorted(V3.all_nonreference_conditions() - set(active_conditions))
    conditions = sorted(active_conditions + inactive_pool[:23])
    active_counts = [8, 8, 7, 9, 8, 7]
    created = 0
    for condition_index, condition in enumerate(conditions):
        directory = raw / condition
        directory.mkdir(parents=True)
        if condition in active_conditions:
            count = active_counts[active_conditions.index(condition)]
        else:
            inactive_index = [item for item in conditions if item not in active_conditions].index(condition)
            count = 11 if inactive_index < 3 else 10
        for frame_index in range(count):
            (directory / f"frame_{frame_index:06d}.png").write_bytes(b"")
            created += 1
    assert created == 280
    rows, audit = V3.discover_holdout_inventory(
        raw,
        pd.DataFrame({"frame_id": []}),
        bank_frame_ids=set(),
        expected_frames=280,
        expected_conditions=29,
        allowed_conditions=set(conditions),
    )
    assert len(rows) == 280 and audit["condition_cells"] == 29
    assert audit["active_frames"] == 47
    assert audit["active_condition_cells"] == 6
    assert audit["active_distance_cells"] == 3
    assert audit["inactive_condition_cells"] == 23
    assert all("sha256" not in row for row in rows)
    assert V3.EVIDENCE_LABEL == (
        "strict parent-disjoint endpoint holdout with 6/10 active conditions "
        "and 3/5 active distances"
    )


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
