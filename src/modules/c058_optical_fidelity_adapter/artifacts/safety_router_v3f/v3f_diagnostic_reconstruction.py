"""Diagnostic-only reconstruction of the preserved v3f public progress.

This file is not an execution or selection input.  It exists to inspect the
scientific values after the frozen v3f aggregator rejected the new unnumbered
inactive bypass key.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
PROGRESS = BASE / "public_e2e_development" / "progress"
V3E_FINAL = (
    BASE.parent
    / "safety_router_v3e"
    / "public_e2e_development"
    / "public_e2e_development_final.json"
)
BOOTSTRAP_SEED = 20261058


def main() -> None:
    references = json.loads(V3E_FINAL.read_text(encoding="utf-8"))[
        "reference_mtf50_native_cy_px_by_distance"
    ]
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    files = sorted(PROGRESS.glob("*.json"))
    for path in files:
        frame = json.loads(path.read_text(encoding="utf-8"))
        if frame["status"] != "ok":
            continue
        for event in frame["events"]:
            valid = [
                float(roi["mtf50_native_cy_px"])
                for roi in event["rois"]
                if roi["valid"]
            ]
            if len(valid) != 4:
                continue
            rows.append(
                {
                    "frame_id": frame["frame_id"],
                    "condition_cell": frame["condition_cell"],
                    "distance_m": float(frame["distance_m"]),
                    "iso": int(frame["iso"]),
                    "route_active": bool(frame["route_active"]),
                    "is_reference": bool(frame["is_reference"]),
                    "run_key": event["run_key"],
                    "mtf50": float(np.mean(valid)),
                }
            )
            diagnostics.append(
                {
                    "route_active": bool(frame["route_active"]),
                    "run_key": event["run_key"],
                    **event["diagnostics"],
                }
            )
    table = pd.DataFrame(rows)
    active = table[table["route_active"] & ~table["is_reference"]].copy()
    condition = (
        active.groupby(["condition_cell", "distance_m", "iso", "run_key"])["mtf50"]
        .median()
        .reset_index()
    )
    raw = condition[condition["run_key"] == "raw_none"][
        ["condition_cell", "mtf50"]
    ].rename(columns={"mtf50": "raw_mtf50"})
    routed = condition[condition["run_key"] != "raw_none"].merge(
        raw, on="condition_cell", validate="one_to_one"
    )
    routed["reference_mtf50"] = [
        float(references[f"{distance:.1f}"]) for distance in routed["distance_m"]
    ]
    routed["delta_abs_error"] = np.abs(
        routed["mtf50"] - routed["reference_mtf50"]
    ) - np.abs(routed["raw_mtf50"] - routed["reference_mtf50"])
    values = routed["delta_abs_error"].to_numpy(dtype=float)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    boot = values[
        generator.integers(0, len(values), size=(10_000, len(values)))
    ].mean(axis=1)
    active_diag = pd.DataFrame(diagnostics)
    active_diag = active_diag[
        active_diag["route_active"] & (active_diag["run_key"] != "raw_none")
    ]
    result = {
        "diagnostic_only": True,
        "progress_files": len(files),
        "conditions": routed[
            [
                "condition_cell",
                "raw_mtf50",
                "mtf50",
                "reference_mtf50",
                "delta_abs_error",
            ]
        ].to_dict(orient="records"),
        "mean_delta_abs_error_cy_px": float(values.mean()),
        "condition_bootstrap_95pct_ci": [
            float(value) for value in np.percentile(boot, [2.5, 97.5])
        ],
        "all_condition_means_lt_zero": bool((values < 0.0).all()),
        "maximum_preclip_overshoot_fraction": float(
            active_diag["preclip_overshoot_fraction"].max()
        ),
        "maximum_introduced_clipping_fraction": float(
            active_diag["introduced_clipping_fraction"].max()
        ),
        "maximum_abs_change_dn": float(active_diag["max_abs_change_dn"].max()),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output = BASE / "v3f_diagnostic_reconstruction.json"
    output.write_text(rendered, encoding="utf-8")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="ascii"
    )
    print(rendered, end="")


if __name__ == "__main__":
    main()
