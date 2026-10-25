"""Select bounded physics-branch gains on the unblinded C009 development set."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c018_common as common  # noqa: E402


OUTPUT = common.C018 / "runs" / "development_v1" / "physics_scan.json"


def load_development() -> tuple[dict, np.ndarray, list[dict[str, str]], object]:
    evaluation = common.import_file(
        "c009_controlled_eval_c018",
        common.C009 / "scripts" / "c009_controlled_eval.py",
    )
    contract = evaluation.common.load_contract()
    bank_path = common.C006 / "artifacts" / "d0_bank_448.npy"
    manifest_path = common.C006 / "artifacts" / "d0_bank_manifest.csv"
    bank_all = np.load(bank_path, allow_pickle=False)
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows_all = list(csv.DictReader(handle))
    selected = [
        (bank_all[int(row["bank_index"])], row)
        for row in rows_all
        if row["split"] == "synthetic_holdout"
    ]
    bank = np.stack([item[0] for item in selected]).astype(np.float32) / 255.0
    rows = [item[1] for item in selected]
    return contract, bank, rows, evaluation


def summarize(values: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([record[key] for record in values]))
        for key in values[0]
    }


def main() -> None:
    protocol = common.load_json(common.DEVELOPMENT_PROTOCOL)
    contract, bank, rows, evaluation = load_development()
    seeds = contract["wp4_ood"]["stochastic_seeds"]
    families = contract["wp4_ood"]["families"]
    gains = [float(value) for value in protocol["physics_gain_grid"]]
    results: dict[str, dict[str, object]] = {}
    for family, severities in families.items():
        if family == "poisson_gaussian":
            continue
        family_results: dict[str, object] = {}
        for severity_index, parameters in enumerate(severities, start=1):
            candidate_gains = gains if family in {
                "linear_motion_psf",
                "spatially_varying_blur",
            } else [0.0]
            gain_results: dict[str, dict[str, float]] = {}
            for gain in candidate_gains:
                records: list[dict[str, float]] = []
                for clean, row in zip(bank, rows):
                    bank_index = int(row["bank_index"])
                    family_index = list(families).index(family)
                    for seed in seeds:
                        rng = evaluation.deterministic_rng(
                            int(seed),
                            bank_index,
                            100 + family_index * 10 + severity_index,
                        )
                        degraded = evaluation.ood_degrade(
                            clean, family, parameters, rng
                        )
                        restored = common.restore_physics(
                            degraded, family, parameters, gain
                        )
                        records.append(
                            {
                                "delta_psnr_vs_degraded": common.psnr(clean, restored)
                                - common.psnr(clean, degraded),
                                "delta_gradient_vs_degraded": common.gradient_mae(
                                    clean, restored
                                )
                                - common.gradient_mae(clean, degraded),
                                "delta_overshoot_vs_degraded": common.edge_normal_overshoot_p95(
                                    clean, restored
                                )
                                - common.edge_normal_overshoot_p95(clean, degraded),
                            }
                        )
                gain_results[f"{gain:.6f}"] = summarize(records)
            eligible = [
                (float(gain), summary)
                for gain, summary in gain_results.items()
                if summary["delta_gradient_vs_degraded"] <= 0.0
                and summary["delta_overshoot_vs_degraded"] <= 0.01
            ]
            pool = eligible or [
                (float(gain), summary) for gain, summary in gain_results.items()
            ]
            selected_gain, selected_summary = max(
                pool, key=lambda item: item[1]["delta_psnr_vs_degraded"]
            )
            family_results[str(severity_index)] = {
                "parameters": parameters,
                "selected_gain": selected_gain,
                "selected": selected_summary,
                "grid": gain_results,
            }
            print(
                json.dumps(
                    {
                        "family": family,
                        "severity": severity_index,
                        "selected_gain": selected_gain,
                        **selected_summary,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        results[family] = family_results
    payload = {
        "schema_version": "MERIT-C018-PHYSICS-SCAN-v1",
        "status": "DEVELOPMENT_COMPLETE",
        "scientific_status": "UNBLINDED_C009_DEVELOPMENT_ONLY",
        "completed_at_kst": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol_sha256": common.sha256_file(common.DEVELOPMENT_PROTOCOL),
        "parents": len({row["parent_sha256"] for row in rows}),
        "crops": len(rows),
        "results": results,
    }
    common.write_json_atomic(OUTPUT, payload)
    print(json.dumps({"status": "PASS", "output": str(OUTPUT.resolve())}))


if __name__ == "__main__":
    main()
