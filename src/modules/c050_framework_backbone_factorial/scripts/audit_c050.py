"""Independent arithmetic and provenance audit for completed C050 outputs."""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
C050 = HERE.parent
REVISION = C050.parent
PROTOCOL = C050 / "protocol" / "factorial_protocol_v1.json"
OUTPUT = C050 / "runs" / "factorial_v1"
BACKBONES = ("nafnet", "mprnet", "restormer")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def factor(cell: np.ndarray) -> dict[str, np.ndarray]:
    framework = np.mean(cell[..., :, 1] - cell[..., :, 0], axis=-1)
    marginal = np.mean(cell, axis=-1)
    backbone = np.max(
        np.stack(
            [
                np.abs(marginal[..., left] - marginal[..., right])
                for left, right in combinations(range(3), 2)
            ],
            axis=-1,
        ),
        axis=-1,
    )
    grand = cell.mean(axis=(-2, -1))
    arm = cell.mean(axis=-2)
    ss_f = 3 * np.sum((arm - grand[..., None]) ** 2, axis=-1)
    ss_b = 2 * np.sum((marginal - grand[..., None]) ** 2, axis=-1)
    return {
        "framework_main_effect_db": framework,
        "max_backbone_marginal_contrast_db": backbone,
        "framework_minus_backbone_db": framework - backbone,
        "framework_to_backbone_ss_ratio": ss_f / np.maximum(ss_b, 1e-18),
    }


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    summary_path = OUTPUT / "factorial_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for backbone in BACKBONES:
        for seed in protocol["model_seeds"]:
            cell_path = OUTPUT / f"{backbone}_seed{seed}.json"
            cell = json.loads(cell_path.read_text(encoding="utf-8"))
            records_path = Path(cell["records_path"])
            cell_rows = [
                json.loads(line)
                for line in records_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            passed = bool(
                cell["records"] == 1350
                and cell["parents"] == 302
                and len(cell_rows) == 1350
                and len({row["parent_sha256"] for row in cell_rows}) == 302
                and cell["protocol_payload_sha256"] == protocol["payload_sha256"]
                and sha256_file(records_path) == cell["records_sha256"]
            )
            checks.append({"cell": f"{backbone}_seed{seed}", "passed": passed})
            rows.extend(cell_rows)

    parents = sorted({row["parent_sha256"] for row in rows})
    conditions = [item["id"] for item in protocol["conditions"]]
    values = np.zeros((len(parents), 3, 3, 2), dtype=np.float64)
    for parent_index, parent in enumerate(parents):
        for backbone_index, backbone in enumerate(BACKBONES):
            for condition_index, condition in enumerate(conditions):
                subset = [
                    row
                    for row in rows
                    if row["parent_sha256"] == parent
                    and row["backbone"] == backbone
                    and row["condition"] == condition
                ]
                if not subset:
                    raise RuntimeError("Missing parent cell")
                values[parent_index, backbone_index, condition_index, 0] = np.mean(
                    [row["base_gain_db"] for row in subset]
                )
                values[parent_index, backbone_index, condition_index, 1] = np.mean(
                    [row["merit_gain_db"] for row in subset]
                )

    point = factor(values.mean(axis=(0, 2)))
    rng = np.random.default_rng(int(protocol["bootstrap_seed"]))
    draws: dict[str, list[np.ndarray]] = {}
    cell_draws: list[np.ndarray] = []
    for begin in range(0, int(protocol["bootstrap_replicates"]), 200):
        count = min(200, int(protocol["bootstrap_replicates"]) - begin)
        indices = rng.integers(0, len(parents), size=(count, len(parents)))
        sampled = values[indices]
        cell_draws.append(np.mean(sampled[..., 1] - sampled[..., 0], axis=1))
        stats = factor(np.mean(sampled, axis=(1, 3)))
        for key, value in stats.items():
            draws.setdefault(key, []).append(value)
    merged = {key: np.concatenate(parts) for key, parts in draws.items()}

    differences = []
    for key, estimate in point.items():
        reported = summary["factorial"][key]
        differences.extend(
            [
                abs(float(estimate) - reported["estimate"]),
                abs(float(np.mean(merged[key])) - reported["mean"]),
                abs(float(np.quantile(merged[key], 0.025)) - reported["ci95_lower"]),
                abs(float(np.quantile(merged[key], 0.975)) - reported["ci95_upper"]),
            ]
        )
    merged_cell = np.concatenate(cell_draws)
    for backbone_index, backbone in enumerate(BACKBONES):
        for condition_index, condition in enumerate(conditions):
            reported = summary["by_backbone_condition"][backbone][condition][
                "framework_delta_psnr_db"
            ]
            point_value = np.mean(
                values[:, backbone_index, condition_index, 1]
                - values[:, backbone_index, condition_index, 0]
            )
            draw = merged_cell[:, backbone_index, condition_index]
            differences.extend(
                [
                    abs(float(point_value) - reported["estimate"]),
                    abs(float(draw.mean()) - reported["mean"]),
                    abs(float(np.quantile(draw, 0.025)) - reported["ci95_lower"]),
                    abs(float(np.quantile(draw, 0.975)) - reported["ci95_upper"]),
                ]
            )

    c049_root = (
        REVISION
        / "c049_restormer_psnr_independent_confirmation"
        / "runs"
        / "confirmation_v1"
    )
    c049_delta = {}
    for seed in protocol["model_seeds"]:
        path = c049_root / f"c040_seed{seed}_records.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            c049_delta[(seed, row["sample_id"])] = row["delta_psnr_db"]
    rest_differences = [
        abs(
            row["framework_delta_db"]
            - c049_delta[(row["model_seed"], row["sample_id"])]
        )
        for row in rows
        if row["backbone"] == "restormer"
    ]

    legacy = {
        "nafnet": {"on": 77.0, "off": 5.0},
        "plainunet": {"on": 41.0, "off": 6.0},
        "restormerlite": {"on": 72.0, "off": 5.0},
        "mambalite": {"on": 39.0, "off": 6.0},
    }
    framework_effects = [value["on"] - value["off"] for value in legacy.values()]
    backbone_marginals = [np.mean(list(value.values())) for value in legacy.values()]
    legacy_analysis = {
        "framework_main_contrast_percentage_points": float(np.mean(framework_effects)),
        "framework_effect_range_percentage_points": [
            float(min(framework_effects)),
            float(max(framework_effects)),
        ],
        "max_backbone_marginal_contrast_percentage_points": float(
            max(backbone_marginals) - min(backbone_marginals)
        ),
        "framework_to_backbone_point_ratio": float(
            np.mean(framework_effects)
            / (max(backbone_marginals) - min(backbone_marginals))
        ),
        "status": "DESCRIPTIVE_ONLY_NO_REPLICATE_CI",
    }
    legacy_source = REVISION / "manuscript_r1_3" / "main.tex"

    maximum_difference = max(differences)
    result = {
        "schema_version": "MERIT-C050-INDEPENDENT-AUDIT-v1",
        "status": "PASS"
        if all(item["passed"] for item in checks)
        and len(rows) == 12150
        and len(parents) == 302
        and maximum_difference <= 1e-12
        and max(rest_differences) <= 1e-12
        else "FAIL",
        "cell_checks": checks,
        "records": len(rows),
        "parents": len(parents),
        "summary_sha256": sha256_file(summary_path),
        "maximum_recalculation_difference": maximum_difference,
        "c049_restormer_delta_max_abs_difference": max(rest_differences),
        "legacy_table5_descriptive_factorial": legacy_analysis,
        "legacy_table5_source": {
            "path": str(legacy_source.resolve()),
            "sha256": sha256_file(legacy_source),
        },
        "legacy_table5_limitation": (
            "The manuscript table contains one detector-region value per cell and "
            "does not provide independent replicate units for a confidence interval."
        ),
    }
    path = OUTPUT / "independent_audit_v1.json"
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
