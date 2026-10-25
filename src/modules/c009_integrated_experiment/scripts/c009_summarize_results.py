"""Apply the frozen G12 gates after every C009 evaluation block is complete."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import c009_evaluation_common as common


HERE = Path(__file__).resolve().parent
C009 = HERE.parent
DEFAULT_OUTPUT = C009 / "evaluation" / "final_summary"
BOOTSTRAP_SEED = 20260924
BOOTSTRAP_REPLICATES = 10_000


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile_ci(values: np.ndarray) -> tuple[float, float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(finite)), float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))


def cluster_bootstrap(
    frame: pd.DataFrame,
    value: str,
    cluster: str,
    *,
    strata: str | None = None,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    reduced = frame[[value, cluster] + ([strata] if strata else [])].dropna()
    if reduced.empty:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "clusters": 0}
    rng = np.random.RandomState(seed)
    if strata is None:
        means = reduced.groupby(cluster, sort=True)[value].mean().to_numpy(np.float64)
        indices = rng.randint(0, len(means), size=(replicates, len(means)))
        samples = means[indices].mean(axis=1)
        point = float(means.mean())
        count = len(means)
    else:
        strata_values = sorted(reduced[strata].unique())
        arrays = [
            group.groupby(cluster, sort=True)[value].mean().to_numpy(np.float64)
            for _, group in reduced.groupby(strata, sort=True)
        ]
        samples = np.zeros(replicates, dtype=np.float64)
        for values in arrays:
            indices = rng.randint(0, len(values), size=(replicates, len(values)))
            samples += values[indices].mean(axis=1) / len(arrays)
        point = float(np.mean([values.mean() for values in arrays]))
        count = int(sum(len(values) for values in arrays))
    return {
        "mean": point,
        "ci95_low": float(np.percentile(samples, 2.5)),
        "ci95_high": float(np.percentile(samples, 97.5)),
        "clusters": count,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }


def one_sided_cluster_sign_flip(
    frame: pd.DataFrame,
    value: str,
    cluster: str,
    *,
    strata: str | None = None,
    seed: int = BOOTSTRAP_SEED,
    random_replicates: int = 100_000,
) -> dict[str, Any]:
    columns = [value, cluster] + ([strata] if strata else [])
    reduced = frame[columns].dropna()
    if reduced.empty:
        return {"observed_mean": None, "one_sided_p": None, "clusters": 0}
    if strata is None:
        effects = reduced.groupby(cluster, sort=True)[value].mean().to_numpy(np.float64)
        weights = np.full(len(effects), 1.0 / len(effects), dtype=np.float64)
    else:
        grouped = reduced.groupby([strata, cluster], sort=True)[value].mean().reset_index()
        strata_count = grouped[strata].nunique()
        counts = grouped.groupby(strata)[cluster].transform("count").to_numpy(np.float64)
        effects = grouped[value].to_numpy(np.float64)
        weights = 1.0 / (strata_count * counts)
    observed = float(np.sum(weights * effects))
    weighted = weights * effects
    rng = np.random.RandomState(seed)
    exceed = 0
    if len(effects) <= 20:
        total = 1 << len(effects)
        bit_positions = np.arange(len(effects), dtype=np.uint64)
        for start in range(0, total, 65_536):
            codes = np.arange(start, min(start + 65_536, total), dtype=np.uint64)
            signs = 1.0 - 2.0 * ((codes[:, None] >> bit_positions) & 1).astype(np.float64)
            exceed += int(np.count_nonzero(signs @ weighted >= observed - 1e-15))
        p_value = exceed / total
        method = "exact_cluster_sign_flip"
        replicates = total
    else:
        for start in range(0, random_replicates, 10_000):
            size = min(10_000, random_replicates - start)
            signs = rng.choice(np.asarray([-1.0, 1.0]), size=(size, len(effects)))
            exceed += int(np.count_nonzero(signs @ weighted >= observed - 1e-15))
        p_value = (exceed + 1) / (random_replicates + 1)
        method = "random_cluster_sign_flip"
        replicates = random_replicates
    return {
        "observed_mean": observed,
        "one_sided_p": float(p_value),
        "clusters": int(len(effects)),
        "method": method,
        "replicates": int(replicates),
        "seed": seed,
    }


def apply_holm(records: dict[str, dict[str, Any]], alpha: float = 0.05) -> None:
    ordered = sorted(records, key=lambda name: float(records[name]["one_sided_sign_flip"]["one_sided_p"]))
    running = 0.0
    count = len(ordered)
    for rank, name in enumerate(ordered):
        raw = float(records[name]["one_sided_sign_flip"]["one_sided_p"])
        running = max(running, min(1.0, (count - rank) * raw))
        records[name]["holm_adjusted_one_sided_p"] = running
        records[name]["holm_positive_inference"] = bool(
            float(records[name]["mean"]) > 0 and running < alpha
        )


def controlled_frame(contract: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    specifications = common.wp1_specifications(contract) + common.wp23_specifications(contract)
    for specification in specifications:
        identity = common.model_id(specification)
        marker = common.load_json(C009 / "evaluation" / "controlled" / identity / "RUN_COMPLETE.json")
        if marker.get("status") != "PASS":
            raise common.EvaluationError(f"Incomplete controlled evaluation: {identity}")
        for row in read_jsonl(Path(marker["sample_metrics_path"])):
            row.update(specification)
            row["model_id"] = identity
            rows.append(row)
    return pd.DataFrame(rows)


def matched_arm_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    wp1 = frame[(frame["kind"] == "wp1") & (frame["domain"] == "id")].copy()
    keys = ["family", "seed", "sample_id", "parent_sha256", "condition", "stochastic_seed"]
    metrics = [
        "restored_psnr_db",
        "restored_lpips_alex",
        "restored_gradient_mae",
        "restored_edge_normal_overshoot_p95",
        "degraded_psnr_db",
        "degraded_edge_normal_overshoot_p95",
    ]
    base = wp1[wp1["arm"] == "base"][keys + metrics].copy()
    full = wp1[wp1["arm"] == "merit_full"][keys + metrics].copy()
    paired = full.merge(base, on=keys, suffixes=("_full", "_base"), validate="one_to_one")
    paired["delta_psnr_full_minus_base"] = paired["restored_psnr_db_full"] - paired["restored_psnr_db_base"]
    paired["delta_lpips_full_minus_base"] = paired["restored_lpips_alex_full"] - paired["restored_lpips_alex_base"]
    paired["delta_gradient_mae_full_minus_base"] = paired["restored_gradient_mae_full"] - paired["restored_gradient_mae_base"]
    paired["delta_overshoot_full_minus_base"] = paired["restored_edge_normal_overshoot_p95_full"] - paired["restored_edge_normal_overshoot_p95_base"]
    paired["delta_overshoot_full_minus_degraded"] = paired["restored_edge_normal_overshoot_p95_full"] - paired["degraded_edge_normal_overshoot_p95_full"]
    paired["delta_psnr_full_minus_degraded"] = paired["restored_psnr_db_full"] - paired["degraded_psnr_db_full"]
    return paired


def summarize_optical(frame: pd.DataFrame, gates: dict[str, Any]) -> dict[str, Any]:
    paired = matched_arm_deltas(frame)
    metrics = {
        name: cluster_bootstrap(paired, name, "parent_sha256")
        for name in (
            "delta_psnr_full_minus_base",
            "delta_lpips_full_minus_base",
            "delta_gradient_mae_full_minus_base",
            "delta_overshoot_full_minus_base",
            "delta_overshoot_full_minus_degraded",
            "delta_psnr_full_minus_degraded",
        )
    }
    parent_efficacy = paired.groupby("parent_sha256")["delta_psnr_full_minus_degraded"].mean()
    metrics["positive_parent_fraction_full_minus_degraded"] = float((parent_efficacy > 0).mean())
    macro = gates["optical"]["macro_primary_pass"]
    efficacy = gates["optical"]["restoration_efficacy_floor"]
    oversharpen = gates["optical"]["oversharpening_safety"]
    optical_pass = bool(
        metrics["delta_psnr_full_minus_base"]["mean"] >= macro["mean_delta_psnr_db_min"]
        and metrics["delta_psnr_full_minus_base"]["ci95_low"] > 0
        and metrics["delta_lpips_full_minus_base"]["mean"] <= macro["mean_delta_lpips_max"]
        and metrics["delta_lpips_full_minus_base"]["ci95_high"] < 0
        and metrics["delta_gradient_mae_full_minus_base"]["ci95_high"] <= 0
    )
    efficacy_pass = bool(
        metrics["delta_psnr_full_minus_degraded"]["mean"] >= efficacy["mean_delta_psnr_db_min"]
        and metrics["delta_psnr_full_minus_degraded"]["ci95_low"] > 0
        and metrics["positive_parent_fraction_full_minus_degraded"]
        >= efficacy["parents_with_positive_delta_psnr_fraction_min"]
    )
    oversharpening_pass = bool(
        metrics["delta_gradient_mae_full_minus_base"]["ci95_high"]
        <= oversharpen["macro_delta_95ci_upper_lte"]
        and metrics["delta_overshoot_full_minus_degraded"]["ci95_high"] <= 0.01
    )
    portability: dict[str, Any] = {}
    for family, group in paired.groupby("family"):
        record = cluster_bootstrap(
            group, "delta_psnr_full_minus_base", "parent_sha256", seed=BOOTSTRAP_SEED + len(portability)
        )
        record["one_sided_sign_flip"] = one_sided_cluster_sign_flip(
            group,
            "delta_psnr_full_minus_base",
            "parent_sha256",
            seed=BOOTSTRAP_SEED + len(portability),
        )
        portability[family] = record
    apply_holm(portability)
    positive = sum(record["mean"] > 0 for record in portability.values())
    holm_positive = sum(record["holm_positive_inference"] for record in portability.values())
    no_large_regression = all(record["mean"] > -0.1 for record in portability.values())
    portability_pass = positive == 3 and holm_positive >= 2 and no_large_regression
    identity = frame[(frame["kind"] == "wp1") & (frame["arm"] == "merit_full") & (frame["domain"] == "identity")]
    identity_by_family: dict[str, Any] = {}
    identity_gate = gates["optical"]["identity_safety"]
    for family, group in identity.groupby("family"):
        summary = {
            "mean_absolute_output_change": float(group["mean_absolute_output_change"].mean()),
            "p99_absolute_output_change": float(group["p99_absolute_output_change"].quantile(0.99)),
            "saturation_fraction_increase": float(
                (group["output_saturation_fraction"] - group["input_saturation_fraction"]).mean()
            ),
            "preclip_out_of_range_fraction": float(group["preclip_out_of_range_fraction"].mean()),
        }
        summary["pass"] = bool(
            summary["mean_absolute_output_change"] <= identity_gate["mean_absolute_output_change_max_0to1"]
            and summary["p99_absolute_output_change"] <= identity_gate["p99_absolute_output_change_max_0to1"]
            and summary["saturation_fraction_increase"] <= identity_gate["saturation_fraction_increase_max"]
            and summary["preclip_out_of_range_fraction"] <= identity_gate["preclip_out_of_range_fraction_max"]
        )
        identity_by_family[family] = summary
    identity_pass = all(value["pass"] for value in identity_by_family.values()) and len(identity_by_family) == 3
    return {
        "metrics": metrics,
        "optical_macro_pass": optical_pass,
        "restoration_efficacy_floor_pass": efficacy_pass,
        "oversharpening_safety_pass": oversharpening_pass,
        "backbone_portability": portability,
        "backbone_portability_pass": portability_pass,
        "identity_by_family": identity_by_family,
        "identity_safety_pass": identity_pass,
    }


def summarize_prompt(contract: dict[str, Any], controlled: pd.DataFrame, gates: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in (0, 1, 2):
        marker = common.load_json(
            C009 / "evaluation" / "wp2_prompt" / f"wp1__nafnet__merit_full__seed{seed}" / "RUN_COMPLETE.json"
        )
        rows.extend(read_jsonl(Path(marker["output_path"])))
    frame = pd.DataFrame(rows)
    keys = ["model_id", "sample_id", "parent_sha256"]
    pivot = frame.pivot(index=keys, columns="prompt_condition", values="restored_psnr_db").reset_index()
    comparisons: dict[str, Any] = {}
    for condition in ("zero", "shuffled", "swapped", "constant", "mismatch_0.5", "mismatch_0.75", "mismatch_1.25", "mismatch_1.5"):
        name = f"correct_minus_{condition}"
        pivot[name] = pivot["correct"] - pivot[condition]
        comparisons[name] = cluster_bootstrap(pivot, name, "parent_sha256")
    rho_values: list[dict[str, Any]] = []
    for parent, group in frame.groupby("parent_sha256"):
        rho = spearmanr(group["prompt_absolute_error"], -group["restored_psnr_db"], nan_policy="omit").statistic
        if np.isfinite(rho):
            rho_values.append({"parent_sha256": parent, "rho": float(rho)})
    rho_frame = pd.DataFrame(rho_values)
    rho_summary = cluster_bootstrap(rho_frame, "rho", "parent_sha256")
    rho_summary["positive_parent_fraction"] = float((rho_frame["rho"] > 0).mean())
    prompt_gate = gates["prompt_reliance"]
    pass_value = bool(
        comparisons["correct_minus_zero"]["mean"] >= prompt_gate["primary"]["mean_delta_psnr_db_min"]
        and comparisons["correct_minus_zero"]["ci95_low"] > 0
        and comparisons["correct_minus_shuffled"]["ci95_low"] > 0
        and comparisons["correct_minus_swapped"]["ci95_low"] > 0
        and comparisons["correct_minus_constant"]["mean"] > 0
        and min(comparisons[name]["mean"] for name in comparisons if name.startswith("correct_minus_mismatch")) > 0
        and rho_summary["mean"] >= prompt_gate["dose_response"]["macro_rho_min"]
        and rho_summary["positive_parent_fraction"] >= prompt_gate["dose_response"]["parents_with_positive_rho_fraction_min"]
        and rho_summary["ci95_low"] > 0
    )
    zero_train = controlled[(controlled["kind"] == "followon") & (controlled["wp"] == "wp2") & (controlled["domain"] == "id")]
    informative = controlled[(controlled["kind"] == "wp1") & (controlled["family"] == "nafnet") & (controlled["arm"] == "merit_full") & (controlled["domain"] == "id")]
    merged = informative.merge(
        zero_train,
        on=["seed", "sample_id", "parent_sha256"],
        suffixes=("_informative", "_zero_train"),
        validate="one_to_one",
    )
    merged["delta_psnr_informative_minus_zero_train"] = merged["restored_psnr_db_informative"] - merged["restored_psnr_db_zero_train"]
    return {
        "perturbation_comparisons": comparisons,
        "dose_response": rho_summary,
        "informative_minus_zero_trained": cluster_bootstrap(
            merged, "delta_psnr_informative_minus_zero_train", "parent_sha256"
        ),
        "prompt_reliance_pass": pass_value,
        "claim_limit": prompt_gate["claim_limit"],
    }


def summarize_architecture(controlled: pd.DataFrame, gates: dict[str, Any]) -> dict[str, Any]:
    single = controlled[(controlled["kind"] == "wp1") & (controlled["family"] == "nafnet") & (controlled["arm"] == "merit_full") & (controlled["domain"] == "id")]
    output: dict[str, Any] = {}
    for structure in ("cascade", "parallel"):
        alternative = controlled[(controlled["kind"] == "followon") & (controlled["structure"] == structure) & (controlled["domain"] == "id")]
        keys = ["seed", "sample_id", "parent_sha256"]
        merged = alternative.merge(single, on=keys, suffixes=("_alternative", "_single"), validate="one_to_one")
        merged["delta_psnr"] = merged["restored_psnr_db_alternative"] - merged["restored_psnr_db_single"]
        merged["delta_lpips"] = merged["restored_lpips_alex_alternative"] - merged["restored_lpips_alex_single"]
        psnr = cluster_bootstrap(merged, "delta_psnr", "parent_sha256")
        lpips_value = cluster_bootstrap(merged, "delta_lpips", "parent_sha256")
        identity = controlled[
            (controlled["kind"] == "followon")
            & (controlled["structure"] == structure)
            & (controlled["domain"] == "identity")
        ]
        identity_gate = gates["optical"]["identity_safety"]
        identity_summary = {
            "mean_absolute_output_change": float(identity["mean_absolute_output_change"].mean()),
            "p99_absolute_output_change": float(identity["p99_absolute_output_change"].quantile(0.99)),
            "saturation_fraction_increase": float(
                (identity["output_saturation_fraction"] - identity["input_saturation_fraction"]).mean()
            ),
            "preclip_out_of_range_fraction": float(identity["preclip_out_of_range_fraction"].mean()),
        }
        identity_summary["pass"] = bool(
            identity_summary["mean_absolute_output_change"] <= identity_gate["mean_absolute_output_change_max_0to1"]
            and identity_summary["p99_absolute_output_change"] <= identity_gate["p99_absolute_output_change_max_0to1"]
            and identity_summary["saturation_fraction_increase"] <= identity_gate["saturation_fraction_increase_max"]
            and identity_summary["preclip_out_of_range_fraction"] <= identity_gate["preclip_out_of_range_fraction_max"]
        )
        passed = bool(
            psnr["mean"] >= 0.1
            and psnr["ci95_low"] > 0
            and lpips_value["ci95_high"] <= 0
            and identity_summary["pass"]
        )
        output[structure] = {
            "delta_psnr": psnr,
            "delta_lpips": lpips_value,
            "identity_safety": identity_summary,
            "pass": passed,
        }
    return output


def summarize_ood(
    controlled: pd.DataFrame,
    gates: dict[str, Any],
    identity_by_family: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    identity_safety_pass = bool(
        len(identity_by_family) == 3
        and all(record["pass"] for record in identity_by_family.values())
    )
    ood = controlled[(controlled["kind"] == "wp1") & (controlled["domain"] == "ood")]
    keys = ["family", "seed", "sample_id", "parent_sha256", "condition", "severity", "stochastic_seed"]
    metrics = [
        "restored_psnr_db",
        "restored_lpips_alex",
        "restored_gradient_mae",
        "restored_edge_normal_overshoot_p95",
        "degraded_edge_normal_overshoot_p95",
    ]
    full = ood[ood["arm"] == "merit_full"][keys + metrics].copy()
    base = ood[ood["arm"] == "base"][keys + metrics].copy()
    paired = full.merge(base, on=keys, suffixes=("_full", "_base"), validate="one_to_one")
    paired["delta_psnr"] = paired["restored_psnr_db_full"] - paired["restored_psnr_db_base"]
    paired["delta_lpips"] = paired["restored_lpips_alex_full"] - paired["restored_lpips_alex_base"]
    paired["delta_gradient"] = paired["restored_gradient_mae_full"] - paired["restored_gradient_mae_base"]
    paired["delta_overshoot_full_minus_degraded"] = (
        paired["restored_edge_normal_overshoot_p95_full"]
        - paired["degraded_edge_normal_overshoot_p95_full"]
    )
    result: dict[str, Any] = {}
    for (condition, severity), group in paired.groupby(["condition", "severity"]):
        psnr = cluster_bootstrap(group, "delta_psnr", "parent_sha256")
        lpips_value = cluster_bootstrap(group, "delta_lpips", "parent_sha256")
        gradient = cluster_bootstrap(group, "delta_gradient", "parent_sha256")
        overshoot = cluster_bootstrap(
            group, "delta_overshoot_full_minus_degraded", "parent_sha256"
        )
        passed = bool(
            psnr["mean"] >= 0.1
            and psnr["ci95_low"] > 0
            and lpips_value["mean"] <= -0.005
            and lpips_value["ci95_high"] < 0
            and gradient["ci95_high"] <= 0
            and overshoot["ci95_high"] <= 0.01
            and identity_safety_pass
        )
        result.setdefault(condition, {})[str(int(severity))] = {
            "delta_psnr": psnr,
            "delta_lpips": lpips_value,
            "delta_gradient": gradient,
            "delta_overshoot_full_minus_degraded": overshoot,
            "identity_safety_pass": identity_safety_pass,
            "pass": passed,
        }
    family_success = {family: all(result[family][severity]["pass"] for severity in ("1", "2")) for family in result}
    broad = all(family_success.values()) and sum(result[family]["3"]["pass"] for family in result) >= 4
    return {"by_family_severity": result, "family_level_success": family_success, "broad_six_family_claim_pass": broad}


def summarize_d2(contract: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for spec in common.wp1_specifications(contract):
        identity = common.model_id(spec)
        root = C009 / "evaluation" / "wp5_external_four_tunnel" / identity
        for fold in ("A", "B", "C", "D"):
            value = common.load_json(root / f"fold_{fold}.json")
            for detector, parents in value["detectors"].items():
                for parent, conditions in parents.items():
                    rows.append(
                        {
                            **spec,
                            "model_id": identity,
                            "tunnel": fold,
                            "detector": detector,
                            "parent": parent,
                            "delta_auprc_restored_minus_degraded": float(conditions["restored"]["pixel_auprc"]) - float(conditions["degraded"]["pixel_auprc"]),
                            "delta_f1_restored_minus_degraded": float(conditions["restored"]["f1_at_0_5"]) - float(conditions["degraded"]["f1_at_0_5"]),
                            "delta_matched_sensitivity_false_positive_pixel_rate": float(
                                value["matched_sensitivity_burden"][detector][parent][
                                    "delta_false_positive_pixel_rate_restored_minus_degraded"
                                ]
                            ),
                        }
                    )
    frame = pd.DataFrame(rows)
    full = frame[frame["arm"] == "merit_full"]
    parent = full.groupby(["detector", "tunnel", "parent"], as_index=False)[
        [
            "delta_auprc_restored_minus_degraded",
            "delta_matched_sensitivity_false_positive_pixel_rate",
        ]
    ].mean()
    macro = cluster_bootstrap(parent, "delta_auprc_restored_minus_degraded", "parent", strata="tunnel")
    burden = cluster_bootstrap(
        parent,
        "delta_matched_sensitivity_false_positive_pixel_rate",
        "parent",
        strata="tunnel",
    )
    detector_results: dict[str, Any] = {}
    for detector, group in parent.groupby("detector"):
        summary = cluster_bootstrap(group, "delta_auprc_restored_minus_degraded", "parent", strata="tunnel")
        summary["one_sided_sign_flip"] = one_sided_cluster_sign_flip(
            group,
            "delta_auprc_restored_minus_degraded",
            "parent",
            strata="tunnel",
            seed=BOOTSTRAP_SEED + len(detector_results),
        )
        tunnel_means = group.groupby("tunnel")["delta_auprc_restored_minus_degraded"].mean().to_dict()
        summary["tunnel_means"] = {key: float(value) for key, value in tunnel_means.items()}
        summary["positive_tunnels"] = sum(value > 0 for value in tunnel_means.values())
        summary["pass"] = bool(summary["mean"] > 0 and summary["ci95_low"] > 0 and summary["positive_tunnels"] >= 3)
        detector_results[detector] = summary
    apply_holm(detector_results)
    for summary in detector_results.values():
        summary["pass"] = bool(
            summary["mean"] > 0
            and summary["holm_positive_inference"]
            and summary["positive_tunnels"] >= 3
        )
    tunnel_macro = parent.groupby("tunnel")["delta_auprc_restored_minus_degraded"].mean().to_dict()
    crack_gate = gates["crack"]
    pass_value = bool(
        macro["mean"] >= crack_gate["macro_pass"]["mean_delta_auprc_min"]
        and macro["ci95_low"] > 0
        and sum(record["mean"] > 0 for record in detector_results.values()) == 3
        and sum(record["holm_positive_inference"] for record in detector_results.values()) >= 2
        and all(record["mean"] > -0.01 for record in detector_results.values())
        and sum(value > 0 for value in tunnel_macro.values()) >= 3
        and all(value > -0.02 for value in tunnel_macro.values())
        and burden["ci95_high"] <= 0
    )
    return {
        "macro": macro,
        "detectors": detector_results,
        "matched_sensitivity_false_positive_pixel_burden": burden,
        "matched_sensitivity_safety_pass": bool(burden["ci95_high"] <= 0),
        "tunnel_macro": {key: float(value) for key, value in tunnel_macro.items()},
        "external_crack_gate_pass": pass_value,
        "rows": len(frame),
    }


def summarize_width(contract: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for spec in common.wp1_specifications(contract):
        identity = common.model_id(spec)
        root = C009 / "evaluation" / "wp8_printed_width" / identity / "progress"
        for path in sorted(root.glob("W*.json")):
            payload = common.load_json(path)
            for measurement in payload["measurements"]:
                rows.append({**spec, "model_id": identity, **measurement})
    frame = pd.DataFrame(rows)
    key = ["family", "seed", "model_id", "source_id", "nominal_width_mm"]
    degraded = frame[frame["condition"] == "controlled_degraded"].copy()
    restored = frame[frame["condition"] == "restored"].copy()
    merged = restored.merge(degraded, on=key, suffixes=("_restored", "_degraded"), validate="one_to_one")
    eligible = merged[(merged["status_restored"] == "ESTIMABLE") & (merged["status_degraded"] == "ESTIMABLE")].copy()
    eligible["delta_absolute_error_restored_minus_degraded"] = eligible["absolute_error_mm_restored"] - eligible["absolute_error_mm_degraded"]
    full = eligible[eligible["arm_restored"] == "merit_full"]
    primary = cluster_bootstrap(full, "delta_absolute_error_restored_minus_degraded", "source_id")
    degraded_error = float(full["absolute_error_mm_degraded"].mean()) if len(full) else float("nan")
    relative_reduction = -float(primary["mean"]) / degraded_error if degraded_error > 0 else float("nan")
    width_means = full.groupby("nominal_width_mm")["delta_absolute_error_restored_minus_degraded"].mean()
    predeclared_non = {
        float(value) for value in contract["wp8_printed_width"]["predeclared_non_estimable_mm"]
    }
    estimable_width_means = width_means[
        ~width_means.index.astype(float).isin(predeclared_non)
    ]
    improved = int((estimable_width_means < 0).sum())
    required_improved = 6 if predeclared_non == {0.1, 0.2} else 7
    base = eligible[eligible["arm_restored"] == "base"]
    increment = full.merge(
        base,
        on=["family", "seed", "source_id", "nominal_width_mm"],
        suffixes=("_full", "_base"),
        validate="one_to_one",
    )
    increment["delta_absolute_error_full_minus_base"] = (
        increment["absolute_error_mm_restored_full"]
        - increment["absolute_error_mm_restored_base"]
    )
    increment_summary = cluster_bootstrap(
        increment, "delta_absolute_error_full_minus_base", "source_id"
    )
    full_all = merged[merged["arm_restored"] == "merit_full"]
    restored_non = float((full_all["status_restored"] != "ESTIMABLE").mean())
    degraded_non = float((full_all["status_degraded"] != "ESTIMABLE").mean())
    gate = gates["printed_width"]
    pass_value = bool(
        relative_reduction >= gate["primary_pass"]["relative_mean_absolute_error_reduction_min"]
        and primary["ci95_high"] < 0
        and improved >= required_improved
        and increment_summary["ci95_high"] <= 0
        and (restored_non - degraded_non) * 100 <= gate["non_estimable_safety"]["rate_increase_max_percentage_points"]
    )
    return {
        "primary_delta_absolute_error": primary,
        "relative_mean_absolute_error_reduction": relative_reduction,
        "nominal_width_delta_means": {str(key): float(value) for key, value in width_means.items()},
        "improved_estimable_nominal_widths": improved,
        "required_improved_estimable_nominal_widths": required_improved,
        "increment_over_base_delta_absolute_error": increment_summary,
        "restored_non_estimable_rate": restored_non,
        "degraded_non_estimable_rate": degraded_non,
        "printed_width_gate_pass": pass_value,
        "eligible_pairs": len(full),
    }


def summarize_d3(contract: dict[str, Any]) -> dict[str, Any]:
    legacy_root = C009.parent / "c006_execution" / "artifacts" / "d3_stream_v1" / "progress"
    raw: dict[str, dict[str, Any]] = {}
    for path in legacy_root.glob("*.json"):
        value = common.load_json(path)
        raw[value["cohort_id"]] = next(event for event in value["events"] if event["condition"] == "raw")
    rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    for spec in common.wp1_specifications(contract):
        identity = common.model_id(spec)
        root = C009 / "evaluation" / "wp6_d3_fullframe" / identity / "progress"
        for path in root.glob("*.json"):
            value = common.load_json(path)
            cohort_id = value["cohort_id"]
            restored = value["endpoint"]["endpoint_by_minimum_area"]["32"]
            baseline = raw[cohort_id]["endpoint_by_minimum_area"]["32"]
            rows.append(
                {
                    **spec,
                    "model_id": identity,
                    "cohort_id": cohort_id,
                    "label": value["consensus_label"],
                    "raw_any": int(bool(baseline["any_eligible_component"])),
                    "restored_any": int(bool(restored["any_eligible_component"])),
                    "raw_components_per_megapixel": float(baseline["components_per_megapixel"]),
                    "restored_components_per_megapixel": float(restored["components_per_megapixel"]),
                }
            )
            quality_rows.append(
                {
                    **spec,
                    "model_id": identity,
                    "cohort_id": cohort_id,
                    **{f"cell_{key}": val for key, val in value.get("cell", {}).items()},
                    **{f"delta_{key}": float(value["restored_quality"][key]) - float(raw_value) for key, raw_value in value["raw_quality"].items()},
                }
            )
    frame = pd.DataFrame(rows)
    summaries: dict[str, Any] = {}
    for identity, group in frame.groupby("model_id"):
        positive = group[group["label"] == "crack_visible"]
        negative = group[group["label"] == "no_visible_crack"]
        summaries[identity] = {
            "visible_sensitivity_raw": float(positive["raw_any"].mean()),
            "visible_sensitivity_restored": float(positive["restored_any"].mean()),
            "no_visible_positive_rate_raw": float(negative["raw_any"].mean()),
            "no_visible_positive_rate_restored": float(negative["restored_any"].mean()),
            "no_visible_components_per_megapixel_delta": float((negative["restored_components_per_megapixel"] - negative["raw_components_per_megapixel"]).mean()),
        }
    quality = pd.DataFrame(quality_rows)
    quality_metrics = [column for column in quality.columns if column.startswith("delta_")]
    quality_summary = {
        metric: float(quality[metric].mean()) for metric in quality_metrics
    }
    return {
        "models": summaries,
        "quality_descriptor_mean_deltas_all_models": quality_summary,
        "frames_per_model": int(frame.groupby("model_id").size().min()),
        "claim_boundary": contract["wp6_different_tunnel_same_system"]["claim_boundary"],
    }


def global_verdict(
    optical: dict[str, Any],
    prompt: dict[str, Any],
    d2: dict[str, Any],
    width: dict[str, Any],
) -> str:
    if (
        not optical["optical_macro_pass"]
        or not optical["restoration_efficacy_floor_pass"]
        or not optical["identity_safety_pass"]
        or not optical["oversharpening_safety_pass"]
    ):
        return "NOT_SUPPORTED"
    if not optical["backbone_portability_pass"]:
        return "ARCHITECTURE_DEPENDENT"
    if not d2["external_crack_gate_pass"] or not width["printed_width_gate_pass"]:
        return "OPTICAL_ONLY"
    if not prompt["prompt_reliance_pass"]:
        return "OPTICAL_CRACK_WIDTH_SUPPORTED_PROMPT_NOT_SUPPORTED"
    return "FULL_MERIT_SUPPORTED"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    contract = common.load_contract()
    queue = common.load_json(C009 / "runs" / "queue" / "full_evaluation_queue_status_v2.json")
    if queue.get("status") != "complete" or int(queue.get("completed_item_count", -1)) != 84:
        raise common.EvaluationError("Full evaluation queue is not complete")
    gates = common.load_json(C009 / "protocol" / "g12_endpoint_gate_freeze_v1.json")
    controlled = controlled_frame(contract)
    optical = summarize_optical(controlled, gates)
    prompt = summarize_prompt(contract, controlled, gates)
    architecture = summarize_architecture(controlled, gates)
    ood = summarize_ood(controlled, gates, optical["identity_by_family"])
    d2 = summarize_d2(contract, gates)
    width = summarize_width(contract, gates)
    d3 = summarize_d3(contract)
    verdict = global_verdict(optical, prompt, d2, width)
    result: dict[str, Any] = {
        "schema_version": "MERIT-C009-FINAL-SUMMARY-v3",
        "status": "PASS_COMPLETE_PREDECLARED_ANALYSIS",
        "contract_sha256": common.EXPECTED_CONTRACT_SHA256,
        "g12_sha256": common.sha256_file(C009 / "protocol" / "g12_endpoint_gate_freeze_v1.json"),
        "global_verdict": verdict,
        "optical": optical,
        "prompt": prompt,
        "architecture": architecture,
        "ood": ood,
        "external_four_tunnel_crack": d2,
        "printed_width": width,
        "different_tunnel_same_system": d3,
        "mandatory_context": (
            "Prior C006/C007 negative and mixed outcomes remain reportable and are not "
            "superseded or deleted by this C009 analysis."
        ),
    }
    result["payload_sha256"] = hashlib.sha256(common.canonical_bytes(result)).hexdigest()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "c009_final_summary_v3.json"
    common.write_json_exclusive(output, result)
    print(json.dumps({"status": result["status"], "global_verdict": verdict, "path": str(output), "sha256": common.sha256_file(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
