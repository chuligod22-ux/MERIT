"""Recompute the acquired-image checks from the public numerical tables."""
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


HERE = Path(__file__).resolve().parent
FEATURES = ("amplitude_dn", "snr", "background_dn", "mask_contrast_dn",
            "reference_width_px")


def read_csv(name):
    with (HERE / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def optional_number(text):
    return float(text) if text else None


def average_seeds(values):
    usable = [v for v in values if v is not None]
    return mean(usable) if len(usable) >= 2 else None


def get_regions():
    records = read_csv("regions.csv")
    for row in records:
        for name in FEATURES + ("closed_mask_width_px", "raw_width_px"):
            row[name] = optional_number(row[name])
        raw = row["raw_width_px"]
        direct = [optional_number(row[f"processed_seed{seed}_px"]) for seed in range(3)]
        row["direct"] = average_seeds(direct)
        row["fallback"] = raw is not None and None in direct
        policy = average_seeds([raw if v is None else v for v in direct])
        row["policy"] = policy
        if raw is not None and policy is not None:
            reference = row["reference_width_px"]
            row["delta"] = abs(policy - reference) - abs(raw - reference)
    return records


def group_means(rows, keys, value):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row[value])
    return [dict(zip(keys, key), **{value: mean(values)}) for key, values in groups.items()]


def span_deltas(rows):
    tiles = group_means(rows, ["tunnel", "span", "tile"], "delta")
    return group_means(tiles, ["tunnel", "span"], "delta")


def hierarchical_delta(rows):
    tunnels = group_means(span_deltas(rows), ["tunnel"], "delta")
    return mean(r["delta"] for r in tunnels)


def bootstrap_interval(pairs):
    spans = span_deltas(pairs)
    grouped = defaultdict(dict)
    for row in spans:
        grouped[row["tunnel"]][row["span"]] = row["delta"]
    draws = read_csv("bootstrap_span_counts.csv")
    estimates = []
    for draw in draws:
        if set(draw) != {r["span"] for r in spans}:
            raise ValueError("Bootstrap span IDs differ from the populated spans")
        tunnel_estimates = []
        for group in grouped.values():
            counts = {s: int(draw[s]) for s in group}
            if min(counts.values()) < 0 or sum(counts.values()) != len(group):
                raise ValueError("Invalid within-tunnel resampling multiplicities")
            tunnel_estimates.append(sum(counts[s] * group[s] for s in group) / len(group))
        estimates.append(mean(tunnel_estimates))
    if len(estimates) != 10000:
        raise ValueError("Expected 10,000 stratified replicates")
    estimates.sort()
    def percentile(p):
        position = (len(estimates) - 1) * p
        lower = math.floor(position)
        return estimates[lower] + (position - lower) * (estimates[lower + 1] - estimates[lower])
    return [percentile(.025), percentile(.975)]


def width_band(reference):
    if reference < 1:
        return "<1"
    if reference < 2:
        return "1-<2"
    if reference < 5:
        return "2-<5"
    return ">=5"


def width_report():
    rows = get_regions()
    if len({r["region"] for r in rows}) != len(rows):
        raise ValueError("Duplicate region IDs")
    pairs = [r for r in rows if "delta" in r]
    unpaired = [r for r in rows if "delta" not in r]
    thin = [r for r in pairs if r["reference_width_px"] < 5]
    wide = [r for r in pairs if r["reference_width_px"] >= 5]
    central = [r for r in pairs if r["raw_peak_location"] == "central"]
    closed = [r["closed_mask_width_px"] for r in thin if r["closed_mask_width_px"] is not None]
    band_results = {}
    for label in ("<1", "1-<2", "2-<5", ">=5"):
        group = [r for r in rows if width_band(r["reference_width_px"]) == label]
        kept = [r for r in group if "delta" in r]
        band_results[label] = {"total": len(group), "paired": len(kept),
                               "region_mean_delta_px": mean(r["delta"] for r in kept)}
    tunnel_results = {}
    for tunnel in sorted({r["tunnel"] for r in rows}):
        group = [r for r in rows if r["tunnel"] == tunnel]
        kept = [r for r in group if "delta" in r]
        tunnel_results[tunnel] = {"total": len(group), "paired": len(kept),
                                  "hierarchical_delta_px": hierarchical_delta(kept)}
    return {
        "regions": len(rows), "policy_pairs": len(pairs), "unpaired": len(unpaired),
        "direct_pairs": sum(r["direct"] is not None for r in pairs),
        "raw_nonestimable": sum(r["raw_width_px"] is None for r in rows),
        "policy_nonestimable": sum(r["policy"] is None for r in rows),
        "any_seed_fallback": sum(r["fallback"] for r in rows),
        "raw_profile_status_counts": dict(Counter(r["raw_profile_status"] for r in rows)),
        "hierarchical_delta_px": hierarchical_delta(pairs),
        "span_bootstrap_95pct_ci_px": bootstrap_interval(pairs),
        "width_bands": band_results, "tunnels": tunnel_results,
        "feature_medians": {label: {f: median(r[f] for r in part) for f in FEATURES}
                            for label, part in (("paired", pairs), ("unpaired", unpaired))},
        "raw_peak_locations": {label: dict(Counter(r["raw_peak_location"] for r in part))
                               for label, part in (("<5", thin), (">=5", wide))},
        "central_peak_pairs": len(central),
        "central_peak_wide": sum(r["reference_width_px"] >= 5 for r in central),
        "central_peak_hierarchical_delta_px": hierarchical_delta(central),
        "closing_thin_pairs": {"total": len(thin), "remained_below_5": sum(x < 5 for x in closed),
                               "at_least_5": sum(x >= 5 for x in closed),
                               "unmeasurable": len(thin) - len(closed),
                               "median_original_width_px": median(r["reference_width_px"] for r in thin),
                               "median_measurable_closed_width_px": median(closed)}
    }


def sampling_report():
    rows = read_csv("sampling_spans.csv")
    arms = ("raw_auprc", "fixed_base_auprc", "fixed_full_auprc", "selected_auprc")
    for r in rows:
        for arm in arms:
            r[arm] = float(r[arm])
        r["tile_count"] = int(r["tile_count"])
        r["positive_percent"] = 100 * int(r["positive_pixels"]) / int(r["total_pixels"])
    overall = {arm: mean(g[arm] for g in group_means(rows, ["tunnel"], arm)) for arm in arms}
    strata = {}
    for design in sorted({r["sampling"] for r in rows}):
        group = [r for r in rows if r["sampling"] == design]
        percents = [r["positive_percent"] for r in group]
        strata[design] = {
            "spans_by_tunnel": dict(Counter(r["tunnel"] for r in group)),
            "spans": len(group), "tiles": sum(r["tile_count"] for r in group),
            "min_span_positive_percent": min(percents), "max_span_positive_percent": max(percents),
            "equal_span_raw_auprc": mean(r["raw_auprc"] for r in group),
            "equal_span_fixed_base_minus_raw": mean(r["fixed_base_auprc"] - r["raw_auprc"] for r in group),
            "equal_span_selected_minus_raw": mean(r["selected_auprc"] - r["raw_auprc"] for r in group)
        }
    return {"equal_tunnel_auprc": overall,
            "equal_tunnel_minus_raw": {a: overall[a] - overall["raw_auprc"] for a in arms[1:]},
            "sampling_strata": strata}


if __name__ == "__main__":
    print(json.dumps({"natural_width": width_report(), "field_detection": sampling_report()},
                     indent=2, allow_nan=False))
