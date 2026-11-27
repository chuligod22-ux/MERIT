"""Recompute MERIT detection, ablation and severity point estimates from CSV."""
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import correlation, fmean

ROOT = Path(__file__).resolve().parent
SCORES = ("raw", "fixed", "selected")


def read(name):
    with (ROOT / f"{name}.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def groups(rows, *keys):
    result = defaultdict(list)
    for row in rows:
        result[tuple(row[key] for key in keys)].append(row)
    return result


def average(rows, metric):
    return fmean(float(row[metric]) for row in rows)


def field_mean(rows, metric):
    spans = groups(rows, "tunnel", "span_id")
    tunnels = defaultdict(list)
    for (tunnel, span), part in spans.items():
        tunnels[tunnel].append(average(part, metric))
    return fmean(fmean(values) for values in tunnels.values())


def source_mean(rows, metric):
    return fmean(average(part, metric) for part in groups(rows, "source_id").values())


def check(actual, expected, label):
    if not math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=2e-8):
        raise ValueError(f"{label}: recomputed {actual}, archived {expected}")


def unique(rows, *keys):
    if len(rows) != len(groups(rows, *keys)):
        raise ValueError(f"Duplicate observations for {keys}")


def ranks(values):
    indices = defaultdict(list)
    for index, value in enumerate(sorted(values), 1):
        indices[value].append(index)
    return [fmean(indices[value]) for value in values]


def audit_routing(routing):
    unique(routing, "panel", "seed", "span_id", "backbone", "detector")
    panels = groups(routing, "panel")
    for ref in read("routing_summary"):
        panel, group, tunnel = (ref[k] for k in ("panel", "group", "tunnel"))
        rows = panels[(panel,)]
        if group == "without_vgg":
            rows = [q for q in rows if q["detector"] != "d2_vgg16_unet"]
        elif group != "panel":
            rows = [q for q in rows if group == q["backbone"] + "/" + q["detector"]]
        if tunnel != "ALL":
            rows = [q for q in rows if q["tunnel"] == tunnel]
        values = {k: field_mean(rows, k) for k in SCORES}
        for a, b in (("fixed", "raw"), ("selected", "raw"), ("selected", "fixed")):
            values[f"{a}_minus_{b}"] = values[a] - values[b]
        for k, v in values.items():
            check(v, ref[k], f"{panel}/{group}/{tunnel}/{k}")
        if group == "panel" and tunnel == "ALL":
            print(f"{panel:10} AUPRC change  selected {values['selected_minus_raw']:+.6f}  "
                  f"fixed {values['fixed_minus_raw']:+.6f}")


def audit_ablation(routing):
    controlled, field = read("ablation_controlled"), read("ablation_field")
    unique(controlled, "seed", "sample_id", "arm")
    unique(field, "seed", "span_id", "detector", "arm")
    data = {"controlled": controlled, "field": field}
    for ref in read("ablation_summary"):
        domain, metric = ref["domain"], ref["metric"]
        values = []
        for arm in ref["contrast"].split("-"):
            rows = [q for q in data[domain] if q["arm"] == arm
                    and (ref["seed"] == "all" or q["seed"] == ref["seed"])
                    and (ref["tunnel"] == "all" or q["tunnel"] == ref["tunnel"])]
            estimator = field_mean if domain == "field" else source_mean
            values.append(estimator(rows, metric))
        value = values[0] if len(values) == 1 else values[0] - values[1]
        check(value, ref["estimate"], tuple(ref[k] for k in ("domain","metric","seed","tunnel","contrast")))
        if ref["seed"] == ref["tunnel"] == "all" and len(values) == 1:
            print(f"ablation {domain:10} {metric:5} {ref['contrast']:10} {value:.6f}")
    previous = {(q["seed"], q["span_id"], q["detector"]): q for q in routing if q["panel"] == "native"}
    matched = [q for q in field if q["seed"] in ("1", "2")]
    for q in matched:
        if q["arm"] == "raw":
            check(q["auprc"], previous[q["seed"], q["span_id"], q["detector"]]["raw"], "paired raw")
    full = field_mean([q for q in matched if q["arm"] == "full"], "auprc")
    raw = field_mean(list(previous.values()), "raw")
    fixed = field_mean(list(previous.values()), "fixed")
    print(f"matched seeds 1 and 2  full {full:.6f}  raw {raw:.6f}  fixed {fixed:.6f}")


def audit_dose():
    dose = read("dose_scores")
    unique(dose, "level", "seed", "span_id", "detector")
    for ref in read("dose_summary"):
        rows = [q for q in dose if ref["level"] == q["level"]
                and (ref["tunnel"] == "all" or ref["tunnel"] == q["tunnel"])]
        check(len(groups(rows, "span_id")), ref["spans"], "dose spans")
        values = {k: field_mean(rows, k) for k in SCORES}
        for arm in ("selected", "fixed"):
            values[arm + "_delta"] = values[arm] - values["raw"]
        for k, v in values.items():
            check(v, ref[k], f"dose/{ref['level']}/{ref['tunnel']}/{k}")
        if ref["tunnel"] == "all":
            print(f"dose {ref['level']:7}  selected {values['selected_delta']:+.6f}  "
                  f"fixed {values['fixed_delta']:+.6f}")


def audit_detail(routing):
    detail = {q["span_id"]: q for q in read("span_detail")}
    for ref in read("detail_bands"):
        rows = [q for q in routing if q["panel"] == ref["panel"]
                and detail[q["span_id"]]["detail_band"] == ref["band"]]
        spans = set(q["span_id"] for q in rows)
        check(len(spans), ref["spans"], "detail span count")
        check(len(groups(rows, "tunnel")), ref["tunnels"], "detail tunnel count")
        check(sum(int(detail[s]["tiles"]) for s in spans), ref["tiles"], "detail tile count")
        values = {k: field_mean(rows, k) for k in SCORES}
        for arm in ("selected", "fixed"):
            values[arm + "_delta"] = values[arm] - values["raw"]
        for k, v in values.items():
            check(v, ref[k], "detail band " + k)
    for ref in read("baseline_association"):
        rows = [q for q in routing if q["panel"] == ref["panel"]]
        units = []
        for (tunnel, span), part in groups(rows, "tunnel", "span_id").items():
            raw = average(part, "raw")
            units.append(dict(tunnel=tunnel, raw=raw, delta=average(part, "selected") - raw))
        rho = correlation(ranks([u["raw"] for u in units]), ranks([u["delta"] for u in units]))
        check(rho, ref["raw_delta_spearman"], "baseline rank correlation")
        centered_raw, centered_delta = [], []
        for part in groups(units, "tunnel").values():
            for key, centered in (("raw", centered_raw), ("delta", centered_delta)):
                center = average(part, key)
                centered.extend(q[key] - center for q in part)
        check(correlation(centered_raw, centered_delta), ref["within_tunnel_pearson"], "within-tunnel r")


def audit_inversion():
    rows = read("inversion_scores")
    unique(rows, "family", "severity", "setting", "crop_id")
    blocks = groups(rows, "family", "severity", "setting")
    for ref in read("inversion_summary"):
        key = tuple(ref[k] for k in ("family", "severity", "setting"))
        for metric in ("raw_psnr", "restored_psnr", "delta_psnr", "delta_lpips", "plateau"):
            value = source_mean(blocks[key], metric)
            check(value, ref[metric], key + (metric,))
            if metric == "delta_psnr" and key[1:] == ("1", "unknown_medium"):
                print(f"mild input, fixed-middle inverse  {key[0]:33} {value:+.3f} dB")


def audit_backbones():
    rows = read("backbone_source_means")
    unique(rows, "model", "level", "source_id")
    blocks = groups(rows, "model", "level")
    for ref in read("backbone_summary"):
        model, level = ref["model"], ref["level"]
        part = blocks[model, level]
        low = {q["source_id"]: q for q in blocks[model, "low"]}
        check(len(part), ref["parents"], model + " source count")
        for metric, key in (("gain_db", "gain_minus_low_db"), ("delta_lpips", "lpips_minus_low")):
            check(average(part, metric), ref[metric], model + "/" + metric)
            diff = fmean(float(q[metric]) - float(low[q["source_id"]][metric]) for q in part)
            check(diff, ref[key], model + "/" + key)
        reduction = 100 * (1 - 10 ** (-average(part, "gain_db") / 10))
        check(reduction, ref["geometric_mse_reduction_pct"], model + " geometric MSE reduction")


def audit_margin():
    rows = read("margin_conditions")
    unique(rows, "condition_id")
    for ref in read("margin_summary"):
        part = [q for q in rows if ref["band"] in ("all", q["band"])]
        raw, restored = (average(part, k) for k in ("raw", "restored"))
        margins = [float(q["optical_margin_cy_px"]) for q in part]
        recomputed = dict(conditions=len(part), frames=sum(int(q["frames"]) for q in part),
                          raw=raw, restored=restored, recovery=100 * (1-restored/raw),
                          min_margin=min(margins), max_margin=max(margins))
        for k, value in recomputed.items():
            check(value, ref[k], "optical margin " + k)


if __name__ == "__main__":
    routing = read("routing_scores")
    audit_routing(routing)
    audit_ablation(routing)
    audit_dose()
    audit_detail(routing)
    audit_inversion()
    audit_backbones()
    audit_margin()
    print("All archived point estimates passed.")
