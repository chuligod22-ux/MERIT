"""Recreate the published percentile bootstrap draws with NumPy's PCG64."""
import sys
from collections import defaultdict

import numpy as np

sys.dont_write_bytecode = True
from verify import SCORES, average, check, groups, read


def draws(blocks, seed):
    """Resample units inside each stratum, then weight strata equally."""
    rng = np.random.default_rng(seed)
    sampled = []
    for key in sorted(blocks):
        values = np.array(blocks[key], dtype=float)
        indices = rng.integers(len(values), size=(10000, len(values)))
        sampled.append(values[indices].mean(axis=1))
    return np.mean(sampled, axis=0)


def endpoints(samples, ref, prefix):
    lo, hi = np.quantile(samples, [0.025, 0.975])
    check(lo, ref[prefix + "lo"], "bootstrap lower " + prefix)
    check(hi, ref[prefix + "hi"], "bootstrap upper " + prefix)


def stratified(rows, metrics):
    strata = defaultdict(list)
    for (tunnel, span), part in sorted(groups(rows, "tunnel", "span_id").items()):
        strata[tunnel].append([average(part, k) for k in metrics])
    return strata


def routing_and_dose():
    routing = read("routing_scores")
    for ref in read("routing_summary"):
        if ref["tunnel"] != "ALL":
            continue
        group = ref["group"]
        rows = [q for q in routing if q["panel"] == ref["panel"]]
        if group == "without_vgg":
            rows = [q for q in rows if q["detector"] != "d2_vgg16_unet"]
        elif group != "panel":
            rows = [q for q in rows if group == q["backbone"] + "/" + q["detector"]]
        raw, fixed, selected = draws(stratified(rows, SCORES), 20261056).T
        for name, values in (("fixed_minus_raw_", fixed-raw), ("selected_minus_raw_", selected-raw),
                             ("selected_minus_fixed_", selected-fixed)):
            endpoints(values, ref, name)
    dose = groups(read("dose_scores"), "level")
    for ref in read("dose_summary"):
        if ref["tunnel"] != "all":
            continue
        raw, fixed, selected = draws(stratified(dose[(ref["level"],)], SCORES), 20261127).T
        for name, values in (("raw",raw), ("fixed",fixed), ("selected",selected),
                             ("fixed_delta",fixed-raw), ("selected_delta",selected-raw)):
            endpoints(values, ref, name + "_")


def ablation():
    data = {"controlled": read("ablation_controlled"), "field": read("ablation_field")}
    refs = [q for q in read("ablation_summary") if q["tunnel"] == "all"]
    for (domain, metric, seed), ref_rows in groups(refs, "domain", "metric", "seed").items():
        arms = sorted({q["arm"] for q in data[domain]})
        rows = [q for q in data[domain] if seed == "all" or seed == q["seed"]]
        for q in rows:
            if domain == "controlled":
                q["tunnel"], q["span_id"] = "controlled", q["source_id"]
        strata = defaultdict(list)
        for (tunnel, span), part in sorted(groups(rows, "tunnel", "span_id").items()):
            by_arm = groups(part, "arm")
            strata[tunnel].append([average(by_arm[(arm,)], metric) for arm in arms])
        samples = dict(zip(arms, draws(strata, 20261056).T))
        for ref in ref_rows:
            contrast = ref["contrast"].split("-")
            values = samples[contrast[0]]
            if len(contrast) == 2:
                values = values - samples[contrast[1]]
            endpoints(values, ref, "")


def severity():
    metrics = ("raw_psnr", "restored_psnr", "delta_psnr", "delta_lpips", "plateau")
    inversion = groups(read("inversion_scores"), "family", "severity", "setting")
    for ref in read("inversion_summary"):
        key = tuple(ref[k] for k in ("family", "severity", "setting"))
        sources = sorted(groups(inversion[key], "source_id").items())
        values = [[average(part, metric) for metric in metrics] for source, part in sources]
        samples = draws({1: values}, 20261127).T
        for metric, sample in zip(metrics, samples):
            endpoints(sample, ref, metric + "_")
    backbones = groups(read("backbone_source_means"), "model", "level")
    for ref in read("backbone_summary"):
        model, level = ref["model"], ref["level"]
        part = sorted(backbones[model, level], key=lambda q: q["source_id"])
        low = {q["source_id"]: float(q["gain_db"]) for q in backbones[model, "low"]}
        values = [float(q["gain_db"])-low[q["source_id"]] for q in part]
        endpoints(draws({1: values}, 20261127), ref, "gain_minus_low_")
    conditions = sorted(read("margin_conditions"), key=lambda q: q["condition_id"])
    for ref in read("margin_summary"):
        band = ref["band"]
        rows = [q for q in conditions if band in ("all", q["band"])]
        values = [[q["raw"],q["restored"]] for q in rows]
        seed = 20261033 if band == "all" else 20261127
        raw, restored = draws({1: values}, seed).T
        endpoints(100*(1-restored/raw), ref, "")


if __name__ == "__main__":
    routing_and_dose()
    ablation()
    severity()
    print("All archived bootstrap endpoints passed.")
