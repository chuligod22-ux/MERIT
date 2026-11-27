"""Independently reconstruct all fold choices and held errors from the frame records."""
import csv
import gzip
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


BASE = Path(__file__).resolve().parent
GRID = list(itertools.product((-.75, -.5, .5, .75), (0., .75), (4., 8.)))
THRESHOLDS = (0., .0005, .001, .002, .004, .008, .016, math.inf)


def verify(folder):
    chart = defaultdict(lambda: defaultdict(list))
    bypass_counts = defaultdict(int)
    coverage = defaultdict(lambda: [0, 0, 0, 0])
    with gzip.open(folder / "measurements.jsonl.gz", "rt", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    assert len(records) == 479
    for rec in records:
        cell = rec["frame_id"].split("/")[0]
        row = chart[cell]
        raw = rec["raw"]
        valid = [i for i, v in enumerate(raw) if v is not None]
        if valid:
            row["margin_raw"].append(mean(raw[i] for i in valid))
        assert len(rec["raw"]) == len(rec["roi_ids"]) == len(set(rec["roi_ids"]))
        assert {(seed, idx) for seed, idx, _, _ in rec["routes"]} == set(itertools.product(range(3), range(16)))
        for seed, idx, bypass, scores in rec["routes"]:
            assert len(scores) == len(rec["roi_ids"]) and isinstance(bypass, bool)
            bypass_counts[cell, seed, idx] += int(bypass)
        common = [i for i in valid if all(item[3][i] is not None for item in rec["routes"])]
        counts = coverage[cell]
        counts[0] += len(common)
        counts[1] += int(bool(common))
        counts[2] += len(raw)
        counts[3] += 1
        if not common:
            continue
        baseline = mean(raw[i] for i in common)
        row["raw"].append(baseline)
        for seed, idx, bypass, vals in rec["routes"]:
            forced = mean(vals[i] for i in common)
            row["safe",seed,idx].append(baseline if bypass else forced)
            row["force",seed,idx].append(forced)
    cells = {c: {k:median(v) for k,v in row.items()} for c,row in chart.items()}
    refs = {c.split("_")[1]:values for c,values in cells.items() if c.startswith("60km_") and c.endswith("_ISO100")}
    eligible, margins, errors = {}, {}, {}
    for c, values in cells.items():
        anchor = refs[c.split("_")[1]]
        eligible[c] = values is not anchor
        margins[c] = anchor["margin_raw"] - values["margin_raw"]
        errors[c] = {k:abs(v-anchor["raw"]) for k,v in values.items()}

    def fit(train, seed, gated):
        candidates = []
        for idx, (a,s,b) in enumerate(GRID):
            for t in THRESHOLDS if gated else [-math.inf]:
                active = [c for c in train if not gated or (eligible[c] and margins[c] > t)]
                losses = [errors[c]["safe", seed, idx] if c in active else errors[c]["raw"] for c in train]
                candidates.append((mean(losses), len(active), abs(a), b, s, a, -t, idx))
        *_, neg_t, idx = min(candidates)
        return idx, -neg_t

    with (folder / "folds.csv").open() as stream:
        folds = list(csv.DictReader(stream))
    assert len(folds) == 150
    worst = 0.
    for row in folds:
        c, seed = row["condition"], int(row["seed"])
        train = [x for x in cells if x != c and (eligible[c] or x.split("_")[1] != c.split("_")[1])]
        idx,t = fit(train, seed, True)
        j,_ = fit(train, seed, False)
        assert len(train) == int(row["train_n"])
        assert (idx,t,j) == (int(row["candidate"]),float(row["threshold"]),int(row["always_candidate"])), (c,seed)
        assert int(row["safety_bypass_frames"]) == bypass_counts[c, seed, idx]
        assert all(abs(v - float(row[k])) < 1e-11 for v, k in zip(GRID[idx], ("alpha", "sigma", "cap")))
        assert all(abs(v - float(row["always_" + k])) < 1e-11 for v, k in zip(GRID[j], ("alpha", "sigma", "cap")))
        assert abs(margins[c] - float(row["raw_margin"])) < 1e-11
        assert abs(refs[c.split("_")[1]]["raw"] - float(row["target_mtf50"])) < 1e-11
        assert (not eligible[c]) == (row["reference"] == "True")
        e = errors[c]
        active = eligible[c] and margins[c] > t
        assert active == (row["active"] == "True")
        values = dict(raw=e["raw"], always=e["safe",seed,idx], unguarded=e["force",seed,idx],
                      gated=e["safe",seed,idx] if active else e["raw"], always_tuned=e["safe",seed,j])
        worst = max(worst, max(abs(v-float(row[k])) for k,v in values.items()))
    assert worst < 1e-11
    verify_tables(folder, folds, coverage)
    print(folder.name, len(folds), "fold choices and held error values verified", worst, flush=True)


def verify_tables(folder, folds, coverage):
    import numpy as np

    def read_rows(filename):
        with (folder / filename).open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    def close(value, expected):
        assert abs(value - float(expected)) < 1e-11, (folder.name, value, expected)

    policies = ("raw", "gated", "always", "unguarded", "always_tuned")
    grouped = defaultdict(list)
    for row in folds:
        grouped[row["condition"]].append(row)
    assert all({int(row["seed"]) for row in group} == {0, 1, 2} for group in grouped.values())
    assert len(coverage) == len(grouped) == 50
    counts = read_rows("coverage.csv")
    assert {row["condition"] for row in counts} == set(grouped)
    count_fields = ("paired_rois", "paired_frames", "available_rois", "available_frames")
    for row in counts:
        assert [int(row[key]) for key in count_fields] == coverage[row["condition"]]
    metadata = json.loads((folder / "checks.json").read_text(encoding="utf-8"))
    assert sum(v[0] for v in coverage.values()) == metadata["common_rois"]
    assert sum(v[1] for v in coverage.values()) == metadata["frames"]
    assert sum(v[2] for v in coverage.values()) == metadata["frozen_rois"]
    close(min(v[0]/v[2] for v in coverage.values()), metadata["minimum_cell_coverage"])
    for row in read_rows("conditions.csv"):
        group = grouped[row["condition"]]
        for policy in policies:
            close(mean(float(f[policy]) for f in group), row[policy])
        assert sum(f["active"] == "True" for f in group) == int(row["active_seeds"])
        for key in ("raw", "always"):
            close(float(row["gated"]) - float(row[key]), row["gate_minus_" + key])
        assert [int(row[key]) for key in count_fields] == coverage[row["condition"]]

    summaries = read_rows("summary.csv")
    splits = defaultdict(list)
    for summary in summaries:
        splits[summary["scope"], summary["seed"]].append(summary)
    assert len(splits) == 12 and len(summaries) == 120
    for (scope, seed), rows in splits.items():
        means = {key: [] for key in policies}
        for condition, group in sorted(grouped.items()):
            reference = group[0]["reference"] == "True"
            if (scope == "reference5" and not reference) or (scope == "nonreference45" and reference):
                continue
            select = [f for f in group if seed == "all" or f["seed"] == seed]
            for key in policies:
                means[key].append(mean(float(f[key]) for f in select))
        means = {key: np.array(val) for key, val in means.items()}
        size = len(means["raw"])
        assert size == {"all50": 50, "reference5": 5, "nonreference45": 45}[scope]
        indices = np.random.default_rng(20261126).integers(size, size=(10000, size))
        for row in rows:
            pair = row["contrast"].split("-")
            delta = means[pair[0]] - means[pair[1] if len(pair) == 2 else "raw"]
            values = delta if len(pair) == 2 else means[pair[0]]
            lo, hi = np.quantile(values[indices].mean(axis=1), [0.025, 0.975])
            for result, key in [(values.mean(), "mean"), (lo, "lower"), (hi, "upper")]:
                close(result, row[key])
            for mask, key in [(delta < -1e-12, "improved"), (abs(delta) <= 1e-12, "tied"),
                              (delta > 1e-12, "worsened")]:
                assert int(mask.sum()) == int(row[key])
            if scope == "all50" and seed == "all" and row["contrast"] == "gated-always":
                print(f"{folder.name}: gated minus always {values.mean():.6f} cycles/pixel "
                      f"(95% condition-bootstrap CI {lo:.6f} to {hi:.6f})")
    print(f"{folder.name}: coverage, condition means and {len(summaries)} summary rows verified.")


if __name__ == "__main__":
    for directory in [BASE / "precursor", BASE / "final"]:
        verify(directory)
