"""Recalculate the confirmation contrasts with equal weight for each tunnel."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parent
with (ROOT / "confirmation_rows.csv").open(newline="") as source:
    rows = list(csv.DictReader(source))
if len(rows) != 336:
    raise ValueError("Expected 336 factorial rows")
cells = defaultdict(list)
for row in rows:
    cells[row["backbone"], row["tunnel"]].append(row)
if len(cells) != 16:
    raise ValueError("Expected four backbones and four tunnels")
raw, active = {}, {}
for backbone in sorted({key[0] for key in cells}):
    for column, means in [("raw", raw), ("framework", active)]:
        means[backbone] = fmean(
            fmean(float(row[column]) for row in cell)
            for (name, tunnel), cell in cells.items() if name == backbone
        )
marginal = [(raw[name] + active[name]) / 2 for name in active]
gain = fmean(active[name] - raw[name] for name in active)
active_range = max(active.values()) - min(active.values())
marginal_range = max(marginal) - min(marginal)
estimates = {
    "framework_main": gain,
    "max_backbone_main": marginal_range,
    "factorial_dominance": gain - marginal_range,
    "max_backbone_conditional": active_range,
    "conditional_sensitivity_dominance": gain - active_range,
}
frozen = json.loads((ROOT / "estimates.json").read_text())
for name, value in estimates.items():
    if not math.isclose(value, frozen[name]["estimate"], rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"{name} differs from the archived estimate")
    print(f"{name}: {value:+.9f}")
