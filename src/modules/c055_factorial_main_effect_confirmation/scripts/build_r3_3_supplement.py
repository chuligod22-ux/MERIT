"""Build auditable parent- and tunnel-level summaries for the R3-3 response."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
SOURCE = RUNS / "confirmation_rows.csv"
PARENT_OUT = RUNS / "r3_3_parent_summary.csv"
TUNNEL_OUT = RUNS / "r3_3_tunnel_summary.csv"


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    with SOURCE.open(encoding="utf-8", newline="") as stream:
        source_rows = list(csv.DictReader(stream))

    if len(source_rows) != 336:
        raise RuntimeError(f"expected 336 C055 confirmation rows, found {len(source_rows)}")

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        grouped[(row["tunnel"], row["parent"])].append(row)

    parent_rows: list[dict[str, object]] = []
    for (tunnel, parent), rows in sorted(grouped.items()):
        seeds = {row["seed"] for row in rows}
        detectors = {row["detector"] for row in rows}
        backbones = {row["backbone"] for row in rows}
        if (len(rows), len(seeds), len(detectors), len(backbones)) != (24, 2, 3, 4):
            raise RuntimeError(f"incomplete parent cell: {(tunnel, parent)}")
        raw = mean([float(row["raw"]) for row in rows])
        framework = mean([float(row["framework"]) for row in rows])
        parent_rows.append(
            {
                "tunnel": tunnel,
                "parent": parent,
                "seed_count": len(seeds),
                "detector_count": len(detectors),
                "backbone_count": len(backbones),
                "raw_auprc": f"{raw:.9f}",
                "complete_path_auprc": f"{framework:.9f}",
                "delta_auprc": f"{framework - raw:.9f}",
                "statistical_role": "physical-parent unit nested within tunnel",
            }
        )

    if len(parent_rows) != 14:
        raise RuntimeError(f"expected 14 physical parents, found {len(parent_rows)}")

    tunnel_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in parent_rows:
        tunnel_groups[str(row["tunnel"])].append(row)

    tunnel_rows: list[dict[str, object]] = []
    for tunnel, rows in sorted(tunnel_groups.items()):
        raw = mean([float(row["raw_auprc"]) for row in rows])
        framework = mean([float(row["complete_path_auprc"]) for row in rows])
        tunnel_rows.append(
            {
                "tunnel": tunnel,
                "parent_count": len(rows),
                "raw_auprc": f"{raw:.9f}",
                "complete_path_auprc": f"{framework:.9f}",
                "delta_auprc": f"{framework - raw:.9f}",
                "statistical_role": "top-level tunnel cluster; equal tunnel weight",
            }
        )

    if len(tunnel_rows) != 4:
        raise RuntimeError(f"expected four tunnel clusters, found {len(tunnel_rows)}")

    write_rows(PARENT_OUT, parent_rows)
    write_rows(TUNNEL_OUT, tunnel_rows)


if __name__ == "__main__":
    main()
