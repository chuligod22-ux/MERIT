"""Create the C057 execution freeze after estimator, runner and tests are final."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from run_mtf50_v2 import (
    DEFAULT_CONTRACT,
    DEFAULT_EXECUTION_FREEZE,
    DEFAULT_FREEZE,
    ROOT,
    load_contract_cohort,
    sha256_file,
    utc_now,
)


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--output", type=Path, default=DEFAULT_EXECUTION_FREEZE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    contract, frozen, _, partition_report = load_contract_cohort(
        args.contract, args.freeze
    )
    base = Path(__file__).parent
    cohort = ROOT / contract["inputs"]["cohort_path"]
    paths = {
        "estimator_source": base / "mtf50_estimator.py",
        "runner_source": base / "run_mtf50_v2.py",
        "test_source": base / "tests" / "test_mtf50_estimator.py",
        "development_contract": args.contract,
        "cohort": cohort,
        "preoutcome_freeze": args.freeze,
    }
    payload = {
        "schema_version": "MERIT-C057-MTF50-V2-EXECUTION-FREEZE-v1",
        "created_utc": utc_now(),
        "state": "FROZEN_BEFORE_OFFICIAL_HOLDOUT_AND_RESTORATION_OUTPUTS",
        "selected_candidate": {
            "localization_gaussian_sigma_px": float(
                frozen["selected_candidate"]["localization_gaussian_sigma_px"]
            ),
            "polarity": frozen["selected_candidate"]["polarity"],
            "line_fit": frozen["selected_candidate"]["line_fit"],
            "esf_input": frozen["selected_candidate"]["esf_input"],
        },
        "files": {
            label: {"path": relative(path), "sha256": sha256_file(path)}
            for label, path in paths.items()
        },
        "partition_validation": partition_report,
        "prohibition": "No source, test, contract, cohort, or selected sigma change is permitted before official holdout execution.",
    }
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if (args.output.exists() or sidecar.exists()) and not args.overwrite:
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    freeze_hash = sha256_file(args.output)
    temporary_sidecar = sidecar.with_suffix(sidecar.suffix + ".tmp")
    temporary_sidecar.write_text(
        f"{freeze_hash}  {args.output.name}\n", encoding="ascii"
    )
    os.replace(temporary_sidecar, sidecar)
    print(json.dumps({"execution_freeze_sha256": freeze_hash}, sort_keys=True))


if __name__ == "__main__":
    main()
