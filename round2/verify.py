"""Check release integrity and optionally run the bundled CPU analyses."""
import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "MANIFEST.sha256"


def verify_files():
    listed = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        checksum, name = line.split("  ", 1)
        if name in listed:
            raise ValueError(f"Duplicate manifest entry: {name}")
        listed[name] = checksum
    files = {p.relative_to(ROOT).as_posix(): p for p in ROOT.rglob("*")
             if p.is_file() and p != MANIFEST and "__pycache__" not in p.parts}
    if listed.keys() != files.keys():
        raise ValueError(f"Manifest membership differs: {sorted(listed.keys() ^ files.keys())}")
    for name, path in files.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != listed[name]:
            raise ValueError(f"Checksum mismatch: {name}")
    print(f"Verified SHA-256 for {len(files)} files.", flush=True)


def verify_analyses():
    for script in ["factorial/verify.py", "detection/verify.py", "detection/verify_intervals.py",
                   "optics/gate/refit.py", "optics/chart/verify.py"]:
        subprocess.run([sys.executable, "-B", str(ROOT / script)], check=True)
    raw = subprocess.check_output([sys.executable, "-B", str(ROOT / "field_width" / "analyze.py")])
    width = json.loads(raw)["natural_width"]
    wide = width["width_bands"][">=5"]
    if (width["regions"], width["policy_pairs"], wide["paired"]) != (1057, 741, 174):
        raise ValueError("Unexpected apparent-width counts")
    for measured, reference in [(width["hierarchical_delta_px"], -0.307222),
                                (wide["region_mean_delta_px"], 0.748563),
                                *zip(width["span_bootstrap_95pct_ci_px"], [-0.414666, -0.219153])]:
        if not math.isclose(measured, reference, rel_tol=0, abs_tol=5e-7):
            raise ValueError(f"Apparent-width estimate differs: {measured}")
    print("Field width verified for 741 pairs, including 174 in the widest band.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyses", action="store_true", help="also recompute the reported CPU checks")
    args = parser.parse_args()
    verify_files()
    if args.analyses:
        verify_analyses()
