"""Assess author Section 2.1 CSV rows against the paper's stated criteria."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

REQUIRED = {"bandgap_onehot", "ef_onehot", "is_neutral_cmn", "minbond_less_than_0.5"}

def boolean(value: str) -> bool:
    if value in ("True", "1"):
        return True
    if value in ("False", "0", ""):
        return False
    raise ValueError(f"Unrecognized boolean: {value!r}")

def finite_number(value: str) -> float:
    try:
        result = float(value)
    except (ValueError, TypeError):
        return math.nan
    return result if math.isfinite(result) else math.nan

def assess(rows: list[dict], expected: int) -> dict:
    if expected < 1 or len(rows) > expected:
        raise ValueError("Require positive denominator and no excess rows.")
    counts = dict(bandgap=0, formation_energy=0, valid_structure=0, joint=0)
    for row in rows:
        if not REQUIRED.issubset(row):
            raise ValueError("Missing required author CSV columns.")
        gap = finite_number(row["bandgap_onehot"])
        energy = finite_number(row["ef_onehot"])
        gap_ok = 2.3 <= gap <= 2.7
        energy_ok = energy < -0.5
        valid = boolean(row["is_neutral_cmn"]) and boolean(row["minbond_less_than_0.5"])
        counts["bandgap"] += int(gap_ok)
        counts["formation_energy"] += int(energy_ok)
        counts["valid_structure"] += int(valid)
        counts["joint"] += int(gap_ok and energy_ok and valid)
    return {"expected_candidates": expected, "saved_rows": len(rows),
            "missing_candidates": expected-len(rows), "counts": counts,
            "rates": {k: v/expected for k,v in counts.items()},
            "criteria": {"bandgap_eV": [2.3, 2.7],
                         "formation_energy_eV_per_atom": "< -0.5",
                         "valid_structure": "is_neutral_cmn AND minbond_less_than_0.5"},
            "scope": "Predictor-based Section 2.1 criteria; not DFT or relaxation validation."}

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--expected-candidates", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        p.error("Run result processing through Slurm.")
    with args.csv.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not REQUIRED.issubset(reader.fieldnames or []):
            raise ValueError("Missing required author CSV columns.")
        report = assess(list(reader), args.expected_candidates)
    report["csv_sha256"] = hashlib.sha256(args.csv.read_bytes()).hexdigest()
    report["job_id"] = os.environ["SLURM_JOB_ID"]
    report["csv_path"] = str(args.csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
