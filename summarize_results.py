"""Combine evaluation CSV and JSON sidecars into one paper-table-ready CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("experiments/results"))
    parser.add_argument("--output", type=Path, default=Path("experiments/results/all_results.csv"))
    args = parser.parse_args()
    rows = []
    for csv_path in sorted(args.input_dir.glob("*.csv")):
        if csv_path.resolve() == args.output.resolve():
            continue
        metadata_path = csv_path.with_suffix(".json")
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append({
                    "model": metadata["model_type"],
                    "training_experiment": metadata["experiment"],
                    "calibration": metadata["calibration_label"],
                    "calibration_region": metadata["calibration_region"],
                    "test_region": metadata["test_region"],
                    **row,
                })
    if not rows:
        raise FileNotFoundError(f"No paired CSV/JSON evaluation outputs in {args.input_dir}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f"wrote {args.output} with {len(rows)} rows")


if __name__ == "__main__":
    main()
