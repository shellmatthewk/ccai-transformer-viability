"""
Check ECA&D station file coverage/validity for a target date range.

Works for both RR (precipitation) and TG (temperature) files — the format
is the same shape (STAID, SOUID, DATE, VALUE, Q_FLAG), just different
column names and missing-value conventions.

Usage:
    python check_station_coverage.py --dir /path/to/RR_files --var RR
    python check_station_coverage.py --dir /path/to/TG_files --var TG
"""

import argparse
import glob
import os
import re
import pandas as pd


def parse_station_file(filepath: str, var: str) -> pd.DataFrame:
    """Parse a single ECA&D station file (RR_* or TG_*) into a DataFrame."""
    q_col = f"Q_{var}"
    col_names = ["STAID", "SOUID", "DATE", var, q_col]

    # Header length varies slightly between RR and TG files; find the actual
    # data start line by locating the first line that starts with digits
    # after stripping whitespace, rather than hardcoding a skiprows count.
    with open(filepath, "r", encoding="latin-1") as f:
        lines = f.readlines()

    data_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"^\d+\s*,\s*\d+\s*,\s*\d{8}", stripped):
            data_start = i
            break

    if data_start is None:
        return pd.DataFrame(columns=col_names)

    df = pd.read_csv(
        filepath,
        skiprows=data_start,
        names=col_names,
        sep=",",
        skipinitialspace=True,
        encoding="latin-1",
    )
    return df


def evaluate_station(df: pd.DataFrame, var: str, start_year: int, end_year: int,
                      completeness_threshold: float) -> dict:
    """Return a dict summarizing coverage/validity for one station's data."""
    q_col = f"Q_{var}"

    df["DATE"] = pd.to_datetime(df["DATE"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["DATE"])

    window = df[(df["DATE"].dt.year >= start_year) & (df["DATE"].dt.year <= end_year)]

    total_days_in_window = (
        pd.Timestamp(f"{end_year}-12-31") - pd.Timestamp(f"{start_year}-01-01")
    ).days + 1

    # Valid = present AND quality flag == 0 (not suspect, not missing) AND not -9999
    valid_mask = (window[q_col] == 0) & (window[var] != -9999)
    n_valid = valid_mask.sum()
    n_present = len(window)

    completeness = n_valid / total_days_in_window if total_days_in_window else 0.0

    return {
        "n_records_in_window": n_present,
        "n_valid_records": int(n_valid),
        "total_days_in_window": total_days_in_window,
        "completeness": round(completeness, 4),
        "is_valid": completeness >= completeness_threshold,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, help="Directory containing station files")
    parser.add_argument("--var", required=True, choices=["RR", "TG"], help="Variable code")
    parser.add_argument("--start_year", type=int, default=2010)
    parser.add_argument("--end_year", type=int, default=2026)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Minimum fraction of days in window that must be valid (default 0.8)",
    )
    parser.add_argument(
        "--out",
        default="station_coverage_report.csv",
        help="Output CSV path for the per-station report",
    )
    args = parser.parse_args()

    pattern = os.path.join(args.dir, f"{args.var}_*.txt")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No files matching {pattern} found.")
        return

    print(f"Found {len(files)} {args.var} station files. Evaluating coverage for "
          f"{args.start_year}-{args.end_year} (threshold={args.threshold})...")

    results = []
    for i, filepath in enumerate(files, 1):
        staid_match = re.search(rf"{args.var}_STAID0*(\d+)\.txt", os.path.basename(filepath))
        staid = staid_match.group(1) if staid_match else os.path.basename(filepath)

        try:
            df = parse_station_file(filepath, args.var)
            if df.empty:
                results.append({
                    "STAID": staid,
                    "file": os.path.basename(filepath),
                    "n_records_in_window": 0,
                    "n_valid_records": 0,
                    "total_days_in_window": None,
                    "completeness": 0.0,
                    "is_valid": False,
                    "error": "empty_or_unparsed",
                })
                continue

            summary = evaluate_station(df, args.var, args.start_year, args.end_year, args.threshold)
            summary["STAID"] = staid
            summary["file"] = os.path.basename(filepath)
            summary["error"] = None
            results.append(summary)

        except Exception as e:
            results.append({
                "STAID": staid,
                "file": os.path.basename(filepath),
                "n_records_in_window": None,
                "n_valid_records": None,
                "total_days_in_window": None,
                "completeness": None,
                "is_valid": False,
                "error": str(e),
            })

        if i % 500 == 0:
            print(f"  processed {i}/{len(files)}...")
            print(f"Last Processed: {staid}")

    report = pd.DataFrame(results)
    report = report[["STAID", "file", "n_records_in_window", "n_valid_records",
                      "total_days_in_window", "completeness", "is_valid", "error"]]
    report.to_csv(args.out, index=False)

    n_valid_stations = report["is_valid"].sum()
    print(f"\nDone. {n_valid_stations} / {len(report)} stations meet the "
          f"{args.threshold:.0%} completeness threshold for {args.start_year}-{args.end_year}.")
    print(f"Full report written to: {args.out}")


if __name__ == "__main__":
    main()