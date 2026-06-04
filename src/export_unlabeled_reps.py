from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "training" / "rep_inference_log.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "training" / "rep_inference_unlabeled.csv"


def export_unlabeled_rows(input_csv: Path, output_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise SystemExit(f"Input log does not exist: {input_csv}")

    df = pd.read_csv(input_csv)
    if "manual_label" not in df.columns:
        raise SystemExit(f"Missing manual_label column in {input_csv}")

    labels = df["manual_label"].fillna("").astype(str).str.strip()
    unlabeled = df[labels.eq("")].copy()

    if unlabeled.empty:
        print(f"No unlabeled rows found in {input_csv}")
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        unlabeled.to_csv(output_csv, index=False)
        print(f"Wrote empty review sheet: {output_csv}")
        return unlabeled

    sort_columns = [column for column in ["source", "rep_index", "start_time_sec"] if column in unlabeled.columns]
    if sort_columns:
        unlabeled = unlabeled.sort_values(sort_columns, kind="mergesort")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    unlabeled.to_csv(output_csv, index=False)
    print(f"Exported {len(unlabeled)} unlabeled rows to: {output_csv}")
    return unlabeled


def main() -> None:
    parser = argparse.ArgumentParser(description="Export unlabeled squat reps from the inference log.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="Input rep inference log CSV")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output CSV for unlabeled rows")
    args = parser.parse_args()

    export_unlabeled_rows(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()