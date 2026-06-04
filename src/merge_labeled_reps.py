from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG = PROJECT_ROOT / "data" / "training" / "rep_inference_log.csv"
DEFAULT_BASE_TABLE = PROJECT_ROOT / "data" / "training" / "rep_level_features.csv"
DEFAULT_LABELED_ONLY = PROJECT_ROOT / "data" / "training" / "rep_manual_labeled.csv"
DEFAULT_MERGED = PROJECT_ROOT / "data" / "training" / "rep_level_features_augmented.csv"


def _normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    if "manual_label" not in df.columns:
        raise SystemExit("Input file must contain a manual_label column.")

    labeled = df[df["manual_label"].fillna("").astype(str).str.strip().ne("")].copy()
    if labeled.empty:
        return labeled

    labeled["class_label"] = labeled["manual_label"].astype(str).str.strip()
    labeled = labeled.drop(columns=["manual_label"], errors="ignore")
    return labeled


def _align_columns(left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_columns = list(dict.fromkeys(list(left.columns) + list(right.columns)))
    left_aligned = left.reindex(columns=all_columns)
    right_aligned = right.reindex(columns=all_columns)
    return left_aligned, right_aligned


def merge_labeled_rows(log_csv: Path, base_table_csv: Path, labeled_only_csv: Path, merged_csv: Path) -> pd.DataFrame:
    if not log_csv.exists():
        raise SystemExit(f"Inference log does not exist: {log_csv}")

    log_df = pd.read_csv(log_csv)
    labeled_df = _normalize_labels(log_df)

    labeled_only_csv.parent.mkdir(parents=True, exist_ok=True)
    labeled_df.to_csv(labeled_only_csv, index=False)
    print(f"Saved labeled rows: {labeled_only_csv}")
    print(f"Labeled rows found: {len(labeled_df)}")

    if base_table_csv.exists():
        base_df = pd.read_csv(base_table_csv)
        print(f"Loaded base rep table: {base_table_csv}")
    else:
        base_df = pd.DataFrame()
        print(f"Base rep table not found at {base_table_csv}; creating augmented table from labeled rows only.")

    if base_df.empty and labeled_df.empty:
        merged_df = pd.DataFrame()
    elif base_df.empty:
        merged_df = labeled_df.copy()
    elif labeled_df.empty:
        merged_df = base_df.copy()
    else:
        base_aligned, labeled_aligned = _align_columns(base_df, labeled_df)
        merged_df = pd.concat([base_aligned, labeled_aligned], ignore_index=True)

    merged_csv.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(merged_csv, index=False)
    print(f"Saved merged rep table: {merged_csv}")
    print(f"Merged rows: {len(merged_df)}")
    return merged_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge manually labeled squat reps back into the training table.")
    parser.add_argument("--log", type=str, default=str(DEFAULT_LOG), help="Input rep inference log CSV")
    parser.add_argument("--base-table", type=str, default=str(DEFAULT_BASE_TABLE), help="Existing rep-level training table")
    parser.add_argument("--labeled-only-output", type=str, default=str(DEFAULT_LABELED_ONLY), help="CSV for rows with manual labels")
    parser.add_argument("--output", type=str, default=str(DEFAULT_MERGED), help="Merged training table output")
    args = parser.parse_args()

    merge_labeled_rows(
        Path(args.log),
        Path(args.base_table),
        Path(args.labeled_only_output),
        Path(args.output),
    )


if __name__ == "__main__":
    main()