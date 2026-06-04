from __future__ import annotations

import argparse
from pathlib import Path

from rep_dataset import export_rep_table


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Export one row per squat rep from per-frame feature CSVs.")
    parser.add_argument("--features-dir", type=str, default=str(PROJECT_ROOT / "data" / "features"), help="Directory containing per-video feature CSV files")
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "data" / "training" / "rep_level_features.csv"), help="Output CSV for rep-level features")
    args = parser.parse_args()

    feature_root = Path(args.features_dir)
    output_csv = Path(args.output)
    export_rep_table(feature_root, output_csv)


if __name__ == "__main__":
    main()
