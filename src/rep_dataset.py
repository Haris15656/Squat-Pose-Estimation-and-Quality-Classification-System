from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REP_UP_THRESHOLD = 160
REP_DOWN_THRESHOLD = 105

BASE_FEATURE_COLUMNS = [
    "left_knee_angle",
    "right_knee_angle",
    "avg_knee_angle",
    "left_hip_angle",
    "right_hip_angle",
    "avg_hip_angle",
    "torso_angle_left",
    "torso_angle_right",
    "avg_torso_angle",
    "hip_depth",
    "avg_knee_angle_smooth",
    "avg_hip_angle_smooth",
    "avg_torso_angle_smooth",
    "hip_depth_smooth",
    "pose_detected",
]

REP_SUMMARY_METRICS = ["mean", "std", "min", "max"]
REP_FEATURE_COLUMNS = [
    f"{column}_{metric}"
    for column in [
        "left_knee_angle",
        "right_knee_angle",
        "avg_knee_angle",
        "left_hip_angle",
        "right_hip_angle",
        "avg_hip_angle",
        "torso_angle_left",
        "torso_angle_right",
        "avg_torso_angle",
        "hip_depth",
        "avg_knee_angle_smooth",
        "avg_hip_angle_smooth",
        "avg_torso_angle_smooth",
        "hip_depth_smooth",
    ]
    for metric in REP_SUMMARY_METRICS
] + [
    "avg_knee_angle_range",
    "avg_torso_angle_range",
    "hip_depth_range",
    "duration_sec",
    "n_frames",
    "pose_coverage",
]


@dataclass
class RepSegment:
    video_path: str
    video_name: str
    class_label: str
    rep_index: int
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float
    duration_sec: float
    n_frames: int


def load_feature_table(feature_root: Path) -> pd.DataFrame:
    csv_files = sorted(feature_root.rglob("*.csv"))
    print(f"Found {len(csv_files)} feature files")

    if not csv_files:
        raise SystemExit(f"No feature CSV files found in {feature_root}")

    tables = [pd.read_csv(csv_path) for csv_path in csv_files]
    all_features = pd.concat(tables, ignore_index=True)
    print(f"Total rows: {len(all_features)}")
    return all_features


def _safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    return float(np.nanmean(values)) if values.notna().any() else float("nan")


def _safe_std(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    return float(np.nanstd(values)) if values.notna().any() else float("nan")


def _safe_min(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    return float(np.nanmin(values)) if values.notna().any() else float("nan")


def _safe_max(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    return float(np.nanmax(values)) if values.notna().any() else float("nan")


def summarize_rep_frames(frames: list[dict], rep_index: int | None = None) -> dict:
    frame_df = pd.DataFrame(frames)
    if frame_df.empty:
        return {}

    first_row = frame_df.iloc[0]
    last_row = frame_df.iloc[-1]

    summary = {
        "rep_index": rep_index if rep_index is not None else 0,
        "duration_sec": float(last_row.get("timestamp_sec", 0.0) - first_row.get("timestamp_sec", 0.0)),
        "n_frames": int(len(frame_df)),
        "pose_coverage": float(pd.to_numeric(frame_df.get("pose_detected", pd.Series(dtype=float)), errors="coerce").fillna(0).mean()),
    }

    summary_columns = [
        "left_knee_angle",
        "right_knee_angle",
        "avg_knee_angle",
        "left_hip_angle",
        "right_hip_angle",
        "avg_hip_angle",
        "torso_angle_left",
        "torso_angle_right",
        "avg_torso_angle",
        "hip_depth",
        "avg_knee_angle_smooth",
        "avg_hip_angle_smooth",
        "avg_torso_angle_smooth",
        "hip_depth_smooth",
    ]

    for col in summary_columns:
        if col not in frame_df.columns:
            continue
        summary[f"{col}_mean"] = _safe_mean(frame_df[col])
        summary[f"{col}_std"] = _safe_std(frame_df[col])
        summary[f"{col}_min"] = _safe_min(frame_df[col])
        summary[f"{col}_max"] = _safe_max(frame_df[col])

    summary["avg_knee_angle_range"] = summary.get("avg_knee_angle_max", float("nan")) - summary.get("avg_knee_angle_min", float("nan"))
    summary["avg_torso_angle_range"] = summary.get("avg_torso_angle_max", float("nan")) - summary.get("avg_torso_angle_min", float("nan"))
    summary["hip_depth_range"] = summary.get("hip_depth_max", float("nan")) - summary.get("hip_depth_min", float("nan"))

    return summary


def rep_model_feature_columns(columns: list[str]) -> list[str]:
    return [column for column in REP_FEATURE_COLUMNS if column in columns]


def _segment_group(group: pd.DataFrame) -> list[pd.DataFrame]:
    ordered = group.sort_values(["timestamp_sec", "frame"], kind="mergesort").reset_index(drop=True)

    rep_segments: list[pd.DataFrame] = []
    state = "up"
    segment_start_idx: int | None = None

    for idx, row in ordered.iterrows():
        knee = row.get("avg_knee_angle_smooth", np.nan)
        if pd.isna(knee):
            continue

        if knee > REP_UP_THRESHOLD:
            if state == "down" and segment_start_idx is not None:
                segment = ordered.iloc[segment_start_idx:idx].copy()
                if not segment.empty:
                    rep_segments.append(segment)
            state = "up"
            segment_start_idx = None
            continue

        if knee < REP_DOWN_THRESHOLD:
            if state != "down":
                segment_start_idx = idx
            state = "down"

    return rep_segments


def _summarize_rep(segment: pd.DataFrame, rep_index: int) -> dict:
    first_row = segment.iloc[0]
    last_row = segment.iloc[-1]

    summary = {
        "video_path": str(first_row.get("video_path", "")),
        "video_name": str(first_row.get("video_name", "")),
        "class_label": str(first_row.get("class_label", "")),
        "rep_index": rep_index,
        "start_frame": int(first_row.get("frame", 0)),
        "end_frame": int(last_row.get("frame", 0)),
        "start_time_sec": float(first_row.get("timestamp_sec", 0.0)),
        "end_time_sec": float(last_row.get("timestamp_sec", 0.0)),
        "duration_sec": float(last_row.get("timestamp_sec", 0.0) - first_row.get("timestamp_sec", 0.0)),
        "n_frames": int(len(segment)),
        "pose_coverage": float(pd.to_numeric(segment.get("pose_detected", pd.Series(dtype=float)), errors="coerce").fillna(0).mean()),
    }

    numeric_features = [
        col for col in BASE_FEATURE_COLUMNS
        if col in segment.columns and col not in {"pose_detected"}
    ]

    for col in numeric_features:
        summary[f"{col}_mean"] = _safe_mean(segment[col])
        summary[f"{col}_std"] = _safe_std(segment[col])
        summary[f"{col}_min"] = _safe_min(segment[col])
        summary[f"{col}_max"] = _safe_max(segment[col])

    summary["avg_knee_angle_range"] = summary.get("avg_knee_angle_max", float("nan")) - summary.get("avg_knee_angle_min", float("nan"))
    summary["avg_torso_angle_range"] = summary.get("avg_torso_angle_max", float("nan")) - summary.get("avg_torso_angle_min", float("nan"))
    summary["hip_depth_range"] = summary.get("hip_depth_max", float("nan")) - summary.get("hip_depth_min", float("nan"))

    return summary


def build_rep_table(all_features: pd.DataFrame) -> pd.DataFrame:
    required = {"video_path", "video_name", "class_label", "frame", "timestamp_sec"}
    missing = required - set(all_features.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    rep_rows: list[dict] = []
    grouped = all_features.groupby("video_path", dropna=False, sort=False)

    for video_path, group in grouped:
        segments = _segment_group(group)
        if not segments:
            print(f"No complete reps detected in {Path(str(video_path)).name}")
            continue

        for rep_index, segment in enumerate(segments, start=1):
            rep_rows.append(_summarize_rep(segment, rep_index))

    rep_table = pd.DataFrame(rep_rows)
    if not rep_table.empty:
        rep_table["video_path"] = rep_table["video_path"].astype(str)
        rep_table["video_name"] = rep_table["video_name"].astype(str)
        rep_table["class_label"] = rep_table["class_label"].astype(str)

    return rep_table


def export_rep_table(feature_root: Path, output_csv: Path) -> pd.DataFrame:
    all_features = load_feature_table(feature_root)
    rep_table = build_rep_table(all_features)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rep_table.to_csv(output_csv, index=False)
    print(f"Saved rep-level table: {output_csv}")
    print(f"Rep-level rows: {len(rep_table)}")
    if not rep_table.empty:
        print("Classes:", sorted(rep_table["class_label"].dropna().unique().tolist()))
    return rep_table
