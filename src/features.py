import numpy as np
import pandas as pd
from pathlib import Path

def angle_2d(a, b, c):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return np.nan

    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))

def angle_to_vertical(top, bottom):
    top = np.array(top, dtype=float)
    bottom = np.array(bottom, dtype=float)
    vec = top - bottom
    vertical = np.array([0.0, -1.0])

    norm_vec = np.linalg.norm(vec)
    if norm_vec == 0:
        return np.nan

    cosine = np.dot(vec, vertical) / norm_vec
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))

def midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)

def get_scale(row):
    # prefer shoulder width, fallback to hip width
    try:
        ls_x, ls_y = row["left_shoulder_x"], row["left_shoulder_y"]
        rs_x, rs_y = row["right_shoulder_x"], row["right_shoulder_y"]
        if not (np.isfinite(ls_x) and np.isfinite(rs_x)):
            raise Exception()
        shoulder_dist = np.hypot(rs_x - ls_x, rs_y - ls_y)
    except Exception:
        try:
            lh_x, lh_y = row["left_hip_x"], row["left_hip_y"]
            rh_x, rh_y = row["right_hip_x"], row["right_hip_y"]
            shoulder_dist = np.hypot(rh_x - lh_x, rh_y - lh_y)
        except Exception:
            shoulder_dist = 1.0

    return max(shoulder_dist, 1e-6)


def get_xy(row, name, scale=1.0):
    x = row.get(f"{name}_x", np.nan)
    y = row.get(f"{name}_y", np.nan)
    if not np.isfinite(x) or not np.isfinite(y):
        return (np.nan, np.nan)
    return (x / scale, y / scale)

def compute_features_for_file(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    feature_rows = []

    for _, row in df.iterrows():
        base = {
            "frame": row["frame"],
            "timestamp_sec": row["timestamp_sec"],
            "video_path": row.get("video_path", ""),
            "video_name": row.get("video_name", ""),
            "class_label": row.get("class_label", ""),
            "pose_detected": row["pose_detected"]
        }

        if row["pose_detected"] != 1:
            feature_rows.append({
                **base,
                "left_knee_angle": np.nan,
                "right_knee_angle": np.nan,
                "avg_knee_angle": np.nan,
                "left_hip_angle": np.nan,
                "right_hip_angle": np.nan,
                "avg_hip_angle": np.nan,
                "torso_angle_left": np.nan,
                "torso_angle_right": np.nan,
                "avg_torso_angle": np.nan,
                "hip_depth": np.nan
            })
            continue

        scale = get_scale(row)

        left_shoulder = get_xy(row, "left_shoulder", scale)
        right_shoulder = get_xy(row, "right_shoulder", scale)
        left_hip = get_xy(row, "left_hip", scale)
        right_hip = get_xy(row, "right_hip", scale)
        left_knee = get_xy(row, "left_knee", scale)
        right_knee = get_xy(row, "right_knee", scale)
        left_ankle = get_xy(row, "left_ankle", scale)
        right_ankle = get_xy(row, "right_ankle", scale)

        left_knee_angle = angle_2d(left_hip, left_knee, left_ankle)
        right_knee_angle = angle_2d(right_hip, right_knee, right_ankle)

        left_hip_angle = angle_2d(left_shoulder, left_hip, left_knee)
        right_hip_angle = angle_2d(right_shoulder, right_hip, right_knee)

        torso_angle_left = angle_to_vertical(left_shoulder, left_hip)
        torso_angle_right = angle_to_vertical(right_shoulder, right_hip)

        hip_mid = midpoint(left_hip, right_hip)
        knee_mid = midpoint(left_knee, right_knee)
        ankle_mid = midpoint(left_ankle, right_ankle)

        hip_depth = (ankle_mid[1] - hip_mid[1]) / max(abs(ankle_mid[1] - knee_mid[1]), 1e-6)

        feature_rows.append({
            **base,
            "left_knee_angle": left_knee_angle,
            "right_knee_angle": right_knee_angle,
            "avg_knee_angle": np.nanmean([left_knee_angle, right_knee_angle]),
            "left_hip_angle": left_hip_angle,
            "right_hip_angle": right_hip_angle,
            "avg_hip_angle": np.nanmean([left_hip_angle, right_hip_angle]),
            "torso_angle_left": torso_angle_left,
            "torso_angle_right": torso_angle_right,
            "avg_torso_angle": np.nanmean([torso_angle_left, torso_angle_right]),
            "hip_depth": hip_depth
        })

    out_df = pd.DataFrame(feature_rows)

    for col in ["avg_knee_angle", "avg_hip_angle", "avg_torso_angle", "hip_depth"]:
        out_df[f"{col}_smooth"] = out_df[col].rolling(window=5, min_periods=1, center=True).mean()

    # save the scale used (shoulder/hip width) averaged per frame for analysis
    scales = []
    for _, row in df.iterrows():
        scales.append(get_scale(row))
    out_df["scale_px"] = pd.Series(scales)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"Saved features: {output_csv}")

def process_feature_dataset(input_root="data/extracted_landmarks", output_root="data/features"):
    input_root = Path(input_root)
    output_root = Path(output_root)

    csv_files = list(input_root.rglob("*_landmarks.csv"))
    if not csv_files:
        print(f"No landmark CSV files found in {input_root}")
        return

    print(f"Found {len(csv_files)} landmark CSV files")

    for input_csv in csv_files:
        class_label = input_csv.parent.name
        output_csv = output_root / class_label / input_csv.name.replace("_landmarks.csv", "_features.csv")
        compute_features_for_file(input_csv, output_csv)

if __name__ == "__main__":
    process_feature_dataset()