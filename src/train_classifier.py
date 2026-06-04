import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.base import clone
import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_ROOT = PROJECT_ROOT / "data" / "features"
MODEL_ROOT = PROJECT_ROOT / "models"
TRAINING_ROOT = PROJECT_ROOT / "data" / "training"

FEATURE_COLUMNS = [
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


def load_features(feature_root: Path) -> pd.DataFrame:
    csv_files = list(feature_root.rglob("*.csv"))
    print(f"Found {len(csv_files)} feature files")

    if not csv_files:
        raise SystemExit(f"No feature CSV files found in {feature_root}")

    dfs = [pd.read_csv(csv) for csv in csv_files]
    all_features = pd.concat(dfs, ignore_index=True)
    print(f"Total rows: {len(all_features)}")
    return all_features


def build_video_level_table(all_features: pd.DataFrame) -> pd.DataFrame:
    required = {"video_path", "video_name", "class_label"}
    missing = required - set(all_features.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    feature_cols = [col for col in FEATURE_COLUMNS if col in all_features.columns]

    numeric = all_features.copy()
    for col in feature_cols:
        numeric[col] = pd.to_numeric(numeric[col], errors="coerce")

    agg_spec = {}
    for col in feature_cols:
        agg_spec[col] = ["mean", "std", "min", "max"]
    agg_spec["class_label"] = "first"
    agg_spec["video_name"] = "first"

    grouped = numeric.groupby("video_path", dropna=False).agg(agg_spec)
    grouped.columns = ["_".join([part for part in col if part]) if isinstance(col, tuple) else col for col in grouped.columns]
    grouped = grouped.reset_index()

    # flatten labels and identifiers back to useful names
    grouped = grouped.rename(columns={"class_label_first": "class_label", "video_name_first": "video_name"})
    grouped["video_path"] = grouped["video_path"].astype(str)
    return grouped


def evaluate_leave_one_video_out(model, X, y, groups):
    logo = LeaveOneGroupOut()
    fold_accuracies = []
    fold_macro_f1 = []

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        fold_model = clone(model)
        fold_model.fit(X_train, y_train)
        y_pred = fold_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        fold_accuracies.append(acc)
        fold_macro_f1.append(macro_f1)

        test_video = str(groups.iloc[test_idx[0]]) if len(test_idx) > 0 else "unknown"
        print(f"LOVO fold {fold_idx:02d} | test_video={Path(test_video).name} | n={len(test_idx)} | acc={acc:.3f} | macro_f1={macro_f1:.3f}")

    print("\nLeave-One-Video-Out summary")
    print(f"Folds: {len(fold_accuracies)}")
    print(f"Mean accuracy: {np.mean(fold_accuracies):.3f} +/- {np.std(fold_accuracies):.3f}")
    print(f"Mean macro F1: {np.mean(fold_macro_f1):.3f} +/- {np.std(fold_macro_f1):.3f}")


def main():
    all_features = load_features(FEATURE_ROOT)
    video_table = build_video_level_table(all_features)

    TRAINING_ROOT.mkdir(parents=True, exist_ok=True)
    video_table_path = TRAINING_ROOT / "video_level_features.csv"
    video_table.to_csv(video_table_path, index=False)
    print(f"Saved video-level table: {video_table_path}")

    print(f"Video-level rows: {len(video_table)}")
    print("Classes:", sorted(video_table["class_label"].dropna().unique().tolist()))

    if video_table["class_label"].nunique() < 2:
        raise SystemExit(
            "Saved the video-level table, but training needs at least 2 classes. Add more labeled videos in different classes first."
        )

    feature_cols = [
        col for col in video_table.columns
        if col not in {"video_path", "video_name", "class_label"}
        and pd.api.types.is_numeric_dtype(video_table[col])
    ]

    X = video_table[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = video_table["class_label"]
    groups = video_table["video_path"]

    print(f"Training matrix shape: {X.shape}")
    print(f"Using {len(feature_cols)} video-level features")

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
    )

    # Stable metric for small datasets: test each video as the holdout once.
    evaluate_leave_one_video_out(model, X, y, groups)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\nEvaluation")
    print(f"Accuracy: {acc:.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=model.classes_))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop feature importances:")
    print(importances.head(10).to_string())

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_ROOT / "squat_classifier.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved: {model_path}")


if __name__ == "__main__":
    main()