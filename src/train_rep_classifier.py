from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut

from rep_dataset import export_rep_table, load_feature_table, rep_model_feature_columns


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_ROOT = PROJECT_ROOT / "data" / "features"
TRAINING_ROOT = PROJECT_ROOT / "data" / "training"
MODEL_ROOT = PROJECT_ROOT / "models"
DEFAULT_REP_TABLE = TRAINING_ROOT / "rep_level_features.csv"


def evaluate_leave_one_video_out(model, X, y, groups) -> None:
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
        print(
            f"LOVO fold {fold_idx:02d} | test_video={Path(test_video).name} | n={len(test_idx)} | acc={acc:.3f} | macro_f1={macro_f1:.3f}"
        )

    print("\nLeave-One-Video-Out summary")
    print(f"Folds: {len(fold_accuracies)}")
    print(f"Mean accuracy: {np.mean(fold_accuracies):.3f} +/- {np.std(fold_accuracies):.3f}")
    print(f"Mean macro F1: {np.mean(fold_macro_f1):.3f} +/- {np.std(fold_macro_f1):.3f}")


def load_rep_table(rep_table_path: Path) -> pd.DataFrame:
    if rep_table_path.exists():
        rep_table = pd.read_csv(rep_table_path)
        print(f"Loaded rep-level table: {rep_table_path}")
        return rep_table

    print(f"Rep-level table not found at {rep_table_path}; generating it from feature CSVs.")
    all_features = load_feature_table(FEATURE_ROOT)
    rep_table = export_rep_table(FEATURE_ROOT, rep_table_path)
    return rep_table


def main() -> None:
    rep_table = load_rep_table(DEFAULT_REP_TABLE)

    if rep_table.empty:
        raise SystemExit("No rep rows were exported. Add more videos or check the rep segmentation thresholds.")

    TRAINING_ROOT.mkdir(parents=True, exist_ok=True)
    rep_table.to_csv(DEFAULT_REP_TABLE, index=False)
    print(f"Rep-level rows: {len(rep_table)}")
    print("Classes:", sorted(rep_table["class_label"].dropna().unique().tolist()))

    if rep_table["class_label"].nunique() < 2:
        raise SystemExit("Need at least 2 classes of rep labels before training the classifier.")

    feature_cols = rep_model_feature_columns(rep_table.columns.tolist())

    X = rep_table[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = rep_table["class_label"]
    groups = rep_table["video_path"]

    print(f"Training matrix shape: {X.shape}")
    print(f"Using {len(feature_cols)} rep-level features")

    model = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight="balanced_subsample",
        min_samples_leaf=1,
    )

    evaluate_leave_one_video_out(model, X, y, groups)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nEvaluation")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=model.classes_))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop feature importances:")
    print(importances.head(12).to_string())

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_ROOT / "squat_rep_classifier.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved: {model_path}")


if __name__ == "__main__":
    main()