# -*- coding: utf-8 -*-
"""Fair comparison of classifiers for the mackerel LOW/MEDIUM/HIGH target.

Requires rf_experiment_search_v3.py in the same directory.
Uses fixed 33/66 thresholds and the same BASE features for every model.
Model selection uses 2565-2566 only; year 2567 remains the final test.
"""

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    import rf_experiment_search_v3 as v3
except ImportError as error:
    raise RuntimeError(
        "กรุณาวาง rf_experiment_search_v3.py ไว้ในโฟลเดอร์เดียวกัน"
    ) from error


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ROLLING_CSV = OUTPUT_DIR / "classification_model_compare_rolling.csv"
FINAL_JSON = OUTPUT_DIR / "classification_model_compare_final.json"

FEATURES = [
    "sst", "chlorophyll_a", "rainfall", "wind_speed",
    "month", "year", "station_id",
]
CLASS_ORDER = ["LOW", "MEDIUM", "HIGH"]
TOP_CANDIDATES = 12
Q_LOW, Q_HIGH = 0.33, 0.66


MODEL_SPECS = [
    {"name": "RandomForest_Baseline", "family": "Random Forest", "params": {
        "n_estimators": 300, "max_depth": None, "min_samples_leaf": 1,
        "max_features": "sqrt", "class_weight": "balanced",
    }},
    {"name": "ExtraTrees_300", "family": "Extra Trees", "params": {
        "n_estimators": 300, "max_depth": None, "min_samples_leaf": 1,
        "max_features": "sqrt", "class_weight": "balanced",
    }},
    {"name": "ExtraTrees_500_leaf2", "family": "Extra Trees", "params": {
        "n_estimators": 500, "max_depth": None, "min_samples_leaf": 2,
        "max_features": "sqrt", "class_weight": "balanced",
    }},
    {"name": "ExtraTrees_depth10", "family": "Extra Trees", "params": {
        "n_estimators": 400, "max_depth": 10, "min_samples_leaf": 2,
        "max_features": 0.7, "class_weight": "balanced",
    }},
    {"name": "GradientBoosting_lr005", "family": "Gradient Boosting", "params": {
        "n_estimators": 200, "learning_rate": 0.05, "max_depth": 2,
        "min_samples_leaf": 2, "subsample": 1.0,
    }},
    {"name": "GradientBoosting_lr010", "family": "Gradient Boosting", "params": {
        "n_estimators": 150, "learning_rate": 0.10, "max_depth": 2,
        "min_samples_leaf": 2, "subsample": 1.0,
    }},
    {"name": "GradientBoosting_depth3", "family": "Gradient Boosting", "params": {
        "n_estimators": 150, "learning_rate": 0.05, "max_depth": 3,
        "min_samples_leaf": 2, "subsample": 0.9,
    }},
    {"name": "HistGradientBoosting_leaf10", "family": "HistGradientBoosting", "params": {
        "learning_rate": 0.05, "max_iter": 200, "max_leaf_nodes": 15,
        "min_samples_leaf": 10, "l2_regularization": 0.1,
    }},
    {"name": "HistGradientBoosting_leaf20", "family": "HistGradientBoosting", "params": {
        "learning_rate": 0.08, "max_iter": 200, "max_leaf_nodes": 15,
        "min_samples_leaf": 20, "l2_regularization": 0.1,
    }},
    {"name": "SVM_RBF_C1", "family": "SVM", "params": {
        "C": 1.0, "gamma": "scale", "class_weight": "balanced",
    }},
    {"name": "SVM_RBF_C3", "family": "SVM", "params": {
        "C": 3.0, "gamma": "scale", "class_weight": "balanced",
    }},
    {"name": "SVM_RBF_C10", "family": "SVM", "params": {
        "C": 10.0, "gamma": "scale", "class_weight": "balanced",
    }},
    {"name": "Logistic_C01", "family": "Logistic Regression", "params": {
        "C": 0.1, "class_weight": "balanced", "max_iter": 3000,
    }},
    {"name": "Logistic_C1", "family": "Logistic Regression", "params": {
        "C": 1.0, "class_weight": "balanced", "max_iter": 3000,
    }},
    {"name": "Logistic_C10", "family": "Logistic Regression", "params": {
        "C": 10.0, "class_weight": "balanced", "max_iter": 3000,
    }},
]


def create_model(spec):
    params = dict(spec["params"])
    family = spec["family"]
    if family == "Random Forest":
        return RandomForestClassifier(**params, random_state=42, n_jobs=1)
    if family == "Extra Trees":
        return ExtraTreesClassifier(**params, random_state=42, n_jobs=1)
    if family == "Gradient Boosting":
        return GradientBoostingClassifier(**params, random_state=42)
    if family == "HistGradientBoosting":
        return HistGradientBoostingClassifier(**params, random_state=42)
    if family == "SVM":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", SVC(**params, kernel="rbf")),
        ])
    if family == "Logistic Regression":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(**params, random_state=42)),
        ])
    raise ValueError(f"ไม่รู้จัก model family: {family}")


def make_labels(amount, low, high):
    values = amount.to_numpy(dtype=float)
    return np.where(values <= low, "LOW", np.where(values <= high, "MEDIUM", "HIGH"))


def evaluate(train, test, spec, details=False):
    train = train.dropna(subset=FEATURES + ["amount"]).copy()
    test = test.dropna(subset=FEATURES + ["amount"]).copy()
    low = float(train["amount"].quantile(Q_LOW))
    high = float(train["amount"].quantile(Q_HIGH))
    y_train = make_labels(train["amount"], low, high)
    y_test = make_labels(test["amount"], low, high)
    model = create_model(spec)
    model.fit(train[FEATURES], y_train)
    prediction = model.predict(test[FEATURES])
    result = {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "macro_precision": float(precision_score(
            y_test, prediction, labels=CLASS_ORDER, average="macro", zero_division=0
        )),
        "macro_recall": float(recall_score(
            y_test, prediction, labels=CLASS_ORDER, average="macro", zero_division=0
        )),
        "macro_f1": float(f1_score(
            y_test, prediction, labels=CLASS_ORDER, average="macro", zero_division=0
        )),
        "threshold_low": low,
        "threshold_high": high,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
    }
    if details:
        result["confusion_matrix"] = confusion_matrix(
            y_test, prediction, labels=CLASS_ORDER
        ).tolist()
        result["test_class_counts"] = {
            label: int(np.sum(y_test == label)) for label in CLASS_ORDER
        }
    return result


def main():
    started = time.time()
    data = v3.load_data()
    rolling = []
    print(f"\nเริ่มเปรียบเทียบ {len(MODEL_SPECS)} configurations", flush=True)

    for index, spec in enumerate(MODEL_SPECS, 1):
        try:
            folds = []
            for year in [2565, 2566]:
                folds.append(evaluate(
                    data[data["year"].between(2562, year - 1)],
                    data[data["year"] == year], spec,
                ))
            accuracies = [fold["accuracy"] for fold in folds]
            f1_values = [fold["macro_f1"] for fold in folds]
            row = {
                "rank_input": index,
                "name": spec["name"],
                "family": spec["family"],
                "parameters": json.dumps(spec["params"], ensure_ascii=False),
                "mean_accuracy": float(np.mean(accuracies)),
                "min_accuracy": float(np.min(accuracies)),
                "std_accuracy": float(np.std(accuracies)),
                "mean_macro_f1": float(np.mean(f1_values)),
                "accuracy_2565": folds[0]["accuracy"],
                "f1_2565": folds[0]["macro_f1"],
                "accuracy_2566": folds[1]["accuracy"],
                "f1_2566": folds[1]["macro_f1"],
                "status": "OK", "error": "",
            }
            rolling.append(row)
            print(
                f"[{index:02d}/{len(MODEL_SPECS)}] {spec['name']}: "
                f"Mean={row['mean_accuracy']*100:.1f}% "
                f"Min={row['min_accuracy']*100:.1f}% "
                f"F1={row['mean_macro_f1']*100:.1f}%",
                flush=True,
            )
        except Exception as error:
            rolling.append({
                "rank_input": index, "name": spec["name"], "family": spec["family"],
                "status": "ERROR", "error": f"{type(error).__name__}: {error}",
            })
            print(f"[{index:02d}] {spec['name']} ERROR: {error}", flush=True)
        pd.DataFrame(rolling).to_csv(ROLLING_CSV, index=False, encoding="utf-8-sig")

    successful = [row for row in rolling if row["status"] == "OK"]
    successful.sort(
        key=lambda row: (row["mean_accuracy"], row["min_accuracy"], row["mean_macro_f1"]),
        reverse=True,
    )
    if not successful:
        raise RuntimeError("ไม่มีโมเดลที่ทดสอบสำเร็จ")

    print("\nTOP ROLLING VALIDATION", flush=True)
    for rank, row in enumerate(successful[:TOP_CANDIDATES], 1):
        print(
            f"{rank:>2}. {row['name']}: Mean={row['mean_accuracy']*100:.1f}% "
            f"Min={row['min_accuracy']*100:.1f}% F1={row['mean_macro_f1']*100:.1f}%",
            flush=True,
        )

    final_train = data[data["year"].between(2562, 2566)]
    final_test = data[data["year"] == 2567]
    final_results = []
    print("\nFINAL TEST 2567", flush=True)
    for rank, candidate in enumerate(successful[:TOP_CANDIDATES], 1):
        spec = next(item for item in MODEL_SPECS if item["name"] == candidate["name"])
        metrics = evaluate(final_train, final_test, spec, details=True)
        result = {
            "rolling_rank": rank,
            "name": spec["name"],
            "family": spec["family"],
            "parameters": spec["params"],
            "rolling_mean_accuracy": candidate["mean_accuracy"],
            "rolling_min_accuracy": candidate["min_accuracy"],
            "rolling_mean_macro_f1": candidate["mean_macro_f1"],
            **metrics,
        }
        final_results.append(result)
        print(
            f"{rank:>2}. {spec['name']}: Accuracy={metrics['accuracy']*100:.1f}% "
            f"F1={metrics['macro_f1']*100:.1f}%",
            flush=True,
        )

    selected = final_results[0]
    payload = {
        "selection_rule": "select by rolling validation 2565-2566; final test 2567 not used for selection",
        "features": FEATURES,
        "target_threshold_quantiles": [Q_LOW, Q_HIGH],
        "selected_model": selected,
        "final_candidates": final_results,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    with FINAL_JSON.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=str)

    print("\nSELECTED MODEL", flush=True)
    print(
        f"{selected['name']} Accuracy={selected['accuracy']*100:.1f}% "
        f"Macro F1={selected['macro_f1']*100:.1f}%",
        flush=True,
    )
    print(f"Rolling CSV: {ROLLING_CSV}", flush=True)
    print(f"Final JSON:  {FINAL_JSON}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {type(error).__name__}: {error}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
