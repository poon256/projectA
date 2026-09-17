# -*- coding: utf-8 -*-
"""Random Forest V4: target LOW/MEDIUM confusion without test-set tuning.

Requires rf_experiment_search_v3.py in the same model directory.

Selection:
  Fold 1: train 2562-2564, validate 2565
  Fold 2: train 2562-2565, validate 2566
  Final:  train 2562-2566, test 2567 (top validation candidates only)

V4 experiments:
  - raw month vs cyclic month_sin/month_cos
  - numeric station_id vs one-hot station category
  - class-specific training weights
  - probability decision weights for LOW/MEDIUM
  - ensemble average across random seeds 21, 42 and 84
"""

import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    import rf_experiment_search_v3 as v3
except ImportError as error:
    raise RuntimeError(
        "ต้องวาง rf_experiment_search_v3.py ไว้ในโฟลเดอร์เดียวกับ V4"
    ) from error


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ROLLING_CSV = OUTPUT_DIR / "rf_experiment_v4_rolling.csv"
FINAL_JSON = OUTPUT_DIR / "rf_experiment_v4_final.json"

CLASS_ORDER = ["LOW", "MEDIUM", "HIGH"]
SEEDS = [21, 42, 84]
TOP_CANDIDATES = 15

THRESHOLDS = [(0.32, 0.65), (0.33, 0.66), (0.34, 0.66)]

FEATURE_SETS = {
    "BASE_RAW": [
        "sst", "chlorophyll_a", "rainfall", "wind_speed",
        "month", "year", "station_id",
    ],
    "MONTH_CYCLIC": [
        "sst", "chlorophyll_a", "rainfall", "wind_speed",
        "month_sin", "month_cos", "year", "station_id",
    ],
    "STATION_ONEHOT": [
        "sst", "chlorophyll_a", "rainfall", "wind_speed",
        "month", "year", "station_cat",
    ],
    "CYCLIC_AND_ONEHOT": [
        "sst", "chlorophyll_a", "rainfall", "wind_speed",
        "month_sin", "month_cos", "year", "station_cat",
    ],
}

# Keep the search focused around the 68.3% full-depth baseline.
MODEL_CONFIGS = [
    {
        "config_name": "FULL_BALANCED",
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced",
    },
    {
        "config_name": "FULL_LOW_12_MED_12",
        "n_estimators": 350,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": {"LOW": 1.2, "MEDIUM": 1.2, "HIGH": 1.0},
    },
    {
        "config_name": "FULL_LOW_12_MED_14",
        "n_estimators": 350,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": {"LOW": 1.2, "MEDIUM": 1.4, "HIGH": 1.0},
    },
    {
        "config_name": "LEAF2_LOW_13_MED_13",
        "n_estimators": 350,
        "max_depth": None,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": {"LOW": 1.3, "MEDIUM": 1.3, "HIGH": 1.0},
    },
]

# Multipliers are applied to averaged predict_proba, then argmax is taken.
DECISION_WEIGHTS = {
    "P_NORMAL": {"LOW": 1.00, "MEDIUM": 1.00, "HIGH": 1.00},
    "P_LOW_105": {"LOW": 1.05, "MEDIUM": 1.00, "HIGH": 1.00},
    "P_LOW_110": {"LOW": 1.10, "MEDIUM": 1.00, "HIGH": 1.00},
    "P_MED_105": {"LOW": 1.00, "MEDIUM": 1.05, "HIGH": 1.00},
    "P_LOW_105_MED_105": {"LOW": 1.05, "MEDIUM": 1.05, "HIGH": 1.00},
}


def prepare_data():
    data = v3.load_data()
    radians = 2.0 * np.pi * (data["month"].astype(float) - 1.0) / 12.0
    data["month_sin"] = np.sin(radians)
    data["month_cos"] = np.cos(radians)
    data["station_cat"] = data["station_id"].astype("Int64").astype("string")
    return data


def thresholds_from_training(train, q_low, q_high):
    low = float(train["amount"].quantile(q_low))
    high = float(train["amount"].quantile(q_high))
    if not math.isfinite(low) or not math.isfinite(high) or low >= high:
        raise ValueError(f"threshold ไม่ถูกต้อง: {low}/{high}")
    return low, high


def labels_from_amount(amount, low, high):
    values = amount.to_numpy(dtype=float)
    return np.where(values <= low, "LOW", np.where(values <= high, "MEDIUM", "HIGH"))


def build_pipeline(features, config, seed):
    categorical = [name for name in features if name == "station_cat"]
    numeric = [name for name in features if name not in categorical]
    transformers = [("numeric", "passthrough", numeric)]
    if categorical:
        transformers.append(
            ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)
        )
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    rf_args = {key: value for key, value in config.items() if key != "config_name"}
    classifier = RandomForestClassifier(
        **rf_args, random_state=seed, n_jobs=1
    )
    return Pipeline([("prepare", preprocessor), ("rf", classifier)])


def aligned_probabilities(model, evaluation, features):
    raw = model.predict_proba(evaluation[features])
    model_classes = list(model.named_steps["rf"].classes_)
    aligned = np.zeros((len(evaluation), len(CLASS_ORDER)), dtype=float)
    for target_index, label in enumerate(CLASS_ORDER):
        if label in model_classes:
            aligned[:, target_index] = raw[:, model_classes.index(label)]
    return aligned


def score_ensemble(
    train, evaluation, features, q_low, q_high, config, decision_weights,
    include_details=False,
):
    train = train.dropna(subset=features + ["amount"]).copy()
    evaluation = evaluation.dropna(subset=features + ["amount"]).copy()
    if train.empty or evaluation.empty:
        raise ValueError("ไม่มีข้อมูลครบสำหรับชุดทดลอง")

    low, high = thresholds_from_training(train, q_low, q_high)
    y_train = labels_from_amount(train["amount"], low, high)
    y_eval = labels_from_amount(evaluation["amount"], low, high)
    if len(set(y_train)) < 3:
        raise ValueError("Training target มีไม่ครบ 3 classes")

    probability_sum = np.zeros((len(evaluation), len(CLASS_ORDER)), dtype=float)
    for seed in SEEDS:
        model = build_pipeline(features, config, seed)
        model.fit(train[features], y_train)
        probability_sum += aligned_probabilities(model, evaluation, features)
    probabilities = probability_sum / len(SEEDS)

    multipliers = np.array(
        [decision_weights[label] for label in CLASS_ORDER], dtype=float
    )
    adjusted = probabilities * multipliers
    prediction = np.array(CLASS_ORDER)[np.argmax(adjusted, axis=1)]

    result = {
        "accuracy": float(accuracy_score(y_eval, prediction)),
        "macro_precision": float(precision_score(
            y_eval, prediction, labels=CLASS_ORDER, average="macro", zero_division=0
        )),
        "macro_recall": float(recall_score(
            y_eval, prediction, labels=CLASS_ORDER, average="macro", zero_division=0
        )),
        "macro_f1": float(f1_score(
            y_eval, prediction, labels=CLASS_ORDER, average="macro", zero_division=0
        )),
        "threshold_low": low,
        "threshold_high": high,
        "train_rows": int(len(train)),
        "eval_rows": int(len(evaluation)),
    }
    if include_details:
        result["confusion_matrix"] = confusion_matrix(
            y_eval, prediction, labels=CLASS_ORDER
        ).tolist()
        result["train_class_counts"] = {
            label: int(np.sum(y_train == label)) for label in CLASS_ORDER
        }
        result["eval_class_counts"] = {
            label: int(np.sum(y_eval == label)) for label in CLASS_ORDER
        }
    return result


def checkpoint_is_complete(candidate_total):
    if not ROLLING_CSV.exists():
        return None
    try:
        frame = pd.read_csv(ROLLING_CSV)
        if (
            len(frame) == candidate_total
            and "status" in frame.columns
            and frame["status"].eq("OK").all()
        ):
            return frame.to_dict("records")
    except Exception:
        return None
    return None


def main():
    started = time.time()
    data = prepare_data()

    baseline_config = MODEL_CONFIGS[0]
    baseline = score_ensemble(
        data[data["year"].between(2562, 2566)],
        data[data["year"] == 2567],
        FEATURE_SETS["BASE_RAW"], 0.33, 0.66,
        baseline_config, DECISION_WEIGHTS["P_NORMAL"], True,
    )
    print("\nV4 ENSEMBLE BASELINE (3 seeds)", flush=True)
    print(
        f"Acc={baseline['accuracy']*100:.1f}% F1={baseline['macro_f1']*100:.1f}% "
        f"threshold={baseline['threshold_low']:.4f}/{baseline['threshold_high']:.4f}",
        flush=True,
    )
    print(f"Confusion matrix={baseline['confusion_matrix']}", flush=True)

    candidate_total = (
        len(THRESHOLDS) * len(FEATURE_SETS)
        * len(MODEL_CONFIGS) * len(DECISION_WEIGHTS)
    )
    total_fits = candidate_total * 2 * len(SEEDS)
    print(
        f"\nRolling search: {candidate_total} candidates, {total_fits} RF fits",
        flush=True,
    )

    results = checkpoint_is_complete(candidate_total)
    if results is not None:
        print("พบ checkpoint ครบแล้ว ข้าม Rolling search", flush=True)
    else:
        results = []
        experiment = 0
        for q_low, q_high in THRESHOLDS:
            for feature_name, features in FEATURE_SETS.items():
                for config in MODEL_CONFIGS:
                    for decision_name, decision in DECISION_WEIGHTS.items():
                        experiment += 1
                        prefix = (
                            f"[{experiment:03d}/{candidate_total}] {feature_name} "
                            f"q={q_low:.2f}/{q_high:.2f} "
                            f"{config['config_name']} {decision_name}"
                        )
                        try:
                            folds = []
                            for validation_year in [2565, 2566]:
                                fold = score_ensemble(
                                    data[data["year"].between(2562, validation_year - 1)],
                                    data[data["year"] == validation_year],
                                    features, q_low, q_high, config, decision,
                                )
                                folds.append(fold)
                            accuracies = [fold["accuracy"] for fold in folds]
                            f1_values = [fold["macro_f1"] for fold in folds]
                            row = {
                                "experiment": experiment,
                                "feature_set": feature_name,
                                "features": ",".join(features),
                                "q_low": q_low,
                                "q_high": q_high,
                                "config_name": config["config_name"],
                                "decision_name": decision_name,
                                "mean_accuracy": float(np.mean(accuracies)),
                                "min_accuracy": float(np.min(accuracies)),
                                "std_accuracy": float(np.std(accuracies)),
                                "mean_macro_f1": float(np.mean(f1_values)),
                                "accuracy_2565": folds[0]["accuracy"],
                                "f1_2565": folds[0]["macro_f1"],
                                "accuracy_2566": folds[1]["accuracy"],
                                "f1_2566": folds[1]["macro_f1"],
                                "status": "OK",
                                "error": "",
                            }
                            results.append(row)
                            print(
                                f"{prefix} -> Mean={row['mean_accuracy']*100:.1f}% "
                                f"Min={row['min_accuracy']*100:.1f}% "
                                f"F1={row['mean_macro_f1']*100:.1f}%",
                                flush=True,
                            )
                        except Exception as error:
                            results.append({
                                "experiment": experiment,
                                "feature_set": feature_name,
                                "features": ",".join(features),
                                "q_low": q_low,
                                "q_high": q_high,
                                "config_name": config["config_name"],
                                "decision_name": decision_name,
                                "status": "ERROR",
                                "error": f"{type(error).__name__}: {error}",
                            })
                            print(f"{prefix} -> ERROR: {error}", flush=True)
                        pd.DataFrame(results).to_csv(
                            ROLLING_CSV, index=False, encoding="utf-8-sig"
                        )

    successful = [row for row in results if row["status"] == "OK"]
    successful.sort(
        key=lambda row: (
            row["mean_accuracy"], row["min_accuracy"], row["mean_macro_f1"]
        ), reverse=True,
    )
    if not successful:
        raise RuntimeError("V4 ไม่มี candidate ที่ทดลองสำเร็จ")

    print("\nTOP ROLLING VALIDATION", flush=True)
    for rank, row in enumerate(successful[:TOP_CANDIDATES], 1):
        print(
            f"{rank:>2}. Mean={row['mean_accuracy']*100:.1f}% "
            f"Min={row['min_accuracy']*100:.1f}% F1={row['mean_macro_f1']*100:.1f}% "
            f"{row['feature_set']} {row['config_name']} {row['decision_name']} "
            f"q={row['q_low']:.2f}/{row['q_high']:.2f}",
            flush=True,
        )

    final_train = data[data["year"].between(2562, 2566)]
    final_test = data[data["year"] == 2567]
    final_results = []
    final_errors = []
    print("\nFINAL TEST 2567", flush=True)
    for rank, candidate in enumerate(successful[:TOP_CANDIDATES], 1):
        try:
            config = next(
                item for item in MODEL_CONFIGS
                if item["config_name"] == candidate["config_name"]
            )
            decision = DECISION_WEIGHTS[candidate["decision_name"]]
            metrics = score_ensemble(
                final_train, final_test,
                FEATURE_SETS[candidate["feature_set"]],
                candidate["q_low"], candidate["q_high"],
                config, decision, True,
            )
            result = {
                "rolling_validation_rank": rank,
                "feature_set": candidate["feature_set"],
                "features": FEATURE_SETS[candidate["feature_set"]],
                "q_low": candidate["q_low"],
                "q_high": candidate["q_high"],
                "config_name": candidate["config_name"],
                "parameters": {k: v for k, v in config.items() if k != "config_name"},
                "decision_name": candidate["decision_name"],
                "decision_weights": decision,
                "seeds": SEEDS,
                "rolling_mean_accuracy": candidate["mean_accuracy"],
                "rolling_min_accuracy": candidate["min_accuracy"],
                "rolling_mean_macro_f1": candidate["mean_macro_f1"],
                "accuracy_2565": candidate["accuracy_2565"],
                "accuracy_2566": candidate["accuracy_2566"],
                **metrics,
            }
            final_results.append(result)
            print(
                f"{rank:>2}. Acc={metrics['accuracy']*100:.1f}% "
                f"F1={metrics['macro_f1']*100:.1f}% "
                f"{candidate['feature_set']} {candidate['config_name']} "
                f"{candidate['decision_name']}",
                flush=True,
            )
        except Exception as error:
            final_errors.append({"rolling_validation_rank": rank, "error": str(error)})
            print(f"{rank:>2}. FINAL ERROR: {error}", flush=True)

    if not final_results:
        raise RuntimeError("Top V4 candidates ไม่สามารถทดสอบปี 2567 ได้")

    # Rank 1 remains the selected model; Final Test never selects parameters.
    selected = final_results[0]
    payload = {
        "selection_rule": "select by rolling validation 2565-2566; final 2567 is untouched until selection",
        "locked_v4_ensemble_baseline": baseline,
        "candidate_count": candidate_total,
        "random_forest_fit_count": total_fits,
        "selected_model": selected,
        "final_candidates": final_results,
        "final_errors": final_errors,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    with FINAL_JSON.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=str)

    print("\nSELECTED MODEL (Rolling rank 1)", flush=True)
    print(
        f"Accuracy={selected['accuracy']*100:.1f}% "
        f"Macro F1={selected['macro_f1']*100:.1f}%",
        flush=True,
    )
    print(f"Rolling CSV: {ROLLING_CSV}", flush=True)
    print(f"Final JSON:  {FINAL_JSON}", flush=True)
    print(f"เวลารวม: {payload['elapsed_seconds']:.2f} วินาที", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nยกเลิกการทำงาน ผลที่เสร็จแล้วอยู่ใน checkpoint CSV", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"\nERROR: {type(error).__name__}: {error}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
