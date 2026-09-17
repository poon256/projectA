# -*- coding: utf-8 -*-
"""Random Forest experiment search V3 for the mackerel project.

Split used for model selection:
    Locked baseline: train 2562-2566, test 2567 (33/66)
    Rolling validation for selection: 2565 and 2566
    Final test after selection: 2567

Outputs:
    model/output/rf_experiment_v3_rolling.csv
    model/output/rf_experiment_v3_final.json

Required packages:
    pip install pandas numpy scikit-learn pymysql
"""

import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pymysql
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "",
    "database": "projecta",
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
}

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ROLLING_CSV = OUTPUT_DIR / "rf_experiment_v3_rolling.csv"
FINAL_JSON = OUTPUT_DIR / "rf_experiment_v3_final.json"

KEYS = ["station_id", "year", "month"]
CLASS_ORDER = ["LOW", "MEDIUM", "HIGH"]
RANDOM_STATE = 42
TOP_CANDIDATES = 15

THRESHOLDS = [
    (0.31, 0.64),
    (0.32, 0.65),
    (0.30, 0.65),
    (0.33, 0.66),
    (0.34, 0.65),
    (0.35, 0.65),
    (0.35, 0.67),
]

BASE = [
    "sst", "chlorophyll_a", "rainfall", "wind_speed",
    "month", "year", "station_id",
]

FEATURE_SETS = {
    "BASE": BASE,
    "BASE_NO_YEAR": [x for x in BASE if x != "year"],
    "BASE_NO_STATION": [x for x in BASE if x != "station_id"],
    "BASE_SSS": BASE + ["sss"],
    "BASE_DEPTH": BASE + ["depth"],
    "BASE_DEPTH_NO_STATION": [x for x in BASE if x != "station_id"] + ["depth"],
    "BASE_PRESSURE": BASE + ["sea_level_pressure"],
    "BASE_AIR_TEMP": BASE + ["air_temperature"],
    "BASE_WIND_DIR": BASE + ["wind_dir_sin", "wind_dir_cos"],
}

RF_CONFIGS = [
    {
        "config_name": "RF_FULL",
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced",
    },
    {
        "config_name": "RF_DEPTH_6_BAL",
        "n_estimators": 350,
        "max_depth": 6,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced",
    },
    {
        "config_name": "RF_DEPTH_8_BAL",
        "n_estimators": 400,
        "max_depth": 8,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced",
    },
    {
        "config_name": "RF_DEPTH_10_BAL",
        "n_estimators": 400,
        "max_depth": 10,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced",
    },
    {
        "config_name": "RF_LEAF_3_BAL",
        "n_estimators": 400,
        "max_depth": None,
        "min_samples_split": 6,
        "min_samples_leaf": 3,
        "max_features": "sqrt",
        "class_weight": "balanced",
    },
    {
        "config_name": "RF_FEATURE_07_BAL",
        "n_estimators": 400,
        "max_depth": 10,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "max_features": 0.7,
        "class_weight": "balanced_subsample",
    },
]


def fetch_dataframe(connection, sql):
    """Fetch through PyMySQL directly (avoids pandas DBAPI warning/issues)."""
    with connection.cursor() as cursor:
        cursor.execute(sql)
        rows = cursor.fetchall()
    return pd.DataFrame(rows)


def prepare_keys(frame, table_name):
    if frame.empty:
        raise RuntimeError(f"ไม่พบข้อมูลใน {table_name}")
    missing = [column for column in KEYS if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{table_name} ไม่มีคอลัมน์: {', '.join(missing)}")
    for column in KEYS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.dropna(subset=KEYS, inplace=True)
    frame[KEYS] = frame[KEYS].astype("int32")
    duplicated = frame.duplicated(KEYS, keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, KEYS].head(5).to_dict("records")
        raise RuntimeError(
            f"{table_name} มีคีย์ station/year/month ซ้ำหลัง GROUP BY: {examples}"
        )
    return frame


def show_size(name, frame):
    memory_mb = frame.memory_usage(index=True, deep=True).sum() / (1024 ** 2)
    print(f"{name:<24} rows={len(frame):>5,}  memory={memory_mb:>7.2f} MB", flush=True)


def load_data():
    print("\n[1/4] กำลังอ่านข้อมูลจาก MySQL...", flush=True)
    connection = pymysql.connect(**DB_CONFIG)
    try:
        catch = fetch_dataframe(connection, """
            SELECT station_id, year, month, SUM(amount) AS amount
            FROM catch_mackereldata
            WHERE status = 1 AND year BETWEEN 2562 AND 2567
            GROUP BY station_id, year, month
        """)
        marine = fetch_dataframe(connection, """
            SELECT station_id, year, month,
                   AVG(sst) AS sst,
                   AVG(chlorophyll_a) AS chlorophyll_a,
                   AVG(sss) AS sss
            FROM marine_environment
            WHERE status = 1 AND year BETWEEN 2562 AND 2567
            GROUP BY station_id, year, month
        """)
        weather = fetch_dataframe(connection, """
            SELECT station_id, year, month,
                   AVG(rainfall) AS rainfall,
                   AVG(wind_speed) AS wind_speed,
                   AVG(sea_level_pressure) AS sea_level_pressure,
                   AVG(air_temperature) AS air_temperature,
                   AVG(wind_direction) AS wind_direction,
                   MAX(monsoon) AS monsoon,
                   MAX(season) AS season
            FROM weather_data
            WHERE status = 1 AND year BETWEEN 2562 AND 2567
            GROUP BY station_id, year, month
        """)
        station = fetch_dataframe(connection, """
            SELECT id AS station_id, depth
            FROM station
            WHERE status = 1
        """)
    finally:
        connection.close()

    catch = prepare_keys(catch, "catch_mackereldata")
    marine = prepare_keys(marine, "marine_environment")
    weather = prepare_keys(weather, "weather_data")

    if station.empty:
        raise RuntimeError("ไม่พบข้อมูลใน station")
    station["station_id"] = pd.to_numeric(station["station_id"], errors="coerce")
    station.dropna(subset=["station_id"], inplace=True)
    station["station_id"] = station["station_id"].astype("int32")
    if station["station_id"].duplicated().any():
        raise RuntimeError("ตาราง station มี id ซ้ำ จึงไม่สามารถ merge อย่างปลอดภัยได้")

    show_size("Catch", catch)
    show_size("Marine", marine)
    show_size("Weather", weather)
    show_size("Station", station)

    print("\n[2/4] กำลัง merge และตรวจสอบจำนวนแถว...", flush=True)
    data = catch.merge(marine, on=KEYS, how="inner", validate="one_to_one")
    show_size("After marine", data)
    data = data.merge(weather, on=KEYS, how="left", validate="one_to_one")
    show_size("After weather", data)
    data = data.merge(station, on="station_id", how="left", validate="many_to_one")
    show_size("After station", data)

    if len(data) > len(catch):
        raise RuntimeError(
            f"จำนวนแถวหลัง merge ({len(data):,}) มากกว่า catch ({len(catch):,}) "
            "จึงหยุดเพื่อป้องกัน RAM เต็ม"
        )

    numeric_columns = [
        "amount", "sst", "chlorophyll_a", "sss", "rainfall", "wind_speed",
        "sea_level_pressure", "air_temperature", "wind_direction", "depth",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce").astype("float64")

    radians = np.deg2rad(data["wind_direction"] % 360.0)
    data["wind_dir_sin"] = np.sin(radians)
    data["wind_dir_cos"] = np.cos(radians)
    for column in ["monsoon", "season"]:
        data[column] = data[column].astype("string").str.strip()
        data.loc[data[column].isin(["", "None", "nan", "<NA>"]), column] = pd.NA

    # V3 does not drop rows for every optional feature here. Each experiment
    # drops only rows missing the features it actually uses. This keeps BASE
    # identical to randomforestclassifier(5).py.
    report_columns = sorted({feature for features in FEATURE_SETS.values() for feature in features})
    missing_counts = data[report_columns].isna().sum()
    if missing_counts.any():
        details = ", ".join(
            f"{name}={int(count)}" for name, count in missing_counts.items() if count
        )
        print(f"ข้อมูลที่ขาด (จะตัดแยกตามชุดทดลอง): {details}", flush=True)

    if data.empty:
        raise RuntimeError("ไม่มีข้อมูลครบทุก feature หลัง merge")
    counts = data.groupby("year").size().to_dict()
    print(f"จำนวนข้อมูลแยกตามปี: {counts}", flush=True)
    for year in [2562, 2563, 2564, 2565, 2566, 2567]:
        if counts.get(year, 0) == 0:
            raise RuntimeError(f"ไม่มีข้อมูลปี {year}")
    return data.reset_index(drop=True)


def thresholds_from_training(train, q_low, q_high):
    low = float(train["amount"].quantile(q_low))
    high = float(train["amount"].quantile(q_high))
    if not math.isfinite(low) or not math.isfinite(high) or low >= high:
        raise ValueError(f"threshold ไม่ถูกต้อง: low={low}, high={high}")
    return low, high


def make_labels(amount, low, high):
    values = amount.to_numpy(dtype=float)
    return np.where(values <= low, "LOW", np.where(values <= high, "MEDIUM", "HIGH"))


def build_model(features, config):
    categorical = [column for column in features if column in ("monsoon", "season")]
    numeric = [column for column in features if column not in categorical]
    transformers = [("numeric", "passthrough", numeric)]
    if categorical:
        transformers.append(
            ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)
        )
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    rf_args = {key: value for key, value in config.items() if key != "config_name"}
    classifier = RandomForestClassifier(
        **rf_args,
        random_state=RANDOM_STATE,
        n_jobs=1,  # safer on Windows/Python 3.14 and keeps memory predictable
    )
    return Pipeline([("prepare", preprocessor), ("rf", classifier)])


def score_model(train, evaluation, features, q_low, q_high, config, include_details=False):
    train = train.dropna(subset=features + ["amount"]).copy()
    evaluation = evaluation.dropna(subset=features + ["amount"]).copy()
    if train.empty or evaluation.empty:
        raise ValueError("ไม่มีข้อมูลครบสำหรับ feature set นี้")
    low, high = thresholds_from_training(train, q_low, q_high)
    y_train = make_labels(train["amount"], low, high)
    y_eval = make_labels(evaluation["amount"], low, high)
    if len(set(y_train)) < 3:
        raise ValueError(f"training target มีเพียง {len(set(y_train))} classes")
    model = build_model(features, config)
    model.fit(train[features], y_train)
    prediction = model.predict(evaluation[features])
    metrics = {
        "accuracy": float(accuracy_score(y_eval, prediction)),
        "macro_precision": float(
            precision_score(y_eval, prediction, labels=CLASS_ORDER, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_eval, prediction, labels=CLASS_ORDER, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_eval, prediction, labels=CLASS_ORDER, average="macro", zero_division=0)
        ),
        "threshold_low": low,
        "threshold_high": high,
        "train_rows": int(len(train)),
        "eval_rows": int(len(evaluation)),
    }
    if include_details:
        metrics["confusion_matrix"] = confusion_matrix(
            y_eval, prediction, labels=CLASS_ORDER
        ).tolist()
        metrics["train_class_counts"] = {
            label: int(np.sum(y_train == label)) for label in CLASS_ORDER
        }
        metrics["eval_class_counts"] = {
            label: int(np.sum(y_eval == label)) for label in CLASS_ORDER
        }
    return metrics


def compact_result(result):
    return (
        f"Acc={result['accuracy'] * 100:5.1f}%  "
        f"F1={result['macro_f1'] * 100:5.1f}%  "
        f"{result['feature_set']}  q={result['q_low']:.2f}/{result['q_high']:.2f}  "
        f"{result['config_name']}"
    )


def main():
    started = time.time()
    data = load_data()
    baseline_config = {
        "config_name": "LOCKED_BASELINE_683",
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced",
    }
    baseline = score_model(
        data[data["year"].between(2562, 2566)],
        data[data["year"] == 2567],
        BASE, 0.33, 0.66, baseline_config, include_details=True,
    )
    print("\nLOCKED BASELINE (ต้องใกล้เคียง 68.3%)", flush=True)
    print(
        f"Acc={baseline['accuracy']*100:.1f}% F1={baseline['macro_f1']*100:.1f}% "
        f"rows={baseline['train_rows']}/{baseline['eval_rows']} "
        f"threshold={baseline['threshold_low']:.4f}/{baseline['threshold_high']:.4f}",
        flush=True,
    )
    print(f"Class train={baseline['train_class_counts']}", flush=True)
    print(f"Class test ={baseline['eval_class_counts']}", flush=True)
    print(f"Confusion matrix={baseline['confusion_matrix']}", flush=True)

    # Parameter selection uses 2565 and 2566 only. Year 2567 stays untouched
    # until the top candidates have been selected.
    validation_years = [2565, 2566]
    candidate_total = len(THRESHOLDS) * len(FEATURE_SETS) * len(RF_CONFIGS)
    fit_total = candidate_total * len(validation_years)
    print(
        f"\n[3/4] Rolling Validation: {candidate_total} candidates, {fit_total} model fits",
        flush=True,
    )
    rolling_results = []
    thresholds_to_run = THRESHOLDS
    if ROLLING_CSV.exists():
        try:
            checkpoint = pd.read_csv(ROLLING_CSV)
            complete = (
                len(checkpoint) == candidate_total
                and "status" in checkpoint.columns
                and checkpoint["status"].eq("OK").all()
            )
            if complete:
                rolling_results = checkpoint.to_dict("records")
                thresholds_to_run = []
                print(
                    f"พบ checkpoint ครบ {candidate_total} candidates: "
                    "ข้าม Rolling Validation และทำ Final Test ต่อทันที",
                    flush=True,
                )
            else:
                print("checkpoint ยังไม่ครบ จะเริ่ม Rolling Validation ใหม่", flush=True)
        except Exception as error:
            print(f"อ่าน checkpoint ไม่สำเร็จ จะทดลองใหม่: {error}", flush=True)
    experiment = 0

    for q_low, q_high in thresholds_to_run:
        for feature_name, features in FEATURE_SETS.items():
            for config in RF_CONFIGS:
                experiment += 1
                prefix = (
                    f"[{experiment:03d}/{candidate_total}] {feature_name} "
                    f"q={q_low:.2f}/{q_high:.2f} {config['config_name']}"
                )
                try:
                    folds = []
                    for validation_year in validation_years:
                        train = data[data["year"].between(2562, validation_year - 1)]
                        validation = data[data["year"] == validation_year]
                        fold = score_model(
                            train, validation, features, q_low, q_high, config
                        )
                        fold["validation_year"] = validation_year
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
                        **{key: value for key, value in config.items() if key != "config_name"},
                        "mean_accuracy": float(np.mean(accuracies)),
                        "min_accuracy": float(np.min(accuracies)),
                        "std_accuracy": float(np.std(accuracies)),
                        "mean_macro_f1": float(np.mean(f1_values)),
                        "accuracy_2565": folds[0]["accuracy"],
                        "f1_2565": folds[0]["macro_f1"],
                        "rows_2565": folds[0]["eval_rows"],
                        "accuracy_2566": folds[1]["accuracy"],
                        "f1_2566": folds[1]["macro_f1"],
                        "rows_2566": folds[1]["eval_rows"],
                        "status": "OK",
                        "error": "",
                    }
                    rolling_results.append(row)
                    print(
                        f"{prefix} -> Mean={row['mean_accuracy']*100:.1f}% "
                        f"Min={row['min_accuracy']*100:.1f}% "
                        f"F1={row['mean_macro_f1']*100:.1f}%",
                        flush=True,
                    )
                except Exception as error:
                    rolling_results.append({
                        "experiment": experiment,
                        "feature_set": feature_name,
                        "features": ",".join(features),
                        "q_low": q_low,
                        "q_high": q_high,
                        "config_name": config["config_name"],
                        "status": "ERROR",
                        "error": f"{type(error).__name__}: {error}",
                    })
                    print(f"{prefix} -> ERROR: {error}", flush=True)

                # Save a checkpoint after every experiment.
                pd.DataFrame(rolling_results).to_csv(
                    ROLLING_CSV, index=False, encoding="utf-8-sig"
                )

    successful = [row for row in rolling_results if row["status"] == "OK"]
    if not successful:
        raise RuntimeError("การทดลองทุกชุดล้มเหลว กรุณาดูคอลัมน์ error ใน CSV")
    successful.sort(
        key=lambda row: (
            row["mean_accuracy"], row["min_accuracy"], row["mean_macro_f1"]
        ),
        reverse=True,
    )

    print("\nTOP ROLLING VALIDATION (เลือกจากปี 2565-2566 เท่านั้น)", flush=True)
    for rank, row in enumerate(successful[:TOP_CANDIDATES], 1):
        print(
            f"{rank:>2}. Mean={row['mean_accuracy']*100:.1f}% "
            f"Min={row['min_accuracy']*100:.1f}% F1={row['mean_macro_f1']*100:.1f}% "
            f"{row['feature_set']} q={row['q_low']:.2f}/{row['q_high']:.2f} "
            f"{row['config_name']}",
            flush=True,
        )

    print(
        f"\n[4/4] Final Test 2567 เฉพาะ Top {min(TOP_CANDIDATES, len(successful))}...",
        flush=True,
    )
    final_train = data[data["year"].between(2562, 2566)].copy()
    final_test = data[data["year"] == 2567].copy()
    if final_test.empty:
        raise RuntimeError("ไม่พบข้อมูล Final Test ปี 2567")
    final_results = []
    final_errors = []
    for rank, candidate in enumerate(successful[:TOP_CANDIDATES], 1):
        features = FEATURE_SETS[candidate["feature_set"]]
        config = next(
            item for item in RF_CONFIGS if item["config_name"] == candidate["config_name"]
        )
        try:
            metrics = score_model(
                final_train,
                final_test,
                features,
                candidate["q_low"],
                candidate["q_high"],
                config, include_details=True,
            )
            result = {
                "rolling_validation_rank": rank,
                "feature_set": candidate["feature_set"],
                "features": features,
                "q_low": candidate["q_low"],
                "q_high": candidate["q_high"],
                "config_name": candidate["config_name"],
                "parameters": {key: value for key, value in config.items() if key != "config_name"},
                "rolling_mean_accuracy": candidate["mean_accuracy"],
                "rolling_min_accuracy": candidate["min_accuracy"],
                "rolling_mean_macro_f1": candidate["mean_macro_f1"],
                "accuracy_2565": candidate["accuracy_2565"],
                "accuracy_2566": candidate["accuracy_2566"],
                **metrics,
            }
            final_results.append(result)
            print(f"{rank:>2}. {compact_result(result)}", flush=True)
        except Exception as error:
            error_text = f"{type(error).__name__}: {error}"
            final_errors.append({"rolling_validation_rank": rank, "error": error_text})
            print(f"{rank:>2}. FINAL ERROR: {error_text}", flush=True)

    if not final_results:
        raise RuntimeError("Top candidates ไม่สามารถทดสอบกับปี 2567 ได้")
    # Keep Rolling Validation order. Do not re-select a model from Final Test.
    selected = final_results[0]
    payload = {
        "selection_rule": "เลือกจากค่าเฉลี่ย Rolling Validation 2565-2566; เปิด Final Test 2567 หลังเลือก Top เท่านั้น",
        "data_split": {
            "rolling_folds": ["2562-2564 -> 2565", "2562-2565 -> 2566"],
            "final_refit_years": "2562-2566",
            "final_test_year": 2567,
        },
        "locked_baseline": baseline,
        "candidate_count": candidate_total,
        "validation_model_fits": fit_total,
        "successful_experiments": len(successful),
        "elapsed_seconds": round(time.time() - started, 2),
        "selected_model": selected,
        "final_candidates": final_results,
        "final_errors": final_errors,
    }
    with FINAL_JSON.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=str)

    print("\nFINAL 2567 (ไม่ได้ใช้เลือก parameter)", flush=True)
    for rank, row in enumerate(final_results, 1):
        print(f"{rank:>2}. {compact_result(row)}", flush=True)
    print("\nSELECTED MODEL (Rolling Validation rank 1)", flush=True)
    print(compact_result(selected), flush=True)
    print(f"Rolling CSV:    {ROLLING_CSV}", flush=True)
    print(f"Final JSON:     {FINAL_JSON}", flush=True)
    print(f"เวลารวม: {payload['elapsed_seconds']:.2f} วินาที", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nผู้ใช้ยกเลิกการทำงาน ผลที่ทดลองแล้วอยู่ในไฟล์ CSV", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"\nERROR: {type(error).__name__}: {error}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
