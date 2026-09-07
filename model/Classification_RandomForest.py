# -*- coding: utf-8 -*-
import pandas as pd
import mysql.connector
import os
import sys
import json
import base64
import warnings


warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "projecta",
    "charset": "utf8mb4"
}
PROJECT_DIR = os.path.dirname(BASE_DIR)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output"
)

def send_json(data):
    """
    ส่ง JSON กลับไปยัง PHP
    """
    print(
        json.dumps(
            data,
            ensure_ascii=False
        )
    )

def decode_province(value, is_base64=False):

    if is_base64:

        try:

            return base64.b64decode(
                value
            ).decode(
                "utf-8"
            ).strip()

        except Exception:

            return value.strip()

    return value.strip()

def read_args():

    if len(sys.argv) < 4:

        send_json({

            "status": "error",

            "message": "Missing arguments: province, year, month"

        })

        sys.exit(0)

    is_base64 = (
        len(sys.argv) >= 5
        and sys.argv[4] == "--b64"
    )

    province = decode_province(
        sys.argv[1],
        is_base64
    )

    try:

        year = int(sys.argv[2])

        month = int(sys.argv[3])

    except Exception:

        send_json({

            "status": "error",

            "message": "Year and month must be numbers"

        })

        sys.exit(0)

    if province == "":

        send_json({

            "status": "error",

            "message": "Province is empty"

        })

        sys.exit(0)

    if month < 1 or month > 12:

        send_json({

            "status": "error",

            "message": "Month must be 1-12"

        })

        sys.exit(0)

    return province, year, month

def load_sql():

    try:

        conn = mysql.connector.connect(
            **DB_CONFIG
        )

        # =========================
        # ข้อมูลปลาทู
        # =========================

        fish_sql = """
            SELECT
                station_id,
                year AS ปี,
                month AS เดือน,
                amount AS `ปริมาณ (ตัน)`
            FROM catch_mackereldata
            WHERE status = 1
        """

        fish = pd.read_sql(
            fish_sql,
            conn
        )

        # =========================
        # ข้อมูลสิ่งแวดล้อม
        # =========================

        env_sql = """
        SELECT
        m.station_id,
        m.year AS ปี,
        m.month AS เดือน,
        m.sst AS `อุณหภูมิผิวทะเล`,
        m.chlorophyll_a AS `คลอโรฟิลล์-เอ`,
        AVG(w.rainfall) AS rainfall,
        AVG(w.wind_speed) AS wind_speed
        FROM marine_environment m
        LEFT JOIN weather_data w
        ON m.station_id = w.station_id
        AND m.year = w.year
        AND m.month = w.month
        WHERE m.status = 1
        GROUP BY
        m.station_id,
        m.year,
        m.month,
        m.sst,
        m.chlorophyll_a
        """

        env = pd.read_sql(
            env_sql,
            conn
        )

        conn.close()

        # =========================
        # ตรวจสอบข้อมูล
        # =========================

        if fish.empty:

            send_json({
                "status": "error",
                "message": "ไม่พบข้อมูลปลาทูในฐานข้อมูล"
            })

            sys.exit(0)

        if env.empty:

            send_json({
                "status": "error",
                "message": "ไม่พบข้อมูลสิ่งแวดล้อมในฐานข้อมูล"
            })

            sys.exit(0)

        return fish, env

    except Exception as e:

        send_json({
            "status": "error",
            "message": f"MySQL Error: {str(e)}"
        })

        sys.exit(0)


def add_province(fish, env):

    station_map = {

        1: "เพชรบุรี",
        2: "สมุทรสงคราม",
        3: "สมุทรสาคร",
        4: "ชลบุรี",
        5: "สมุทรปราการ"

    }

    fish["จังหวัด"] = (
        fish["station_id"]
        .map(station_map)
    )

    env["จังหวัด"] = (
        env["station_id"]
        .map(station_map)
    )

    return fish, env        

def convert_month_columns(fish, env):

    month_map = {

        "มกราคม": 1,
        "กุมภาพันธ์": 2,
        "มีนาคม": 3,
        "เมษายน": 4,
        "พฤษภาคม": 5,
        "มิถุนายน": 6,
        "กรกฎาคม": 7,
        "สิงหาคม": 8,
        "กันยายน": 9,
        "ตุลาคม": 10,
        "พฤศจิกายน": 11,
        "ธันวาคม": 12,

    }

    for data in [fish, env]:

        data["จังหวัด"] = data["จังหวัด"].astype(str).str.strip()

        data["เดือน"] = data["เดือน"].astype(str).str.strip()

        data["เดือน"] = data["เดือน"].map(month_map).fillna(
            pd.to_numeric(
                data["เดือน"],
                errors="coerce"
            )
        )

        data["เดือน"] = pd.to_numeric(
            data["เดือน"],
            errors="coerce"
        )

        data["ปี"] = pd.to_numeric(
            data["ปี"],
            errors="coerce"
        )

    fish = fish.dropna(
        subset=[
            "จังหวัด",
            "ปี",
            "เดือน"
        ]
    )

    env = env.dropna(
        subset=[
            "จังหวัด",
            "ปี",
            "เดือน"
        ]
    )

    fish["ปี"] = fish["ปี"].astype(int)
    fish["เดือน"] = fish["เดือน"].astype(int)

    env["ปี"] = env["ปี"].astype(int)
    env["เดือน"] = env["เดือน"].astype(int)

    return fish, env

def prepare_data(
    fish,
    env,
    province
):

    fish_p = fish[
        fish["จังหวัด"] == province
    ].copy()

    env_p = env[
        env["จังหวัด"] == province
    ].copy()

    if fish_p.empty or env_p.empty:

        send_json({

            "status": "error",

            "message": f"ไม่พบจังหวัดดังกล่าว : {province}"

        })

        sys.exit(0)

    fish_monthly = (

        fish_p

        .groupby([
            "ปี",
            "เดือน"
        ])["ปริมาณ (ตัน)"]

        .sum()

        .reset_index()

        .rename(
            columns={
                "ปริมาณ (ตัน)": "catch"
            }
        )

    )

    df = pd.merge(

        env_p,

        fish_monthly,

        on=[
            "ปี",
            "เดือน"
        ],

        how="inner"

    )

    df = df.dropna(

        subset=[

            "อุณหภูมิผิวทะเล",

            "คลอโรฟิลล์-เอ",

            "ปี",

            "เดือน",

            "catch"

        ]

    )

    if df.empty:

        send_json({

            "status": "error",

            "message": "Merge ข้อมูลไม่สำเร็จ"

        })

        sys.exit(0)

    return (

        fish_monthly,

        env_p,

        df

    )

def create_density_class(train_df):

    train_df = train_df.copy()

    def classify_density(catch):

        if catch >= 1 and catch <= 20:
            return "LOW"

        elif catch > 20 and catch <= 40:
            return "MEDIUM"

        elif catch > 40:
            return "HIGH"

        else:
            return None

    train_df["density_level"] = train_df["catch"].apply(
        classify_density
    )

    return train_df

def train_random_forest(df, selected_year):

    train_df = df[
        df["ปี"] < selected_year
    ].copy()

    if len(train_df) < 6:
        train_df = df.copy()

    if len(train_df) < 6:

        send_json({
            "status": "error",
            "message": "ข้อมูลสำหรับ Train ไม่เพียงพอ"
        })

        sys.exit(0)

    train_df = create_density_class(train_df)

    # ลบข้อมูลที่ไม่มี class
    train_df = train_df.dropna(
        subset=["density_level"]
    )

    X = train_df[
        [
            "อุณหภูมิผิวทะเล",
            "คลอโรฟิลล์-เอ",
            "rainfall",
            "wind_speed",
            "เดือน"
        ]
    ]

    y = train_df["density_level"]

    # ตรวจสอบจำนวน class ก่อนแบ่ง Train/Test
    class_counts = y.value_counts()

    if len(class_counts) < 2:
        send_json({
            "status": "error",
            "message": "ข้อมูลต้องมี Density Level อย่างน้อย 2 ระดับ"
        })
        sys.exit(0)

    # ถ้าแต่ละ class มีข้อมูลอย่างน้อย 2 ตัว
    # ใช้ stratify เพื่อรักษาสัดส่วนของแต่ละ class
    use_stratify = class_counts.min() >= 2

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if use_stratify else None
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    # Train
    model.fit(
        X_train,
        y_train
    )

    # Predict Test
    y_pred = model.predict(
        X_test
    )

    # Accuracy
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    return model, accuracy

def predict_density_level(
    model,
    env_p,
    selected_year,
    selected_month
):

    month_env = env_p[
        (env_p["ปี"] == selected_year)
        &
        (env_p["เดือน"] == selected_month)
    ].copy()

    if month_env.empty:

        month_env = (

            env_p

            .groupby("เดือน")[

                [

                    "อุณหภูมิผิวทะเล",

                    "คลอโรฟิลล์-เอ",

                    "rainfall",

                    "wind_speed"

                ]

            ]

            .mean()

            .reset_index()

        )

        month_env = month_env[
            month_env["เดือน"] == selected_month
        ]

    if month_env.empty:

        send_json({

            "status": "error",

            "message": "ไม่พบข้อมูลสำหรับเดือนที่เลือก"

        })

        sys.exit(0)

    X_predict = pd.DataFrame(
    [
        {
            "อุณหภูมิผิวทะเล":
                float(
                    month_env.iloc[0]["อุณหภูมิผิวทะเล"]
                ),

            "คลอโรฟิลล์-เอ":
                float(
                    month_env.iloc[0]["คลอโรฟิลล์-เอ"]
                ),

            "rainfall":
                float(
                    month_env.iloc[0]["rainfall"]
                ),

            "wind_speed":
                float(
                    month_env.iloc[0]["wind_speed"]
                ),

            "เดือน":
                int(selected_month)
        }
    ]
)

    level = model.predict(
        X_predict
    )[0]

    probability_values = model.predict_proba(
        X_predict
    )[0]

    class_names = model.classes_

    probability = {}

    for label, value in zip(
        class_names,
        probability_values
    ):

        probability[label] = round(
            float(value) * 100,
            2
        )

    level_info = {

        "LOW": {

            "description": "Low Density Area",

            "badge_color": "success"

        },

        "MEDIUM": {

            "description": "Medium Density Area",

            "badge_color": "warning"

        },

        "HIGH": {

            "description": "High Density Area",

            "badge_color": "danger"

        }

    }

    return {

        "level": level,

        "description": level_info[level]["description"],

        "badge_color": level_info[level]["badge_color"],

        "probability": probability,

        "sst":

            float(
                month_env.iloc[0]["อุณหภูมิผิวทะเล"]
            ),

        "chlor_a":

            float(
                month_env.iloc[0]["คลอโรฟิลล์-เอ"]
            ),

        "rainfall":

        float(
            month_env.iloc[0]["rainfall"]
            ),

        "wind_speed":

        float(
            month_env.iloc[0]["wind_speed"]
            )

    }
def save_probability_graph(probability):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    labels = list(
        probability.keys()
    )

    values = list(
        probability.values()
    )

    plt.figure(figsize=(6,4))

    plt.bar(
        labels,
        values
    )

    plt.ylim(0,100)

    plt.ylabel("Probability (%)")

    plt.title(
        "Random Forest Classification Probability"
    )

    plt.tight_layout()

    filename = "classification_probability.png"

    save_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    plt.savefig(
        save_path,
        dpi=160
    )

    plt.close()

    return "../model/output/" + filename

def main():

    province, selected_year, selected_month = read_args()

    # โหลดข้อมูลจาก MySQL
    fish, env = load_sql()

    # แปลง station_id เป็นชื่อจังหวัด
    fish, env = add_province(
        fish,
        env
    )

    # จัดการปีและเดือน
    fish, env = convert_month_columns(
        fish,
        env
    )

    # เตรียมข้อมูล
    fish_monthly, env_p, df = prepare_data(
        fish,
        env,
        province
    )

    # Train Random Forest
    model, accuracy = train_random_forest(
        df,
        selected_year
    )

    # Predict จังหวัด / ปี / เดือนที่เลือก
    result = predict_density_level(
        model,
        env_p,
        selected_year,
        selected_month
    )

    # สร้างกราฟ Probability
    probability_graph = save_probability_graph(
        result["probability"]
    )

    # ส่งผลกลับ PHP
    send_json({

        "status": "success",

        "province": province,

        "year": int(selected_year),

        "month": int(selected_month),

        "level": result["level"],

        "description": result["description"],

        "badge_color": result["badge_color"],

        "sst": result["sst"],

        "chlor_a": result["chlor_a"],

        "rainfall": result["rainfall"],

        "wind_speed": result["wind_speed"],

        "probability": result["probability"],

        "probability_graph": probability_graph,

        "accuracy": round(
            float(accuracy) * 100,
            2
        )

    })

if __name__ == "__main__":

    try:

        main()

    except Exception as e:

        send_json({

            "status": "error",

            "message": str(e)

        })


