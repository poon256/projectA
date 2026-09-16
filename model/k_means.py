import pandas as pd
import mysql.connector

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import f_oneway


# =========================================================
# Connect MySQL
# =========================================================

conn = mysql.connector.connect(
    host="127.0.0.1",
    port=3306,
    user="root",
    password="",
    database="projecta",
    use_pure=True
)


# =========================================================
# อ่านข้อมูลจาก dataset_ml
# 1 แถว = station + year + month + equipment
# =========================================================

sql = """
SELECT
    station_id,
    year,
    month,
    equipment_id,

    SUM(amount) AS amount,

    AVG(sst) AS sst,
    AVG(chlorophyll_a) AS chlorophyll_a,
    AVG(rainfall) AS rainfall,
    AVG(wind_speed) AS wind_speed,
    AVG(sea_level_pressure) AS sea_level_pressure

FROM dataset_ml

GROUP BY
    station_id,
    year,
    month,
    equipment_id

ORDER BY
    year,
    month,
    station_id,
    equipment_id
"""

df = pd.read_sql(sql, conn)


# =========================================================
# ทำความสะอาดข้อมูล
# =========================================================

df = df.dropna(
    subset=[
        "amount",
        "sst",
        "chlorophyll_a",
        "rainfall",
        "wind_speed",
        "sea_level_pressure"
    ]
)

df = df[df["amount"] > 0].copy()

print("จำนวนข้อมูลที่ใช้:", len(df))


# =========================================================
# รวมข้อมูลระดับสถานี + ปี + เดือน
#
# equipment_id ไม่ใช้ในการ Clustering
# เพราะ K-Means ต้องการจัดกลุ่มตามสภาพแวดล้อม
# =========================================================

df_cluster = (
    df.groupby(
        ["station_id", "year", "month"]
    )
    .agg({
        "amount": "sum",
        "sst": "mean",
        "chlorophyll_a": "mean",
        "rainfall": "mean",
        "wind_speed": "mean",
        "sea_level_pressure": "mean"
    })
    .reset_index()
)

print(
    "จำนวนข้อมูลสำหรับ Clustering:",
    len(df_cluster)
)


# =========================================================
# Feature สำหรับ K-Means
# =========================================================

features = [
    "sst",
    "chlorophyll_a",
    "rainfall",
    "wind_speed",
    "sea_level_pressure"
]

X = df_cluster[features].copy()


# =========================================================
# Standardize
# =========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# =========================================================
# ทดลอง K = 2 ถึง 8
# =========================================================

best_k = None
best_score = -1

wcss = []
silhouette_scores = []
dbi_scores = []

print("\n=== ทดสอบจำนวน Cluster ===")

for k in range(2, 9):

    kmeans_test = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    labels = kmeans_test.fit_predict(X_scaled)

    # WCSS
    wcss.append(kmeans_test.inertia_)

    # Silhouette
    score = silhouette_score(
        X_scaled,
        labels
    )

    # Davies-Bouldin Index
    dbi = davies_bouldin_score(
        X_scaled,
        labels
    )

    silhouette_scores.append(score)
    dbi_scores.append(dbi)

    print(
        f"K = {k} | "
        f"WCSS = {kmeans_test.inertia_:.2f} | "
        f"Silhouette = {score:.4f} | "
        f"DBI = {dbi:.4f}"
    )

    # เลือก K จาก Silhouette สูงสุด
    if score > best_score:
        best_score = score
        best_k = k


print("\n================================")
print("Best K =", best_k)
print(
    "Best Silhouette Score =",
    round(best_score, 4)
)
print("================================")


# =========================================================
# พล็อตกราฟประเมิน K-Means
# =========================================================

fig, axs = plt.subplots(
    1,
    3,
    figsize=(18, 5)
)


# -------------------------
# 1. Elbow Method
# -------------------------

axs[0].plot(
    range(2, 9),
    wcss,
    marker="o"
)

axs[0].set_title(
    "Elbow Method (Look for Knee)"
)

axs[0].set_xlabel(
    "Number of Clusters (K)"
)

axs[0].set_ylabel(
    "WCSS (Inertia)"
)

axs[0].grid(True)


# -------------------------
# 2. Silhouette Score
# -------------------------

axs[1].plot(
    range(2, 9),
    silhouette_scores,
    marker="o"
)

axs[1].set_title(
    "Silhouette Score (Higher is Better)"
)

axs[1].set_xlabel(
    "Number of Clusters (K)"
)

axs[1].set_ylabel(
    "Score"
)

axs[1].grid(True)


# -------------------------
# 3. Davies-Bouldin Index
# -------------------------

axs[2].plot(
    range(2, 9),
    dbi_scores,
    marker="o"
)

axs[2].set_title(
    "Davies-Bouldin Index (Lower is Better)"
)

axs[2].set_xlabel(
    "Number of Clusters (K)"
)

axs[2].set_ylabel(
    "Index Value"
)

axs[2].grid(True)


plt.tight_layout()

plt.show()


# =========================================================
# สร้าง K-Means ตัวสุดท้าย
# =========================================================

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=20
)

df_cluster["cluster"] = (
    kmeans.fit_predict(X_scaled)
)


# =========================================================
# เรียงหมายเลข Cluster ตามปริมาณจับปลา
#
# Cluster 0 = ปริมาณจับเฉลี่ยน้อยกว่า
# Cluster 1 = ปริมาณจับเฉลี่ยมากกว่า
# ถ้ามี K=3 ก็จะเรียง น้อย -> กลาง -> มาก
# =========================================================

cluster_amount = (
    df_cluster
    .groupby("cluster")["amount"]
    .mean()
    .sort_values()
)

mapping = {
    old_cluster: new_cluster
    for new_cluster, old_cluster
    in enumerate(cluster_amount.index)
}

df_cluster["cluster"] = (
    df_cluster["cluster"]
    .map(mapping)
)


# =========================================================
# PCA
# =========================================================

pca = PCA(
    n_components=2
)

pca_components = (
    pca.fit_transform(X_scaled)
)

df_cluster["PC1"] = (
    pca_components[:, 0]
)

df_cluster["PC2"] = (
    pca_components[:, 1]
)


# =========================================================
# PCA Explained Variance
# =========================================================

variance_ratio = (
    pca.explained_variance_ratio_
    * 100
)

print("\n=== PCA Explained Variance ===")

print(
    f"PC1 = {variance_ratio[0]:.2f}%"
)

print(
    f"PC2 = {variance_ratio[1]:.2f}%"
)

print(
    f"รวม = {variance_ratio.sum():.2f}%"
)


# =========================================================
# PCA Scatter Plot
# =========================================================

plt.figure(
    figsize=(8, 6)
)

sns.scatterplot(
    x="PC1",
    y="PC2",
    hue="cluster",
    palette="viridis",
    data=df_cluster,
    s=70,
    alpha=0.8,
    edgecolor="w"
)

plt.title(
    f"PCA Scatter Plot of Fish Catch Clusters (K={best_k})"
)

plt.xlabel(
    f"Principal Component 1 "
    f"({variance_ratio[0]:.2f}%)"
)

plt.ylabel(
    f"Principal Component 2 "
    f"({variance_ratio[1]:.2f}%)"
)

plt.legend(
    title="Cluster (Sorted by Amount)",
    loc="best"
)

plt.grid(
    True,
    linestyle="--",
    alpha=0.5
)

plt.show()


# =========================================================
# แสดงผลการจัดกลุ่ม
# =========================================================

print("\n=== ผลการจัดกลุ่ม ===")

print(
    df_cluster[
        [
            "station_id",
            "year",
            "month",
            "amount",
            "sst",
            "chlorophyll_a",
            "rainfall",
            "wind_speed",
            "sea_level_pressure",
            "cluster"
        ]
    ]
    .head(20)
    .to_string(index=False)
)


# =========================================================
# สรุปแต่ละ Cluster
# =========================================================

print(
    "\n=== Cluster Summary (Mean Baseline) ==="
)

summary = (
    df_cluster
    .groupby("cluster")
    .agg({
        "amount": "mean",
        "sst": "mean",
        "chlorophyll_a": "mean",
        "rainfall": "mean",
        "wind_speed": "mean",
        "sea_level_pressure": "mean"
    })
)

print(
    summary.to_string()
)


# =========================================================
# One-Way ANOVA
#
# ตรวจว่าค่าของแต่ละ Feature แตกต่างกันระหว่าง Cluster หรือไม่
# =========================================================

print(
    "\n========== ANOVA Test =========="
)

for var in features:

    groups = [
        group[var].values
        for _, group
        in df_cluster.groupby("cluster")
    ]

    F, p = f_oneway(*groups)

    print(
        f"{var:20s} "
        f"F = {F:10.3f}  "
        f"p-value = {p:.5f}"
    )


# =========================================================
# Update Cluster กลับไป dataset_ml
#
# Cluster เป็นระดับ station + year + month
# ดังนั้น equipment ทุกตัวในเดือนเดียวกัน
# จะได้รับ Cluster เดียวกัน
# =========================================================

cursor = conn.cursor()

for _, row in df_cluster.iterrows():

    cursor.execute(
        """
        UPDATE dataset_ml

        SET cluster = %s

        WHERE station_id = %s
        AND year = %s
        AND month = %s
        """,

        (
            int(row["cluster"]),
            int(row["station_id"]),
            int(row["year"]),
            int(row["month"])
        )
    )


conn.commit()


print(
    "\nบันทึก Cluster กลับลง database เรียบร้อย"
)

print("Finish")


# =========================================================
# ปิด Connection
# =========================================================

cursor.close()

conn.close()