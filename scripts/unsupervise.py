import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
import warnings

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Tahoma", "Leelawadee UI", "Arial"]
matplotlib.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "charts", "unsupervised")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 1. โหลดข้อมูล (โหลดข้อมูล CSV ที่ทำความสะอาดแล้ว)
print("=" * 60)
print(" 1. กำลังโหลดข้อมูลที่ทำความสะอาดแล้ว")
print("=" * 60)

df = pd.read_csv(os.path.join(DATA_DIR, "CatFood_cleaned.csv"))
print(f"   Shape: {df.shape}")


# 2. เลือกฟีเจอร์สำหรับการจัดกลุ่ม (เลือกคุณลักษณะที่เกี่ยวข้องสำหรับ clustering)
print("\n" + "=" * 60)
print(" 2. กำลังเลือกฟีเจอร์สำหรับการจัดกลุ่ม")
print("=" * 60)

factor_cols = [
    "factor_natural", "factor_imported", "factor_taste",
    "factor_foreign", "factor_brand_fame",
]
pkg_cols = [
    "pkg_premium", "pkg_cat_image", "pkg_kibble_image",
    "pkg_ingredient_image", "pkg_eco_friendly",
    "pkg_origin_symbol", "pkg_benefit_symbol", "pkg_guarantee",
]
option_want_cols = [f"opt{i}_want_buy" for i in range(1, 11)]

feature_cols = factor_cols + pkg_cols + option_want_cols
df_feat = df[feature_cols].dropna().copy()
print(f"   Features: {len(feature_cols)} columns, {len(df_feat)} rows")

# ปรับขนาด (ทำให้ข้อมูลอยู่ในมาตราส่วนเดียวกัน)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_feat)


# 3. สถิติเชิงพรรณนา (แสดงค่าเฉลี่ย ค่ามิน ค่าแม็กซ์ เป็นต้น)
print("\n" + "=" * 60)
print(" 3. สถิติเชิงพรรณนา")
print("=" * 60)

desc = df_feat.describe().round(2)
print(desc.to_string())


# 4. การวิเคราะห์ความสัมพันธ์ (หา correlation ระหว่างฟีเจอร์)
print("\n" + "=" * 60)
print(" 4. การวิเคราะห์ความสัมพันธ์")
print("=" * 60)

corr = df_feat.corr()

fig, ax = plt.subplots(figsize=(16, 12))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5,
            annot_kws={"size": 6})
ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
path_corr = os.path.join(OUTPUT_DIR, "unsup_1_correlation.png")
fig.savefig(path_corr, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved: {path_corr}")


# 5. K-Means Clustering (เทคนิคที่ 1 - จัดกลุ่มข้อมูลเป็น 3 กลุ่ม)
print("\n" + "=" * 60)
print(" 5. K-Means Clustering (k=3)")
print("=" * 60)

# 5.1 วิธี Elbow (หาจำนวน cluster ที่เหมาะสม)
inertias = []
sil_scores = []
K_range = range(2, 8)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(K_range, inertias, "bo-", linewidth=2, markersize=8)
ax1.axvline(x=3, color="red", linestyle="--", alpha=0.7, label="k=3")
ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("Inertia")
ax1.set_title("Elbow Method", fontweight="bold")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(K_range, sil_scores, "go-", linewidth=2, markersize=8)
ax2.axvline(x=3, color="red", linestyle="--", alpha=0.7, label="k=3")
ax2.set_xlabel("Number of Clusters (k)")
ax2.set_ylabel("Silhouette Score")
ax2.set_title("Silhouette Score", fontweight="bold")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
path_elbow = os.path.join(OUTPUT_DIR, "unsup_2_elbow_silhouette.png")
fig.savefig(path_elbow, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved: {path_elbow}")

# 5.2 สร้างโมเดล K-Means โดยกำหนด k=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_feat["cluster"] = kmeans.fit_predict(X_scaled)
df["cluster"] = -1
df.loc[df_feat.index, "cluster"] = df_feat["cluster"]

sil = silhouette_score(X_scaled, df_feat["cluster"])
print(f"   Silhouette Score (k=3): {sil:.4f}")
print(f"   Cluster sizes: {df_feat['cluster'].value_counts().sort_index().to_dict()}")


# 6. PCA Dimensionality Reduction (เทคนิคที่ 2 - ลดมิติข้อมูลให้เหลือ 2 มิติ)
print("\n" + "=" * 60)
print(" 6. PCA Dimensionality Reduction")
print("=" * 60)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"   Explained variance: PC1={pca.explained_variance_ratio_[0]:.4f}, PC2={pca.explained_variance_ratio_[1]:.4f}")
print(f"   Total: {sum(pca.explained_variance_ratio_):.4f}")

# กราฟ PCA Scatter (แสดงกลุ่มข้อมูลบน 2 มิติ)
fig, ax = plt.subplots(figsize=(10, 8))
colors_map = {0: "#667eea", 1: "#00b894", 2: "#e17055"}
cluster_names = {0: "Cluster 0", 1: "Cluster 1", 2: "Cluster 2"}

for c in sorted(df_feat["cluster"].unique()):
    mask = df_feat["cluster"] == c
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors_map[c],
               label=cluster_names[c], s=60, alpha=0.7, edgecolors="white", linewidth=0.5)

# พล็อตจุดกลางของแต่ละกลุ่ม
centroids_pca = pca.transform(kmeans.cluster_centers_)
ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c="red", marker="X",
           s=200, edgecolors="black", linewidth=2, label="Centroids", zorder=5)

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
ax.set_title("K-Means Clusters (PCA Visualization)", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
path_pca = os.path.join(OUTPUT_DIR, "unsup_3_pca_clusters.png")
fig.savefig(path_pca, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved: {path_pca}")


# 7. การวิเคราะห์โปรไฟล์กลุ่มและบุคลิกลูกค้า (สร้างรูปแบบลูกค้า)
print("\n" + "=" * 60)
print(" 7. การวิเคราะห์โปรไฟล์กลุ่มและบุคลิกลูกค้า")
print("=" * 60)

cluster_profile = df_feat.groupby("cluster")[feature_cols].mean().round(2)
print(cluster_profile.T.to_string())

# 7.1 แผนภูมิ Radar สำหรับแต่ละกลุ่ม (ปัจจัยและบรรจุภัณฑ์)
radar_cols = factor_cols + pkg_cols
radar_labels = [
    "Natural", "Imported", "Taste", "Foreign", "Brand Fame",
    "Premium", "Cat Image", "Kibble", "Ingredient", "Eco",
    "Origin", "Benefit", "Guarantee"
]

fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(projection="polar"))
angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
angles += angles[:1]

for c in range(3):
    vals = cluster_profile.loc[c, radar_cols].values.tolist()
    vals += vals[:1]
    ax = axes[c]
    ax.fill(angles, vals, alpha=0.25, color=colors_map[c])
    ax.plot(angles, vals, "o-", linewidth=2, color=colors_map[c])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=7)
    ax.set_ylim(1, 5)
    ax.set_title(f"Cluster {c}", fontsize=12, fontweight="bold", pad=20)

fig.suptitle("Cluster Profiles — Factor & Packaging Attributes", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
path_radar = os.path.join(OUTPUT_DIR, "unsup_4_cluster_radar.png")
fig.savefig(path_radar, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved: {path_radar}")

# 7.2 คะแนนความต้องการซื้อออปชันตามกลุ่ม (แผนภูมิแท่งเรียง)
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(10)
width = 0.25

for c in range(3):
    vals = cluster_profile.loc[c, option_want_cols].values
    ax.bar(x + c * width, vals, width, label=f"Cluster {c}", color=colors_map[c], edgecolor="white")

ax.set_xlabel("Design Option", fontsize=12)
ax.set_ylabel("Avg Want-to-Buy Score", fontsize=12)
ax.set_title("Design Option Preference by Cluster", fontsize=14, fontweight="bold")
ax.set_xticks(x + width)
ax.set_xticklabels([f"Opt {i}" for i in range(1, 11)])
ax.set_ylim(1, 5)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
path_opt = os.path.join(OUTPUT_DIR, "unsup_5_option_by_cluster.png")
fig.savefig(path_opt, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved: {path_opt}")

# 7.3 ข้อมูลประชากรศาสตร์ตามกลุ่ม (อายุ เพศ สถานะการแต่งงาน)
df_clustered = df[df["cluster"] >= 0].copy()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, col in enumerate(["age", "gender", "marital_status"]):
    ct = pd.crosstab(df_clustered["cluster"], df_clustered[col], normalize="index") * 100
    ct.plot(kind="bar", stacked=True, ax=axes[idx], colormap="Set2", edgecolor="white")
    axes[idx].set_title(f"{col.replace('_', ' ').title()} by Cluster", fontweight="bold")
    axes[idx].set_ylabel("Percentage (%)")
    axes[idx].set_xlabel("Cluster")
    axes[idx].legend(fontsize=7, loc="upper right")
    axes[idx].tick_params(axis="x", rotation=0)

plt.tight_layout()
path_demo = os.path.join(OUTPUT_DIR, "unsup_6_demographics_cluster.png")
fig.savefig(path_demo, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved: {path_demo}")

# 7.4 การกระจายขนาดกลุ่ม (แผนภูมิวงกลม)
fig, ax = plt.subplots(figsize=(7, 7))
sizes = df_feat["cluster"].value_counts().sort_index()
labels_pie = [f"Cluster {i}\n({sizes[i]} persons)" for i in range(3)]
wedges, texts, autotexts = ax.pie(
    sizes.values, labels=labels_pie, autopct="%1.1f%%",
    colors=[colors_map[i] for i in range(3)],
    explode=(0.05, 0.05, 0.05), shadow=True, startangle=90,
    textprops={"fontsize": 11}
)
ax.set_title("Cluster Distribution", fontsize=14, fontweight="bold")
plt.tight_layout()
path_pie = os.path.join(OUTPUT_DIR, "unsup_7_cluster_distribution.png")
fig.savefig(path_pie, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved: {path_pie}")


# 8. สร้างบุคลิกลูกค้า (สร้างรูปแบบลูกค้าแต่ละกลุ่ม)
print("\n" + "=" * 60)
print(" 8. บุคลิกลูกค้า")
print("=" * 60)

personas = {}
for c in range(3):
    cluster_data = df_clustered[df_clustered["cluster"] == c]
    profile = cluster_profile.loc[c]

    # ปัจจัยที่สำคัญที่สุด
    factor_scores = profile[factor_cols].sort_values(ascending=False)
    top_factor = factor_scores.index[0].replace("factor_", "").replace("_", " ").title()

    # คุณลักษณะบรรจุภัณฑ์ที่สำคัญที่สุด
    pkg_scores = profile[pkg_cols].sort_values(ascending=False)
    top_pkg = pkg_scores.index[0].replace("pkg_", "").replace("_", " ").title()

    # ออปชันที่ต้องการซื้อมากที่สุด
    opt_scores = profile[option_want_cols].sort_values(ascending=False)
    top_opt = opt_scores.index[0]
    top_opt_num = top_opt.replace("opt", "").replace("_want_buy", "")

    # โหมดข้อมูลประชากรศาสตร์ (ค่าที่ปรากฏบ่อยที่สุด)
    top_age = cluster_data["age"].mode().iloc[0] if len(cluster_data["age"].mode()) > 0 else "N/A"
    top_gender = cluster_data["gender"].mode().iloc[0] if len(cluster_data["gender"].mode()) > 0 else "N/A"

    # ระดับคะแนนเฉลี่ยโดยรวม
    avg_all = profile[factor_cols + pkg_cols].mean()

    if avg_all >= 4.0:
        engagement = "High Engagement"
        desc = "กลุ่มผู้บริโภคที่ใส่ใจในคุณภาพของบรรจุภัณฑ์เป็นอย่างมาก และมักมองหาผลิตภัณฑ์พรีเมียม"
    elif avg_all >= 3.0:
        engagement = "Moderate Engagement"
        desc = "กลุ่มผู้ซื้อที่เน้นความสมดุล ซึ่งพิจารณาปัจจัยหลายด้านควบคู่กันไป และมีความคาดหวังในระดับปานกลาง"
    else:
        engagement = "Low Engagement"
        desc = "กลุ่มผู้บริโภคที่คำนึงถึงราคาเป็นหลัก และให้ความสำคัญกับบรรจุภัณฑ์น้อยที่สุด โดยเน้นที่การใช้งานจริงเป็นหลัก"

    personas[c] = {
        "name": f"Cluster {c} — {engagement}",
        "size": len(cluster_data),
        "pct": round(len(cluster_data) / len(df_clustered) * 100, 1),
        "engagement": engagement,
        "description": desc,
        "top_factor": top_factor,
        "top_pkg": top_pkg,
        "top_option": f"Option {top_opt_num}",
        "age": top_age,
        "gender": top_gender,
        "avg_factor": round(profile[factor_cols].mean(), 2),
        "avg_pkg": round(profile[pkg_cols].mean(), 2),
    }

    print(f"\n   --- {personas[c]['name']} ---")
    print(f"   Size: {personas[c]['size']} ({personas[c]['pct']}%)")
    print(f"   Description: {desc}")
    print(f"   Top Factor: {top_factor}, Top Packaging: {top_pkg}")
    print(f"   Preferred Design: Option {top_opt_num}")
    print(f"   Demographics: {top_age}, {top_gender}")


# 9. บันทึกผลลัพธ์ (บันทึกข้อมูลที่จัดกลุ่มแล้ว)
print("\n" + "=" * 60)
print(" 9. กำลังบันทึกผลลัพธ์")
print("=" * 60)

# บันทึกข้อมูลที่จัดกลุ่มแล้ว
df_clustered.to_csv(os.path.join(DATA_DIR, "CatFood_clustered.csv"), index=False, encoding="utf-8-sig")
print("   Saved: data/CatFood_clustered.csv")


# 10. การตรวจจับความผิดปกติ (หาข้อมูลที่ผิดปกติ)
print("\n" + "=" * 60)
print(" 10. การตรวจจับความผิดปกติ (Isolation Forest)")
print("=" * 60)

iso_forest = IsolationForest(n_estimators=100, contamination=0.08, random_state=42)
anomaly_labels = iso_forest.fit_predict(X_scaled)
anomaly_scores = iso_forest.decision_function(X_scaled)

n_anomalies = (anomaly_labels == -1).sum()
n_normal = (anomaly_labels == 1).sum()
print(f"   Normal: {n_normal}, Anomaly: {n_anomalies} ({n_anomalies/len(anomaly_labels)*100:.1f}%)")

# แสดงผลความผิดปกติบน PCA (วาด scatter และ histogram)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 10.1 กราฟ PCA Scatter แสดงความผิดปกติ
ax1 = axes[0]
normal_mask = anomaly_labels == 1
ax1.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1], c="#667eea", s=40, alpha=0.6, label=f"Normal ({n_normal})")
ax1.scatter(X_pca[~normal_mask, 0], X_pca[~normal_mask, 1], c="#ff6b6b", s=80, alpha=0.9, marker="X", edgecolors="black", linewidth=1, label=f"Anomaly ({n_anomalies})")
ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax1.set_title("Anomaly Detection (Isolation Forest)", fontweight="bold")
ax1.legend()
ax1.grid(alpha=0.3)

# 10.2 การกระจายคะแนนความผิดปกติ
ax2 = axes[1]
ax2.hist(anomaly_scores[normal_mask], bins=30, alpha=0.7, color="#667eea", label="Normal", edgecolor="white")
ax2.hist(anomaly_scores[~normal_mask], bins=15, alpha=0.7, color="#ff6b6b", label="Anomaly", edgecolor="white")
ax2.axvline(x=0, color="red", linestyle="--", alpha=0.5, label="Threshold")
ax2.set_xlabel("Anomaly Score")
ax2.set_ylabel("Frequency")
ax2.set_title("Anomaly Score Distribution", fontweight="bold")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
path_anomaly = os.path.join(OUTPUT_DIR, "unsup_8_anomaly_detection.png")
fig.savefig(path_anomaly, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved: {path_anomaly}")


# 11. บันทึกผลลัพธ์ทั้งหมด (บันทึกโมเดลและข้อมูล)
print("\n" + "=" * 60)
print(" 11. กำลังบันทึกผลลัพธ์ทั้งหมด")
print("=" * 60)

joblib.dump({
    "kmeans": kmeans,
    "scaler": scaler,
    "pca": pca,
    "feature_cols": feature_cols,
    "personas": personas,
    "cluster_profile": cluster_profile.to_dict(),
    "silhouette": sil,
    "anomaly_count": int(n_anomalies),
    "anomaly_pct": round(n_anomalies / len(anomaly_labels) * 100, 1),
}, os.path.join(MODELS_DIR, "unsup_model.pkl"))
print("   Saved: output/unsup_model.pkl")

print("\n" + "=" * 60)
print(" สรุปผลลัพธ์")
print("=" * 60)
print(f"   เทคนิค: K-Means + PCA + Isolation Forest")
print(f"   จำนวนกลุ่ม: 3, Silhouette: {sil:.4f}")
print(f"   PCA อธิบายความแปรปรวน: {sum(pca.explained_variance_ratio_)*100:.1f}%")
print(f"   ข้อมูลผิดปกติ: {n_anomalies} ({n_anomalies/len(anomaly_labels)*100:.1f}%)") 
print(f"   กราฟ: unsup_1 ~ unsup_8")
print("=" * 60)
print(" เสร็จสิ้น")
