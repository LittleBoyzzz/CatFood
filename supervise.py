"""
Supervised Learning — Cat Food Packaging Survey
=================================================
Target : packaging_influence (1=มีผล, 0=ไม่มีผล)
         → ทำนายว่าบรรจุภัณฑ์มีผลต่อการตัดสินใจซื้อหรือไม่

Features: ปัจจัยที่ส่งผลต่อการซื้อ, คุณสมบัติบรรจุภัณฑ์,
          คะแนนดีไซน์ Option 1-10, ข้อมูลประชากรศาสตร์

Models : DecisionTree, RandomForest, LogisticRegression
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import joblib
import os
import warnings

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    ConfusionMatrixDisplay, RocCurveDisplay
)

warnings.filterwarnings("ignore")

# ============================================================
# ตั้งค่า Font สำหรับภาษาไทย (fallback ถ้าไม่มี)
# ============================================================
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.unicode_minus"] = False

# พยายามใช้ font ไทย ถ้ามี
try:
    matplotlib.rcParams["font.sans-serif"] = ["Tahoma", "Angsana New", "Leelawadee UI", "Arial"]
except:
    pass

# สร้างโฟลเดอร์เก็บผลลัพธ์
OUTPUT_DIR = r"d:\Work\Final\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. โหลดข้อมูลที่ทำความสะอาดแล้ว
# ============================================================
print("=" * 65)
print(" 1. Loading cleaned data")
print("=" * 65)

df = pd.read_csv(r"d:\Work\Final\data\CatFood_cleaned.csv")
print(f"   Data shape: {df.shape}")

# ============================================================
# 2. กำหนด Target & Features
# ============================================================
print("\n" + "=" * 65)
print(" 2. Defining Target & Features")
print("=" * 65)

TARGET = "packaging_influence"
print(f"   Target: {TARGET}")
print(f"   Target distribution:\n{df[TARGET].value_counts().to_string()}")

# --- เลือก Feature columns ---
# 2.1 ปัจจัยที่ส่งผลต่อการซื้อ (Likert 1-5)
factor_cols = [
    "factor_natural", "factor_imported", "factor_taste",
    "factor_foreign", "factor_brand_fame",
]

# 2.2 คุณสมบัติบรรจุภัณฑ์ที่ส่งผลต่อการซื้อ (Likert 1-5)
pkg_cols = [
    "pkg_premium", "pkg_cat_image", "pkg_kibble_image",
    "pkg_ingredient_image", "pkg_eco_friendly",
    "pkg_origin_symbol", "pkg_benefit_symbol", "pkg_guarantee",
]

# 2.3 คะแนนดีไซน์ Option 1-10 (Likert 1-5)
option_cols = [c for c in df.columns if c.startswith("opt")]

# 2.4 ข้อมูลประชากรศาสตร์ (Categorical → encode)
demo_cols = ["age", "gender", "marital_status"]

# รวม Feature ทั้งหมด
feature_cols = factor_cols + pkg_cols + option_cols + demo_cols
print(f"   Total features: {len(feature_cols)}")

# ============================================================
# 3. เตรียมข้อมูล (Encoding & Splitting)
# ============================================================
print("\n" + "=" * 65)
print(" 3. Preparing data")
print("=" * 65)

df_model = df[feature_cols + [TARGET]].dropna().copy()
print(f"   Rows after dropna: {len(df_model)}")

# Label Encode categorical columns
label_encoders = {}
for col in demo_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    label_encoders[col] = le
    print(f"   Encoded '{col}': {list(le.classes_)}")

# Split features and target
X = df_model[factor_cols + pkg_cols + option_cols + demo_cols]
y = df_model[TARGET].astype(int)

print(f"\n   X shape: {X.shape}")
print(f"   y distribution: 1={y.sum()}, 0={len(y)-y.sum()}")

# Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# Scale สำหรับ Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 4. Train 3 Models
# ============================================================
print("\n" + "=" * 65)
print(" 4. Training Models")
print("=" * 65)

models = {
    "DecisionTree": DecisionTreeClassifier(
        max_depth=5, random_state=42, class_weight="balanced"
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42, class_weight="balanced"
    ),
    "LogisticRegression": LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced"
    ),
}

results = {}

for name, model in models.items():
    print(f"\n   --- {name} ---")

    # ใช้ scaled data สำหรับ LogisticRegression
    if name == "LogisticRegression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5, scoring="accuracy")
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cv_mean = cv_scores.mean()

    results[name] = {
        "model": model,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "cv_mean": cv_mean,
        "cv_std": cv_scores.std(),
    }

    print(f"   Accuracy:    {acc:.4f}")
    print(f"   Precision:   {prec:.4f}")
    print(f"   Recall:      {rec:.4f}")
    print(f"   F1-Score:    {f1:.4f}")
    print(f"   CV Accuracy: {cv_mean:.4f} (+/- {cv_scores.std():.4f})")

# ============================================================
# 5. เปรียบเทียบและเลือก Best Model
# ============================================================
print("\n" + "=" * 65)
print(" 5. Model Comparison & Best Model Selection")
print("=" * 65)

comparison_df = pd.DataFrame({
    name: {
        "Accuracy": r["accuracy"],
        "Precision": r["precision"],
        "Recall": r["recall"],
        "F1-Score": r["f1"],
        "CV Accuracy": r["cv_mean"],
    }
    for name, r in results.items()
}).T

print(comparison_df.round(4).to_string())

# เลือกโมเดลที่ F1 สูงที่สุด
best_name = max(results, key=lambda k: results[k]["f1"])
best_model = results[best_name]["model"]
print(f"\n   >>> Best Model: {best_name} (F1={results[best_name]['f1']:.4f})")

# บันทึก Best Model เป็น .pkl
pkl_path = os.path.join(OUTPUT_DIR, "best_model.pkl")
joblib.dump({
    "model": best_model,
    "scaler": scaler if best_name == "LogisticRegression" else None,
    "label_encoders": label_encoders,
    "feature_names": list(X.columns),
    "model_name": best_name,
}, pkl_path)
print(f"   Saved: {pkl_path}")

# ============================================================
# 6. Visualizations — บันทึกเป็น PNG ทั้งหมด
# ============================================================
print("\n" + "=" * 65)
print(" 6. Creating Visualizations")
print("=" * 65)

# ──────────────────────────────────────────────
# 6.1 Model Performance Comparison (Bar Chart)
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "CV Accuracy"]
x = np.arange(len(metrics))
width = 0.25
colors = ["#3498db", "#2ecc71", "#e74c3c"]

for i, (name, r) in enumerate(results.items()):
    vals = [r["accuracy"], r["precision"], r["recall"], r["f1"], r["cv_mean"]]
    bars = ax.bar(x + i * width, vals, width, label=name, color=colors[i], edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_xlabel("Metrics", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
ax.set_xticks(x + width)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.15)
ax.legend(loc="upper right")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
path1 = os.path.join(OUTPUT_DIR, "1_model_comparison.png")
fig.savefig(path1, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved: {path1}")

# ──────────────────────────────────────────────
# 6.2 Confusion Matrices (3 models side-by-side)
# ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for idx, (name, r) in enumerate(results.items()):
    cm = confusion_matrix(y_test, r["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx],
                xticklabels=["No Effect (0)", "Effect (1)"],
                yticklabels=["No Effect (0)", "Effect (1)"])
    axes[idx].set_title(f"{name}", fontsize=12, fontweight="bold")
    axes[idx].set_ylabel("Actual")
    axes[idx].set_xlabel("Predicted")

fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
path2 = os.path.join(OUTPUT_DIR, "2_confusion_matrices.png")
fig.savefig(path2, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved: {path2}")

# ──────────────────────────────────────────────
# 6.3 ROC Curves (all 3 models on one plot)
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for name, r, color in zip(results.keys(), results.values(), colors):
    fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={roc_auc:.3f})")

ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC=0.500)")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves", fontsize=14, fontweight="bold")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
path3 = os.path.join(OUTPUT_DIR, "3_roc_curves.png")
fig.savefig(path3, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved: {path3}")

# ──────────────────────────────────────────────
# 6.4 Feature Importance (Top 15 from RandomForest)
# ──────────────────────────────────────────────
rf_model = results["RandomForest"]["model"]
importances = rf_model.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False).head(15)

fig, ax = plt.subplots(figsize=(10, 7))
feat_imp_sorted = feat_imp.sort_values(ascending=True)
bars = ax.barh(range(len(feat_imp_sorted)), feat_imp_sorted.values, color="#3498db", edgecolor="white")
ax.set_yticks(range(len(feat_imp_sorted)))
ax.set_yticklabels(feat_imp_sorted.index, fontsize=10)
ax.set_xlabel("Importance", fontsize=12)
ax.set_title("Top 15 Feature Importances (RandomForest)", fontsize=14, fontweight="bold")

for bar, val in zip(bars, feat_imp_sorted.values):
    ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9)

ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
path4 = os.path.join(OUTPUT_DIR, "4_feature_importance.png")
fig.savefig(path4, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved: {path4}")

# ──────────────────────────────────────────────
# 6.5 Decision Tree Visualization
# ──────────────────────────────────────────────
dt_model = results["DecisionTree"]["model"]
fig, ax = plt.subplots(figsize=(24, 10))
plot_tree(
    dt_model,
    feature_names=list(X.columns),
    class_names=["No Effect", "Effect"],
    filled=True,
    rounded=True,
    fontsize=8,
    ax=ax,
    max_depth=3,
)
ax.set_title("Decision Tree (max_depth=3 shown)", fontsize=16, fontweight="bold")
plt.tight_layout()
path5 = os.path.join(OUTPUT_DIR, "5_decision_tree.png")
fig.savefig(path5, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved: {path5}")

# ──────────────────────────────────────────────
# 6.6 Cross-Validation Scores Comparison (Box Plot)
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
cv_data = {}
for name, model_obj in models.items():
    if name == "LogisticRegression":
        scores = cross_val_score(model_obj, scaler.transform(X), y, cv=5, scoring="accuracy")
    else:
        scores = cross_val_score(model_obj, X, y, cv=5, scoring="accuracy")
    cv_data[name] = scores

bp = ax.boxplot(cv_data.values(), labels=cv_data.keys(), patch_artist=True)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Cross-Validation Accuracy (5-Fold)", fontsize=14, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

for i, (name, scores) in enumerate(cv_data.items()):
    ax.text(i + 1, scores.mean() + 0.005, f"Mean: {scores.mean():.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
path6 = os.path.join(OUTPUT_DIR, "6_cv_boxplot.png")
fig.savefig(path6, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved: {path6}")

# ──────────────────────────────────────────────
# 6.7 Classification Report Heatmap (Best Model)
# ──────────────────────────────────────────────
best_r = results[best_name]
report = classification_report(y_test, best_r["y_pred"], output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).iloc[:3, :2].T  # precision, recall, f1 for classes 0 and 1
report_df.index = ["No Effect (0)", "Effect (1)"]

fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(report_df.astype(float), annot=True, fmt=".3f", cmap="YlGnBu",
            ax=ax, vmin=0, vmax=1, linewidths=0.5)
ax.set_title(f"Classification Report — {best_name} (Best Model)",
             fontsize=14, fontweight="bold")
ax.set_ylabel("Class")
plt.tight_layout()
path7 = os.path.join(OUTPUT_DIR, "7_classification_report_heatmap.png")
fig.savefig(path7, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved: {path7}")

# ──────────────────────────────────────────────
# 6.8 Target Distribution (Pie Chart)
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))
target_counts = y.value_counts()
labels = ["Effect (1)", "No Effect (0)"]
explode = (0.05, 0.05)
ax.pie(target_counts.values, labels=labels, autopct="%1.1f%%",
       colors=["#2ecc71", "#e74c3c"], explode=explode,
       shadow=True, startangle=90, textprops={"fontsize": 12})
ax.set_title("Target Distribution — Packaging Influence",
             fontsize=14, fontweight="bold")
plt.tight_layout()
path8 = os.path.join(OUTPUT_DIR, "8_target_distribution.png")
fig.savefig(path8, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Saved: {path8}")

# ============================================================
# 7. สรุปผลลัพธ์
# ============================================================
print("\n" + "=" * 65)
print(" SUMMARY")
print("=" * 65)
print(f"\n   Target:       {TARGET}")
print(f"   Features:     {len(X.columns)} columns")
print(f"   Train/Test:   {X_train.shape[0]} / {X_test.shape[0]}")
print(f"\n   Model Performance:")
print(comparison_df.round(4).to_string())
print(f"\n   Best Model:  {best_name}")
print(f"   Best F1:     {results[best_name]['f1']:.4f}")
print(f"   Saved .pkl:  {pkl_path}")
print(f"\n   Visualizations saved to: {OUTPUT_DIR}")
print(f"   - 1_model_comparison.png")
print(f"   - 2_confusion_matrices.png")
print(f"   - 3_roc_curves.png")
print(f"   - 4_feature_importance.png")
print(f"   - 5_decision_tree.png")
print(f"   - 6_cv_boxplot.png")
print(f"   - 7_classification_report_heatmap.png")
print(f"   - 8_target_distribution.png")
print("\n" + "=" * 65)
print(" DONE!")
print("=" * 65)
