import pandas as pd
import numpy as np
import joblib
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Fix encoding for Windows terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "CatFood_cleaned.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  Favorite Option Model Training")
print("=" * 60)

# ── 1. Load data ──────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"\n  Loaded: {df.shape[0]} rows × {df.shape[1]} cols")

# ── 2. Build target: option with highest want_buy score ───────
opt_cols = [c for c in df.columns if c.startswith("opt") and "want_buy" in c]
print(f"  Option columns found: {opt_cols}")

# argmax → option number (1-indexed)
df["fav_option"] = df[opt_cols].apply(
    lambda row: int(row.idxmax().replace("opt", "").replace("_want_buy", "")),
    axis=1
)
print(f"\n  Target distribution (fav_option):")
print(df["fav_option"].value_counts().sort_index().to_string())

# ── 3. Features ───────────────────────────────────────────────
factor_cols = [
    "factor_natural", "factor_imported", "factor_taste",
    "factor_foreign", "factor_brand_fame",
]
pkg_cols = [
    "pkg_premium", "pkg_cat_image", "pkg_kibble_image",
    "pkg_ingredient_image", "pkg_eco_friendly",
    "pkg_origin_symbol", "pkg_benefit_symbol", "pkg_guarantee",
]
demo_cols = ["age", "gender", "marital_status"]
feature_cols = factor_cols + pkg_cols + demo_cols

df_model = df[feature_cols + ["fav_option"]].dropna().copy()
print(f"\n  Rows after dropna: {len(df_model)}")

# ── 4. Encode categoricals ────────────────────────────────────
label_encoders = {}
for col in demo_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    label_encoders[col] = le
    print(f"  Encoded '{col}': {[str(c.encode('utf-8')) for c in le.classes_]}")

X = df_model[feature_cols]
y = df_model["fav_option"].astype(int)

# ── 5. Train/Test split ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n  Train: {len(X_train)}, Test: {len(X_test)}")

# ── 6. Train RandomForest ─────────────────────────────────────
print("\n  Training RandomForest (multi-class)...")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=2,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
print(f"  Accuracy  : {acc:.4f}")
print(f"  F1 (weighted): {f1:.4f}")

# Class names
classes = sorted(y.unique())
print(f"\n  Classes: {classes}")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# ── 7. Save model bundle ──────────────────────────────────────
bundle = {
    "model": rf,
    "label_encoders": label_encoders,
    "feature_names": list(X.columns),
    "classes": classes,
    "accuracy": acc,
    "f1": f1,
}
save_path = os.path.join(OUTPUT_DIR, "option_model.pkl")
joblib.dump(bundle, save_path)
print(f"\n  Saved → {save_path}")
print("\n" + "=" * 60)
print("  DONE!")
print("=" * 60)
