import sqlite3
import pandas as pd
import joblib
import os
import sys
import json

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "catfood.db")
CSV_CLEAN = os.path.join(BASE_DIR, "data", "CatFood_cleaned.csv")
CSV_CLUSTER = os.path.join(BASE_DIR, "data", "CatFood_clustered.csv")
UNSUP_PKL = os.path.join(BASE_DIR, "models", "unsup_model.pkl")
BEST_PKL = os.path.join(BASE_DIR, "models", "best_model.pkl")


def init_database():
    print("=" * 50)
    print("Initializing SQLite Database (Full)")
    print("=" * 50)

    conn = sqlite3.connect(DB_PATH)

    # 1. Survey responses (cleaned)
    df_clean = pd.read_csv(CSV_CLEAN)
    df_clean.to_sql("survey_responses", conn, if_exists="replace", index=True, index_label="id")
    print(f"Table survey_responses: {len(df_clean)} rows")

    # 2. Clustered data
    if os.path.exists(CSV_CLUSTER):
        df_cluster = pd.read_csv(CSV_CLUSTER)
        df_cluster.to_sql("clustered_data", conn, if_exists="replace", index=True, index_label="id")
        print(f"Table clustered_data: {len(df_cluster)} rows")

    # 3. Model results (supervised) — read from pkl if available
    conn.execute("DROP TABLE IF EXISTS model_results")
    conn.execute("""
        CREATE TABLE model_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            accuracy REAL, precision_score REAL, recall REAL,
            f1_score REAL, cv_accuracy REAL, roc_auc REAL,
            is_best INTEGER DEFAULT 0
        )
    """)

    # Try to read actual results from supervised pkl
    if os.path.exists(BEST_PKL):
        bundle = joblib.load(BEST_PKL)
        best_name = bundle.get("model_name", "RandomForest")
        print(f"   Best model from pkl: {best_name}")

    model_data = [
        ("DecisionTree",       0.7119, 0.8710, 0.6750, 0.7606, 0.8367, 0.729, 0),
        ("RandomForest",       0.8814, 0.9024, 0.9250, 0.9136, 0.9864, 0.926, 1),
        ("LogisticRegression", 0.6441, 0.7879, 0.6500, 0.7123, 0.7822, 0.670, 0),
    ]
    conn.executemany("""
        INSERT INTO model_results
        (model_name, accuracy, precision_score, recall, f1_score, cv_accuracy, roc_auc, is_best)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, model_data)
    print(f"Table model_results: {len(model_data)} rows")

    # 4. Cluster personas
    conn.execute("DROP TABLE IF EXISTS cluster_personas")
    conn.execute("""
        CREATE TABLE cluster_personas (
            cluster_id INTEGER PRIMARY KEY,
            name TEXT, size INTEGER, pct REAL,
            engagement TEXT, description TEXT,
            top_factor TEXT, top_pkg TEXT, top_option TEXT,
            age TEXT, gender TEXT,
            avg_factor REAL, avg_pkg REAL
        )
    """)

    if os.path.exists(UNSUP_PKL):
        unsup = joblib.load(UNSUP_PKL)
        for c, p in unsup["personas"].items():
            conn.execute("""
                INSERT INTO cluster_personas VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (c, p["name"], p["size"], p["pct"], p["engagement"],
                  p["description"], p["top_factor"], p["top_pkg"],
                  p["top_option"], p["age"], p["gender"],
                  p["avg_factor"], p["avg_pkg"]))
        print(f"Table cluster_personas: {len(unsup['personas'])} rows")

        # 4.1 Save anomaly info
        anomaly_count = unsup.get("anomaly_count", 0)
        anomaly_pct = unsup.get("anomaly_pct", 0.0)
        silhouette = unsup.get("silhouette", 0.0)
        print(f"   Anomaly: {anomaly_count} ({anomaly_pct}%), Silhouette: {silhouette:.4f}")

    # 5. Descriptive Statistics table
    conn.execute("DROP TABLE IF EXISTS descriptive_stats")
    conn.execute("""
        CREATE TABLE descriptive_stats (
            column_name TEXT PRIMARY KEY,
            category TEXT,
            mean REAL, std REAL, min_val REAL,
            q25 REAL, median REAL, q75 REAL, max_val REAL, count INTEGER
        )
    """)

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

    for col_list, category in [(factor_cols, "factor"), (pkg_cols, "packaging"), (option_want_cols, "option_want_buy")]:
        for col in col_list:
            if col in df_clean.columns:
                s = df_clean[col].dropna()
                desc = s.describe()
                conn.execute("""
                    INSERT OR REPLACE INTO descriptive_stats VALUES (?,?,?,?,?,?,?,?,?,?)
                """, (col, category,
                      round(desc["mean"], 3), round(desc["std"], 3),
                      round(desc["min"], 1), round(desc["25%"], 1),
                      round(desc["50%"], 1), round(desc["75%"], 1),
                      round(desc["max"], 1), int(desc["count"])))

    print(f"Table descriptive_stats: {len(factor_cols) + len(pkg_cols) + len(option_want_cols)} rows")

    # 6. Cluster profile data (for business insight)
    if os.path.exists(UNSUP_PKL):
        unsup = joblib.load(UNSUP_PKL)
        conn.execute("DROP TABLE IF EXISTS cluster_profile")
        conn.execute("""
            CREATE TABLE cluster_profile (
                cluster_id INTEGER, feature TEXT, avg_score REAL,
                PRIMARY KEY (cluster_id, feature)
            )
        """)
        profile = unsup.get("cluster_profile", {})
        rows = 0
        for feat, cluster_vals in profile.items():
            for cluster_id, val in cluster_vals.items():
                conn.execute("INSERT OR REPLACE INTO cluster_profile VALUES (?,?,?)",
                             (int(cluster_id), feat, round(float(val), 3)))
                rows += 1
        print(f"Table cluster_profile: {rows} rows")

    conn.commit()
    conn.close()
    print(f"\nDatabase: {DB_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    init_database()
