"""
Flask Web Application — Cat Food AI Dashboard
================================================
4 Pages: Home, Unsupervised, Supervised, Business Insight
Database: SQLite (catfood.db)
"""

from flask import Flask, render_template, jsonify, send_from_directory, request
import joblib
import numpy as np
import sqlite3
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "catfood.db")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ──────────────────────────────────────
# Page 1: Home
# ──────────────────────────────────────
@app.route("/")
def home():
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) as c FROM survey_responses").fetchone()["c"]
    best = conn.execute("SELECT * FROM model_results WHERE is_best=1").fetchone()
    personas = conn.execute("SELECT * FROM cluster_personas ORDER BY cluster_id").fetchall()
    age = conn.execute("SELECT age, COUNT(*) as c FROM survey_responses GROUP BY age ORDER BY c DESC").fetchall()
    gender = conn.execute("SELECT gender, COUNT(*) as c FROM survey_responses GROUP BY gender ORDER BY c DESC").fetchall()
    conn.close()
    return render_template("home.html", total=total, best=dict(best), personas=[dict(p) for p in personas],
                           age=[dict(a) for a in age], gender=[dict(g) for g in gender])


# ──────────────────────────────────────
# Page 2: Unsupervised Learning
# ──────────────────────────────────────
@app.route("/unsupervised")
def unsupervised():
    conn = get_db()
    personas = conn.execute("SELECT * FROM cluster_personas ORDER BY cluster_id").fetchall()
    personas = [dict(p) for p in personas]

    cluster_sizes = conn.execute(
        "SELECT cluster, COUNT(*) as c FROM clustered_data WHERE cluster>=0 GROUP BY cluster ORDER BY cluster"
    ).fetchall()
    cluster_sizes = [dict(c) for c in cluster_sizes]

    conn.close()

    charts = [
        {"file": "unsup_1_correlation.png", "title": "Correlation Matrix", "desc": "Correlation between all features"},
        {"file": "unsup_2_elbow_silhouette.png", "title": "Elbow & Silhouette", "desc": "Optimal number of clusters analysis"},
        {"file": "unsup_3_pca_clusters.png", "title": "PCA Cluster Visualization", "desc": "K-Means clusters projected onto 2D PCA space"},
        {"file": "unsup_4_cluster_radar.png", "title": "Cluster Radar Profiles", "desc": "Factor & packaging attributes per cluster"},
        {"file": "unsup_5_option_by_cluster.png", "title": "Option Preference by Cluster", "desc": "Design option want-to-buy scores"},
        {"file": "unsup_6_demographics_cluster.png", "title": "Demographics by Cluster", "desc": "Age, gender, marital status distribution"},
        {"file": "unsup_7_cluster_distribution.png", "title": "Cluster Distribution", "desc": "Proportion of each cluster"},
    ]

    return render_template("unsupervised.html", personas=personas, cluster_sizes=cluster_sizes, charts=charts)


# ──────────────────────────────────────
# Page 3: Supervised Learning
# ──────────────────────────────────────
@app.route("/supervised")
def supervised():
    conn = get_db()
    models = conn.execute("SELECT * FROM model_results ORDER BY f1_score DESC").fetchall()
    models = [dict(m) for m in models]
    best = conn.execute("SELECT * FROM model_results WHERE is_best=1").fetchone()
    best = dict(best)
    total = conn.execute("SELECT COUNT(*) as c FROM survey_responses").fetchone()["c"]

    target_dist = conn.execute(
        "SELECT packaging_influence, COUNT(*) as c FROM survey_responses GROUP BY packaging_influence"
    ).fetchall()
    target_dist = {str(r["packaging_influence"]): r["c"] for r in target_dist}

    factor_avg = conn.execute("""
        SELECT ROUND(AVG(factor_natural),2) as factor_natural,
               ROUND(AVG(factor_imported),2) as factor_imported,
               ROUND(AVG(factor_taste),2) as factor_taste,
               ROUND(AVG(factor_foreign),2) as factor_foreign,
               ROUND(AVG(factor_brand_fame),2) as factor_brand_fame
        FROM survey_responses
    """).fetchone()

    pkg_avg = conn.execute("""
        SELECT ROUND(AVG(pkg_premium),2) as pkg_premium,
               ROUND(AVG(pkg_cat_image),2) as pkg_cat_image,
               ROUND(AVG(pkg_kibble_image),2) as pkg_kibble_image,
               ROUND(AVG(pkg_ingredient_image),2) as pkg_ingredient_image,
               ROUND(AVG(pkg_eco_friendly),2) as pkg_eco_friendly,
               ROUND(AVG(pkg_origin_symbol),2) as pkg_origin_symbol,
               ROUND(AVG(pkg_benefit_symbol),2) as pkg_benefit_symbol,
               ROUND(AVG(pkg_guarantee),2) as pkg_guarantee
        FROM survey_responses
    """).fetchone()

    conn.close()

    charts = [
        {"file": "1_model_comparison.png", "title": "Model Performance Comparison"},
        {"file": "2_confusion_matrices.png", "title": "Confusion Matrices"},
        {"file": "3_roc_curves.png", "title": "ROC Curves"},
        {"file": "4_feature_importance.png", "title": "Feature Importance (Top 15)"},
        {"file": "5_decision_tree.png", "title": "Decision Tree"},
        {"file": "6_cv_boxplot.png", "title": "Cross-Validation Box Plot"},
        {"file": "7_classification_report_heatmap.png", "title": "Classification Report"},
        {"file": "8_target_distribution.png", "title": "Target Distribution"},
    ]

    return render_template("supervised.html", models=models, best=best, total=total,
                           target_dist=target_dist, factor_avg=dict(factor_avg),
                           pkg_avg=dict(pkg_avg), charts=charts)


# ──────────────────────────────────────
# Page 4: Business Insight
# ──────────────────────────────────────
@app.route("/business")
def business():
    conn = get_db()
    personas = conn.execute("SELECT * FROM cluster_personas ORDER BY cluster_id").fetchall()
    personas = [dict(p) for p in personas]
    best = conn.execute("SELECT * FROM model_results WHERE is_best=1").fetchone()
    best = dict(best)

    # Top packaging attributes
    pkg_avg = conn.execute("""
        SELECT ROUND(AVG(pkg_premium),2) as pkg_premium,
               ROUND(AVG(pkg_cat_image),2) as pkg_cat_image,
               ROUND(AVG(pkg_kibble_image),2) as pkg_kibble_image,
               ROUND(AVG(pkg_ingredient_image),2) as pkg_ingredient_image,
               ROUND(AVG(pkg_eco_friendly),2) as pkg_eco_friendly,
               ROUND(AVG(pkg_origin_symbol),2) as pkg_origin_symbol,
               ROUND(AVG(pkg_benefit_symbol),2) as pkg_benefit_symbol,
               ROUND(AVG(pkg_guarantee),2) as pkg_guarantee
        FROM survey_responses
    """).fetchone()

    # Option averages
    opt_query = ", ".join([f"ROUND(AVG(opt{i}_want_buy),2) as opt{i}" for i in range(1, 11)])
    opt_avg = conn.execute(f"SELECT {opt_query} FROM survey_responses").fetchone()

    total = conn.execute("SELECT COUNT(*) as c FROM survey_responses").fetchone()["c"]
    conn.close()

    return render_template("business.html", personas=personas, best=best,
                           pkg_avg=dict(pkg_avg), opt_avg=dict(opt_avg), total=total)


@app.route("/output/<path:filename>")
def serve_chart(filename):
    return send_from_directory(OUTPUT_DIR, filename)


# ──────────────────────────────────────
# API: Predict (Supervised)
# ──────────────────────────────────────
_cached_model_bundle = None

def load_model_bundle():
    global _cached_model_bundle
    if _cached_model_bundle is None:
        pkl_path = os.path.join(OUTPUT_DIR, "best_model.pkl")
        _cached_model_bundle = joblib.load(pkl_path)
    return _cached_model_bundle


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        bundle = load_model_bundle()
        model        = bundle["model"]
        scaler       = bundle["scaler"]
        label_enc    = bundle["label_encoders"]
        feature_names = bundle["feature_names"]
        model_name   = bundle["model_name"]

        # Build feature vector in correct order
        row = []
        for col in feature_names:
            val = data.get(col, 3)  # default to middle of Likert scale
            # Encode categorical features
            if col in label_enc:
                le = label_enc[col]
                val_str = str(val)
                if val_str in le.classes_:
                    val = int(le.transform([val_str])[0])
                else:
                    val = 0
            row.append(float(val))

        X_input = np.array([row])

        if scaler is not None:
            X_input = scaler.transform(X_input)

        prediction   = int(model.predict(X_input)[0])
        probability  = float(model.predict_proba(X_input)[0][prediction])
        proba_effect = float(model.predict_proba(X_input)[0][1])

        return jsonify({
            "success": True,
            "model_name": model_name,
            "prediction": prediction,
            "label": "มีผลต่อการซื้อ" if prediction == 1 else "ไม่มีผลต่อการซื้อ",
            "probability": round(probability * 100, 1),
            "proba_effect": round(proba_effect * 100, 1),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ──────────────────────────────────────
# API: Predict Favorite Design Option
# ──────────────────────────────────────
_cached_option_bundle = None

def load_option_bundle():
    global _cached_option_bundle
    if _cached_option_bundle is None:
        pkl_path = os.path.join(OUTPUT_DIR, "option_model.pkl")
        _cached_option_bundle = joblib.load(pkl_path)
    return _cached_option_bundle


# Design option descriptions (th)
OPTION_DESC = {
    1:  "ดีไซน์คลาสสิก สีธรรมชาติ โทนอบอุ่น",
    2:  "ดีไซน์โมเดิร์น สีสดใส กราฟิกชัดเจน",
    3:  "ดีไซน์มินิมอล สะอาดตา เน้นฉลากข้อมูล",
    4:  "ดีไซน์พรีเมียม สีทอง-ดำ ดูหรูหรา",
    5:  "ดีไซน์อีโค่ สีเขียว วัสดุรักษ์โลก",
    6:  "ดีไซน์ภาพแมวน่ารัก พาสเทล เน้นอารมณ์",
    7:  "ดีไซน์เน้นส่วนผสม ภาพวัตถุดิบชัดเจน",
    8:  "ดีไซน์ฉลากนำเข้า มีสัญลักษณ์ต่างประเทศ",
    9:  "ดีไซน์สุขภาพ เน้นโภชนาการและประโยชน์",
    10: "ดีไซน์แบรนด์แนม โลโก้ใหญ่ เน้นชื่อเสียง",
}


def _encode_option_input(data, bundle):
    feature_names = bundle["feature_names"]
    label_enc = bundle["label_encoders"]
    row = []
    for col in feature_names:
        val = data.get(col, 3)
        if col in label_enc:
            le = label_enc[col]
            val_str = str(val)
            val = int(le.transform([val_str])[0]) if val_str in le.classes_ else 0
        row.append(float(val))
    return np.array([row])


@app.route("/predict_favorite_option", methods=["POST"])
def predict_favorite_option():
    try:
        data = request.get_json(force=True)
        bundle = load_option_bundle()
        model  = bundle["model"]

        X_input = _encode_option_input(data, bundle)
        proba_arr = model.predict_proba(X_input)[0]   # shape (10,)
        classes   = [int(c) for c in model.classes_]

        # Build top-3 ranking
        proba_dict = {cls: float(p) for cls, p in zip(classes, proba_arr)}
        top3 = sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)[:3]

        ranking = []
        for rank, (opt, prob) in enumerate(top3, 1):
            ranking.append({
                "rank":   rank,
                "option": opt,
                "probability": round(prob * 100, 1),
                "description": OPTION_DESC.get(opt, f"Option {opt}"),
            })

        winner = ranking[0]["option"]

        return jsonify({
            "success": True,
            "winner": winner,
            "winner_desc": OPTION_DESC.get(winner, f"Option {winner}"),
            "winner_prob": ranking[0]["probability"],
            "ranking": ranking,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    print(f"\n  Cat Food AI Dashboard")
    print(f"  http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
