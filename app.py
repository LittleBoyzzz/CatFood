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
MODELS_DIR = os.path.join(BASE_DIR, "models")
CHARTS_DIR = os.path.join(BASE_DIR, "static", "charts")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ──────────────────────────────────────
# Page 1: Home + Descriptive Statistics
# ──────────────────────────────────────
@app.route("/")
def home():
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) as c FROM survey_responses").fetchone()["c"]
    best = conn.execute("SELECT * FROM model_results WHERE is_best=1").fetchone()
    personas = conn.execute("SELECT * FROM cluster_personas ORDER BY cluster_id").fetchall()
    age = conn.execute("SELECT age, COUNT(*) as c FROM survey_responses GROUP BY age ORDER BY c DESC").fetchall()
    gender = conn.execute("SELECT gender, COUNT(*) as c FROM survey_responses GROUP BY gender ORDER BY c DESC").fetchall()

    # Descriptive Statistics
    desc_factor = conn.execute("SELECT * FROM descriptive_stats WHERE category='factor' ORDER BY mean DESC").fetchall()
    desc_pkg = conn.execute("SELECT * FROM descriptive_stats WHERE category='packaging' ORDER BY mean DESC").fetchall()
    desc_opt = conn.execute("SELECT * FROM descriptive_stats WHERE category='option_want_buy' ORDER BY mean DESC").fetchall()

    conn.close()
    return render_template("home.html", total=total, best=dict(best), personas=[dict(p) for p in personas],
                           age=[dict(a) for a in age], gender=[dict(g) for g in gender],
                           desc_factor=[dict(d) for d in desc_factor],
                           desc_pkg=[dict(d) for d in desc_pkg],
                           desc_opt=[dict(d) for d in desc_opt])


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

    # Load unsup model for silhouette + anomaly info
    unsup_pkl_path = os.path.join(MODELS_DIR, "unsup_model.pkl")
    sil_score = 0.0
    anomaly_count = 0
    anomaly_pct = 0.0
    if os.path.exists(unsup_pkl_path):
        unsup = joblib.load(unsup_pkl_path)
        sil_score = unsup.get("silhouette", 0.0)
        anomaly_count = unsup.get("anomaly_count", 0)
        anomaly_pct = unsup.get("anomaly_pct", 0.0)

    conn.close()

    charts = [
        {"file": "unsupervised/unsup_1_correlation.png", "title": "Correlation Matrix", "desc": "ความสัมพันธ์ระหว่าง Features ทั้งหมด"},
        {"file": "unsupervised/unsup_2_elbow_silhouette.png", "title": "Elbow & Silhouette", "desc": "วิเคราะห์จำนวน Cluster ที่เหมาะสม"},
        {"file": "unsupervised/unsup_3_pca_clusters.png", "title": "PCA Cluster Visualization", "desc": "K-Means Clusters บน PCA 2 มิติ"},
        {"file": "unsupervised/unsup_4_cluster_radar.png", "title": "Cluster Radar Profiles", "desc": "โปรไฟล์ปัจจัยและบรรจุภัณฑ์แต่ละ Cluster"},
        {"file": "unsupervised/unsup_5_option_by_cluster.png", "title": "Option Preference by Cluster", "desc": "คะแนนความต้องการซื้อแต่ละดีไซน์ตาม Cluster"},
        {"file": "unsupervised/unsup_6_demographics_cluster.png", "title": "Demographics by Cluster", "desc": "อายุ เพศ สถานภาพ แต่ละ Cluster"},
        {"file": "unsupervised/unsup_7_cluster_distribution.png", "title": "Cluster Distribution", "desc": "สัดส่วนสมาชิกแต่ละ Cluster"},
        {"file": "unsupervised/unsup_8_anomaly_detection.png", "title": "Anomaly Detection", "desc": "ตรวจจับข้อมูลผิดปกติด้วย Isolation Forest"},
    ]

    return render_template("unsupervised.html", personas=personas, cluster_sizes=cluster_sizes,
                           charts=charts, sil_score=round(sil_score, 4),
                           anomaly_count=anomaly_count, anomaly_pct=anomaly_pct)


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
        {"file": "supervised/1_model_comparison.png", "title": "เปรียบเทียบประสิทธิภาพโมเดล"},
        {"file": "supervised/2_confusion_matrices.png", "title": "Confusion Matrices"},
        {"file": "supervised/3_roc_curves.png", "title": "ROC Curves"},
        {"file": "supervised/4_feature_importance.png", "title": "Feature Importance (Top 15)"},
        {"file": "supervised/5_decision_tree.png", "title": "Decision Tree"},
        {"file": "supervised/6_cv_boxplot.png", "title": "Cross-Validation Box Plot"},
        {"file": "supervised/7_classification_report_heatmap.png", "title": "Classification Report"},
        {"file": "supervised/8_target_distribution.png", "title": "Target Distribution"},
    ]

    return render_template("supervised.html", models=models, best=best, total=total,
                           target_dist=target_dist, factor_avg=dict(factor_avg),
                           pkg_avg=dict(pkg_avg), charts=charts)


# ──────────────────────────────────────
# Page 4: Business Insight (from real DB)
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

    # Option averages (want_buy) from DB
    opt_query = ", ".join([f"ROUND(AVG(opt{i}_want_buy),2) as opt{i}" for i in range(1, 11)])
    opt_avg = conn.execute(f"SELECT {opt_query} FROM survey_responses").fetchone()

    # Find best option from DB data
    opt_dict = dict(opt_avg)
    best_option = max(opt_dict, key=opt_dict.get)
    best_option_num = best_option.replace("opt", "")
    best_option_score = opt_dict[best_option]

    # Cluster profile from DB (real data)
    cluster_profiles = {}
    try:
        profile_rows = conn.execute("SELECT * FROM cluster_profile ORDER BY cluster_id").fetchall()
        for row in profile_rows:
            r = dict(row)
            cid = r["cluster_id"]
            if cid not in cluster_profiles:
                cluster_profiles[cid] = {}
            cluster_profiles[cid][r["feature"]] = r["avg_score"]
    except:
        pass

    # Packaging influence percentage
    total = conn.execute("SELECT COUNT(*) as c FROM survey_responses").fetchone()["c"]
    influenced = conn.execute("SELECT COUNT(*) as c FROM survey_responses WHERE packaging_influence=1").fetchone()["c"]
    influence_pct = round(influenced / total * 100, 1) if total > 0 else 0

    conn.close()

    # Option 3 analysis charts
    opt3_charts = [
        {"file": "option/opt3_1_want_buy_comparison.png", "title": "คะแนนความอยากซื้อ — เปรียบเทียบทุก Option",
         "desc": "Option 3 มีคะแนน want-to-buy สูงที่สุดจากทุก Option"},
        {"file": "option/opt3_2_radar_comparison.png", "title": "Radar — เปรียบเทียบ Top 4 Options (5 มิติ)",
         "desc": "เปรียบเทียบ 5 มิติ: อยากซื้อ, โดดเด่น, พรีเมียม, รสชาติ, เหมาะกับฉัน"},
        {"file": "option/opt3_3_cluster_comparison.png", "title": "คะแนนแต่ละ Option ตาม Cluster",
         "desc": "ทุก Cluster ให้คะแนน Option 3 สูงกว่าตัวเลือกอื่น"},
        {"file": "option/opt3_4_boxplot_comparison.png", "title": "Box Plot — การกระจายคะแนน Top 3 Options",
         "desc": "เปรียบเทียบการกระจายคะแนนในทุกมิติ"},
    ]

    return render_template("business.html", personas=personas, best=best,
                           pkg_avg=dict(pkg_avg), opt_avg=dict(opt_avg), total=total,
                           best_option_num=best_option_num, best_option_score=best_option_score,
                           influence_pct=influence_pct, cluster_profiles=cluster_profiles,
                           opt3_charts=opt3_charts)


@app.route("/output/<path:filename>")
def serve_chart(filename):
    return send_from_directory(CHARTS_DIR, filename)


# ──────────────────────────────────────
# API: Predict (Supervised)
# ──────────────────────────────────────
_cached_model_bundle = None

def load_model_bundle():
    global _cached_model_bundle
    if _cached_model_bundle is None:
        pkl_path = os.path.join(MODELS_DIR, "best_model.pkl")
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
        pkl_path = os.path.join(MODELS_DIR, "option_model.pkl")
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
        proba_arr = model.predict_proba(X_input)[0]
        classes   = [int(c) for c in model.classes_]

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
