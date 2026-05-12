"""
Option 3 Analysis — Why Option 3 is the Best Design
=====================================================
สร้างกราฟวิเคราะห์เปรียบเทียบว่าทำไม Option 3 ถึงเป็นดีไซน์ที่ดีที่สุด
- กราฟ 1: เปรียบเทียบ want_buy ทุก Option (Bar Chart)
- กราฟ 2: Radar Chart เปรียบเทียบ Option 3 กับ Option อื่น (5 มิติ)
- กราฟ 3: Option want_buy ตาม Cluster
- กราฟ 4: Box plot คะแนน 5 มิติ ของ Option 3 vs Top 3 อื่น
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Tahoma', 'Leelawadee UI', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
import os
import sys
import warnings

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "charts", "option")
DATA_DIR = os.path.join(BASE_DIR, "data")

# ============================================================
# 1. Load Data
# ============================================================
print("=" * 60)
print(" Option 3 Analysis — Why It's the Best")
print("=" * 60)

df = pd.read_csv(os.path.join(DATA_DIR, "CatFood_cleaned.csv"))
# Load clustered if available
clustered_path = os.path.join(DATA_DIR, "CatFood_clustered.csv")
if os.path.exists(clustered_path):
    df_cluster = pd.read_csv(clustered_path)
else:
    df_cluster = None

aspects = ["want_buy", "standout", "premium", "taste", "suit_me"]
aspect_labels = ["อยากซื้อ", "โดดเด่น", "พรีเมียม", "รสชาติ", "เหมาะกับฉัน"]

# ============================================================
# 2. Chart 1: Want-to-Buy Score Comparison (All Options)
# ============================================================
print("\n  Chart 1: Want-to-Buy comparison...")

want_cols = [f"opt{i}_want_buy" for i in range(1, 11)]
avg_want = df[want_cols].mean()
opt_labels = [f"Option {i}" for i in range(1, 11)]

fig, ax = plt.subplots(figsize=(12, 6))
colors = ["#667eea"] * 10
best_idx = avg_want.values.argmax()
colors[best_idx] = "#00b894"  # Highlight best (Option 3)

bars = ax.bar(opt_labels, avg_want.values, color=colors, edgecolor="white", linewidth=1.5)
for i, (bar, val) in enumerate(zip(bars, avg_want.values)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
            f"{val:.2f}", ha="center", va="bottom", fontsize=10,
            fontweight="bold" if i == best_idx else "normal",
            color="#00b894" if i == best_idx else "#666")

ax.set_ylabel("Average Want-to-Buy Score (1-5)", fontsize=12)
ax.set_title("Design Option Comparison — Want-to-Buy Score", fontsize=14, fontweight="bold")
ax.set_ylim(1, 5)
ax.grid(axis="y", alpha=0.3)
ax.axhline(y=avg_want.values[best_idx], color="#00b894", linestyle="--", alpha=0.4, label=f"Best: Option {best_idx+1}")
ax.legend(fontsize=10)
plt.tight_layout()
path1 = os.path.join(OUTPUT_DIR, "opt3_1_want_buy_comparison.png")
fig.savefig(path1, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path1}")

# ============================================================
# 3. Chart 2: Radar — Option 3 vs Top 3 Others (5 dimensions)
# ============================================================
print("  Chart 2: Radar comparison...")

# Calculate average scores for each option across 5 dimensions
opt_scores = {}
for i in range(1, 11):
    scores = []
    for asp in aspects:
        col = f"opt{i}_{asp}"
        scores.append(df[col].mean())
    opt_scores[i] = scores

# Sort by want_buy to find top 4
sorted_opts = sorted(opt_scores.items(), key=lambda x: x[1][0], reverse=True)
top4 = sorted_opts[:4]  # Top 4 options

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))
angles = np.linspace(0, 2 * np.pi, len(aspects), endpoint=False).tolist()
angles += angles[:1]

radar_colors = ["#00b894", "#667eea", "#e17055", "#fdcb6e"]
for idx, (opt_num, scores) in enumerate(top4):
    vals = scores + scores[:1]
    ax.fill(angles, vals, alpha=0.15, color=radar_colors[idx])
    ax.plot(angles, vals, "o-", linewidth=2.5 if opt_num == best_idx+1 else 1.5,
            color=radar_colors[idx], markersize=6, label=f"Option {opt_num}")

ax.set_xticks(angles[:-1])
ax.set_xticklabels(aspect_labels, fontsize=10)
ax.set_ylim(1, 5)
ax.set_title("Top 4 Options — 5 Dimension Comparison", fontsize=13, fontweight="bold", pad=25)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
plt.tight_layout()
path2 = os.path.join(OUTPUT_DIR, "opt3_2_radar_comparison.png")
fig.savefig(path2, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path2}")

# ============================================================
# 4. Chart 3: Option Want-Buy by Cluster
# ============================================================
print("  Chart 3: Options by Cluster...")

if df_cluster is not None and "cluster" in df_cluster.columns:
    df_c = df_cluster[df_cluster["cluster"] >= 0].copy()
    n_clusters = df_c["cluster"].nunique()

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(10)
    width = 0.8 / n_clusters
    cluster_colors = ["#667eea", "#00b894", "#e17055", "#fdcb6e"]

    for c in range(n_clusters):
        mask = df_c["cluster"] == c
        vals = [df_c.loc[mask, f"opt{i}_want_buy"].mean() for i in range(1, 11)]
        offset = (c - n_clusters/2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=f"Cluster {c}",
                      color=cluster_colors[c % len(cluster_colors)], edgecolor="white")

    ax.set_xlabel("Design Option", fontsize=12)
    ax.set_ylabel("Avg Want-to-Buy Score", fontsize=12)
    ax.set_title("Want-to-Buy Score by Cluster — All Options", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Opt {i}" for i in range(1, 11)])
    ax.set_ylim(1, 5)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Highlight Option 3
    ax.axvspan(1.5, 2.5, alpha=0.08, color="#00b894")
    ax.text(2, 4.85, "Option 3", ha="center", fontsize=9, color="#00b894", fontweight="bold")

    plt.tight_layout()
    path3 = os.path.join(OUTPUT_DIR, "opt3_3_cluster_comparison.png")
    fig.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path3}")
else:
    print("  Skipped (no cluster data)")

# ============================================================
# 5. Chart 4: Box Plot — Option 3 vs Others (5 dimensions)
# ============================================================
print("  Chart 4: Box plot comparison...")

top3_opts = [t[0] for t in top4[:3]]  # Top 3 options
fig, axes = plt.subplots(1, 5, figsize=(20, 5))

for asp_idx, (asp, asp_lbl) in enumerate(zip(aspects, aspect_labels)):
    ax = axes[asp_idx]
    data_list = []
    labels_list = []
    for opt_num in top3_opts:
        col = f"opt{opt_num}_{asp}"
        data_list.append(df[col].dropna().values)
        labels_list.append(f"Opt {opt_num}")

    bp = ax.boxplot(data_list, labels=labels_list, patch_artist=True)
    box_colors = ["#00b894", "#667eea", "#e17055"]
    for patch, color in zip(bp["boxes"], box_colors[:len(top3_opts)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_title(asp_lbl, fontsize=11, fontweight="bold")
    ax.set_ylim(0.5, 5.5)
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("Top 3 Options — Score Distribution (5 Dimensions)", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
path4 = os.path.join(OUTPUT_DIR, "opt3_4_boxplot_comparison.png")
fig.savefig(path4, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path4}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print(" SUMMARY")
print("=" * 60)
best_opt = best_idx + 1
print(f"  Best Option (highest want_buy): Option {best_opt}")
print(f"  Want-Buy Score: {avg_want.values[best_idx]:.3f}")
print(f"  All scores: {dict(zip([f'Opt{i}' for i in range(1,11)], avg_want.values.round(3)))}")
print(f"\n  Charts saved:")
print(f"  - opt3_1_want_buy_comparison.png")
print(f"  - opt3_2_radar_comparison.png")
print(f"  - opt3_3_cluster_comparison.png")
print(f"  - opt3_4_boxplot_comparison.png")
print("=" * 60)
print(" DONE!")
print("=" * 60)
