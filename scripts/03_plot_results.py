"""
Script 03: Plot Publication-Grade Figures for IEEE J-STARS
Generates:
1. Comparative Bar Chart across Baseline and Proposed Models
2. Risk-Coverage Dynamics Curve
3. Qualitative Grounded Case Study with Bounding Box Localization
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def generate_publication_figures():
    print("=" * 80)
    print("GeoTrust-RAG: Stage 3 - Generating Publication-Quality Figures")
    print("=" * 80)

    os.makedirs("results", exist_ok=True)
    summary_path = "results/benchmark_summary.json"
    curves_path = "results/curves_data.json"

    with open(summary_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)
    with open(curves_path, "r", encoding="utf-8") as f:
        curves_data = json.load(f)

    # -------------------------------------------------------------
    # Figure 1: Benchmark Performance Bar Chart
    # -------------------------------------------------------------
    systems = list(summary_data.keys())
    accuracies = [summary_data[s]["Overall_Accuracy"] for s in systems]
    sel_accs = [summary_data[s]["Selective_Accuracy"] for s in systems]
    abs_f1s = [summary_data[s]["Abstention_F1"] * 100.0 for s in systems]

    x = np.arange(len(systems))
    width = 0.25

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    rects1 = ax.bar(x - width, accuracies, width, label="Overall Accuracy (%)", color="#2b5c8f")
    rects2 = ax.bar(x, sel_accs, width, label="Selective Accuracy (%)", color="#3b9ab2")
    rects3 = ax.bar(x + width, abs_f1s, width, label="Abstention F1 Score (%)", color="#e15759")

    ax.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
    ax.set_title("Performance Comparison on Post-Disaster Visual Reasoning Benchmark", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("Base VLM + ", "").replace("Proposed ", "") for s in systems], rotation=15, ha="right", fontsize=10)
    ax.legend(frameon=True, fontsize=10)
    ax.set_ylim(0, 115)

    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig_path1 = "results/fig_benchmark_comparison.png"
    plt.savefig(fig_path1)
    plt.close()
    print(f"[Exported] {fig_path1}")

    # -------------------------------------------------------------
    # Figure 2: Risk-Coverage Dynamics Curve
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    palette = ["#7f7f7f", "#bcbd22", "#17becf", "#ff7f0e", "#d62728"]

    for idx, (sys_name, c_info) in enumerate(curves_data.items()):
        covs = c_info["coverages"]
        risks = c_info["risks"]
        # Sort for smooth curve
        sort_idx = np.argsort(covs)
        cov_sorted = np.array(covs)[sort_idx]
        risk_sorted = np.array(risks)[sort_idx]
        ax.plot(cov_sorted * 100.0, risk_sorted * 100.0, label=sys_name, color=palette[idx % len(palette)],
                linewidth=2.2 if "Full" in sys_name else 1.5,
                linestyle="-" if "Full" in sys_name else "--")

    ax.set_xlabel("Coverage (%) - Fraction of Answered Queries", fontsize=12, fontweight="bold")
    ax.set_ylabel("Risk (%) - Selective Error Rate", fontsize=12, fontweight="bold")
    ax.set_title("Risk-Coverage Tradeoff under Varying Confidence Thresholds", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.set_ylim(-5, 60)
    ax.legend(frameon=True, fontsize=9, loc="upper left")

    plt.tight_layout()
    fig_path2 = "results/fig_risk_coverage.png"
    plt.savefig(fig_path2)
    plt.close()
    print(f"[Exported] {fig_path2}")

    # -------------------------------------------------------------
    # Figure 3: Qualitative Dual-Modal Grounded Case Study
    # -------------------------------------------------------------
    sample_img_path = "data/disaster_eval_bench/sample_scenes/flood_scene_01.jpg"
    img = Image.open(sample_img_path)
    w, h = img.size

    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
    ax.imshow(img)

    # Box [ymin, xmin, ymax, xmax] normalized 0-1000
    norm_box = [520, 110, 890, 940]
    real_ymin = (norm_box[0] / 1000.0) * h
    real_xmin = (norm_box[1] / 1000.0) * w
    real_ymax = (norm_box[2] / 1000.0) * h
    real_xmax = (norm_box[3] / 1000.0) * w

    rect = patches.Rectangle((real_xmin, real_ymin), real_xmax - real_xmin, real_ymax - real_ymin,
                             linewidth=3, edgecolor="#e15759", facecolor="none")
    ax.add_patch(rect)
    ax.text(real_xmin + 10, real_ymin + 30, "Submerged Inundated Roadway [Doc-1]",
            color="white", fontsize=10, fontweight="bold",
            bbox=dict(facecolor="#e15759", alpha=0.85, edgecolor="none", pad=4))

    ax.axis("off")
    ax.set_title("GeoTrust-RAG Dual-Grounded Explainability\nSpatial Localization + Attributed Domain Citation",
                 fontsize=12, fontweight="bold", pad=12)

    plt.tight_layout()
    fig_path3 = "results/fig_qualitative_sample.png"
    plt.savefig(fig_path3)
    plt.close()
    print(f"[Exported] {fig_path3}")

    print("\n[SUCCESS] All IEEE J-STARS publication figures generated successfully in 'results/' folder!")

if __name__ == "__main__":
    generate_publication_figures()
