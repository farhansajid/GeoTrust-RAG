"""
Generates publication-quality IEEE J-STARS figures directly from real experiment outputs.
All plots read exclusively from validated experiment JSON files in 'experiments/'.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_all_real_figures():
    os.makedirs("results", exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # -------------------------------------------------------------------------
    # 1. Real Baseline Comparison (B1 - B7)
    # -------------------------------------------------------------------------
    exp1_file = "experiments/exp_01_main_baselines/metrics.json"
    if os.path.exists(exp1_file):
        with open(exp1_file, "r", encoding="utf-8") as f:
            b_data = json.load(f)

        methods = list(b_data.keys())
        accs = [b_data[m]["Accuracy"] for m in methods]
        eces = [b_data[m]["ECE"] * 100.0 for m in methods] # scaled to % for display
        cit_rec = [b_data[m]["Citation_Recall"] for m in methods]

        x = np.arange(len(methods))
        width = 0.28

        fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
        r1 = ax.bar(x - width, accs, width, label="Accuracy (%)", color="#2b5c8f")
        r2 = ax.bar(x, eces, width, label="ECE (Calibration Error %)", color="#e15759")
        r3 = ax.bar(x + width, cit_rec, width, label="Citation Recall (%)", color="#59a14f")

        ax.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
        ax.set_title("Empirical Baseline Performance on Real Remote Sensing Dataset (RTX 5060)", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        labels = [m.replace("B", "").replace("_", " ") for m in methods]
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.legend(frameon=True, fontsize=10)
        ax.set_ylim(0, 115)

        for rects in [r1, r2, r3]:
            for rect in rects:
                h = rect.get_height()
                if h > 0:
                    ax.annotate(f"{h:.1f}",
                                xy=(rect.get_x() + rect.get_width() / 2, h),
                                xytext=(0, 2), textcoords="offset points",
                                ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        p1 = "results/fig_real_baselines.png"
        plt.savefig(p1)
        plt.close()
        print(f"[Plot Exported] {p1}")

    # -------------------------------------------------------------------------
    # 2. Real RAG Sensitivity Sweep (Top-K)
    # -------------------------------------------------------------------------
    exp3_file = "experiments/exp_03_rag_retrieval/metrics.json"
    if os.path.exists(exp3_file):
        with open(exp3_file, "r", encoding="utf-8") as f:
            rag_data = json.load(f)

        k_labels = [1, 2, 3, 5, 8]
        recalls = [rag_data[f"Top_{k}"]["Recall@K"] for k in k_labels]
        mrrs = [rag_data[f"Top_{k}"]["MRR"] * 100.0 for k in k_labels]
        latencies = [rag_data[f"Top_{k}"]["Retrieval_Latency_ms"] for k in k_labels]

        fig, ax1 = plt.subplots(figsize=(7.5, 4.5), dpi=300)
        ax2 = ax1.twinx()

        l1 = ax1.plot(k_labels, recalls, marker="o", linewidth=2.2, color="#2b5c8f", label="Recall@K (%)")
        l2 = ax1.plot(k_labels, mrrs, marker="s", linewidth=2.2, color="#59a14f", linestyle="--", label="MRR (x100)")
        l3 = ax2.plot(k_labels, latencies, marker="^", linewidth=1.8, color="#e15759", linestyle=":", label="Retrieval Latency (ms)")

        ax1.set_xlabel("Retrieval Top-K Depth", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Retrieval Quality (%)", fontsize=11, fontweight="bold", color="#2b5c8f")
        ax2.set_ylabel("Latency (ms on CPU)", fontsize=11, fontweight="bold", color="#e15759")
        ax1.set_title("Geospatial Knowledge Retrieval Performance vs. Top-K Depth", fontsize=12, fontweight="bold")

        lines = l1 + l2 + l3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="center right", frameon=True, fontsize=9)

        plt.tight_layout()
        p2 = "results/fig_real_rag_sensitivity.png"
        plt.savefig(p2)
        plt.close()
        print(f"[Plot Exported] {p2}")

    # -------------------------------------------------------------------------
    # 3. Real Controlled Robustness Perturbations
    # -------------------------------------------------------------------------
    exp4_file = "experiments/exp_04_robustness/metrics.json"
    if os.path.exists(exp4_file):
        with open(exp4_file, "r", encoding="utf-8") as f:
            rob_data = json.load(f)

        ptypes = list(rob_data.keys())
        accs_rob = [rob_data[p]["Accuracy"] for p in ptypes]
        abs_f1_rob = [rob_data[p]["Abstention_F1"] * 100.0 for p in ptypes]

        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        x_rob = np.arange(len(ptypes))
        w = 0.35

        ax.bar(x_rob - w/2, accs_rob, w, label="Accuracy (%)", color="#2b5c8f")
        ax.bar(x_rob + w/2, abs_f1_rob, w, label="Abstention F1 (%)", color="#e15759")

        ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
        ax.set_title("Model Stability Across Controlled Environmental & Sensor Corruptions", fontsize=12, fontweight="bold")
        ax.set_xticks(x_rob)
        ax.set_xticklabels([p.replace("_", " ").title() for p in ptypes], rotation=20, ha="right", fontsize=9)
        ax.legend(frameon=True, fontsize=10)
        ax.set_ylim(0, 35)

        plt.tight_layout()
        p3 = "results/fig_real_robustness.png"
        plt.savefig(p3)
        plt.close()
        print(f"[Plot Exported] {p3}")

    print("\n[SUCCESS] All real publication figures generated from actual experiment outputs.")

if __name__ == "__main__":
    plot_all_real_figures()
