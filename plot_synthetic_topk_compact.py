"""
Plot synthetic experiment top-k metrics for skill-matched distribution only.
2x2 grid: Top-1 Accuracy, Top-3 Precision, Top-5 Precision, Top-5 KTD.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_topk_metrics(filename="synthetic_results_merged.csv"):
    print(f"Loading {filename}...")
    df = pd.read_csv(filename)
    print(f"Loaded {len(df)} trials.")

    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        plt.style.use('ggplot')

    metrics = ["top1", "top3p", "top5p", "top5_ktd"]
    metric_labels = {
        "top1": "Top-1 Accuracy\n(higher is better)",
        "top3p": "Top-3 Precision\n(higher is better)",
        "top5p": "Top-5 Precision\n(higher is better)",
        "top5_ktd": "Top-5 KTD\n(lower is better)",
    }

    contest_counts = sorted(df['contests'].unique())

    WEIGHT_KEYS = ["uniform", "logarithmic", "vigna", "quadratic"]
    COLORS = {
        "uniform":     "#377eb8",
        "logarithmic": "#984ea3",
        "vigna":       "#e41a1c",
        "quadratic":   "#ff7f00",
    }
    LABELS = {
        "uniform":     "Uniform",
        "logarithmic": "Logarithmic",
        "vigna":       "Hyperbolic",
        "quadratic":   "Quadratic",
    }
    MARKERS = {
        "uniform":     "o",
        "logarithmic": "s",
        "vigna":       "^",
        "quadratic":   "D",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for idx, metric in enumerate(metrics):
        row, col = divmod(idx, 2)
        ax = axes[row, col]

        for w in WEIGHT_KEYS:
            means = []
            cis = []
            for n in contest_counts:
                subset = df[
                    (df['dist'] == "skill_matched") &
                    (df['contests'] == n) &
                    (df['weight'] == w)
                ]
                mean = subset[metric].mean()
                ci = 1.96 * (subset[metric].std() / np.sqrt(len(subset)))
                means.append(mean)
                cis.append(ci)

            means = np.array(means)
            cis = np.array(cis)

            ax.errorbar(contest_counts, means, yerr=cis,
                        label=LABELS[w], color=COLORS[w],
                        marker=MARKERS[w], markersize=6,
                        linewidth=2, capsize=4, alpha=0.9)

        ax.set_xlabel("Number of Contests", fontsize=11)
        ax.set_ylabel(metric_labels[metric], fontsize=11)
        ax.set_xticks(contest_counts)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if row == 0 and col == 0:
            ax.legend(loc="lower right", frameon=True, framealpha=1.0, fontsize=9)

    fig.suptitle("Skill-Matched Distribution", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig("synthetic_topk_plot_compact.pdf", format='pdf', bbox_inches='tight', dpi=300)
    print("Saved to synthetic_topk_plot_compact.pdf")


if __name__ == "__main__":
    plot_topk_metrics()
