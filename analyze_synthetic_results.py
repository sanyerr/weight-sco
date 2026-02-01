"""
Analyze Synthetic Experiment Results
Replicates Tables in Section 7.2.1 and 7.2.2 of the paper.
Calculates Mean ± 95% Confidence Intervals.
Includes statistical significance testing for top-k metrics.
"""

import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations

def analyze_synthetic_results(filename="synthetic_results_merged.csv"):
    print(f"Loading {filename}...")
    
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: Could not find {filename}. Run synthetic_experiment_merged.py first.")
        return

    print(f"Loaded {len(df)} trials.")

    # ---------------------------------------------------------
    # HELPER: Compute Mean ± CI String
    # ---------------------------------------------------------
    def format_stat(series):
        if len(series) < 2:
            return f"{series.mean():.2f}"
        
        mean = series.mean()
        # 95% CI = 1.96 * (std / sqrt(n))
        ci = 1.96 * (series.std() / np.sqrt(len(series)))
        return f"{mean:.2f} ± {ci:.2f}"

    # ---------------------------------------------------------
    # TABLE 1: GLOBAL METRICS (KTD, MTRD)
    # ---------------------------------------------------------
    print("\n" + "="*80)
    print("GLOBAL METRICS (Replicates Section 7.2.1)")
    print("="*80)
    print(f"{'Dist':<10} | {'Contests':<8} | {'Weight':<10} | {'KTD':<18} | {'MTRD':<18}")
    print("-" * 80)

    # Grouping order matches paper: Dist -> Contests -> Weight
    distributions = ["uniform", "skill_matched"]
    contest_counts = sorted(df['contests'].unique())
    weights = ["uniform", "vigna", "quadratic"]

    for dist in distributions:
        print(f"--- {dist.upper()} DISTRIBUTION ---")
        for n in contest_counts:
            for w in weights:
                # Filter data
                subset = df[
                    (df['dist'] == dist) & 
                    (df['contests'] == n) & 
                    (df['weight'] == w)
                ]
                
                if len(subset) == 0:
                    continue

                ktd_str = format_stat(subset['ktd'])
                mtrd_str = format_stat(subset['mtrd'])
                
                print(f"{dist:<10} | {n:<8} | {w:<10} | {ktd_str:<18} | {mtrd_str:<18}")
        print("")


    # ---------------------------------------------------------
    # TABLE 2: TOP-K METRICS
    # ---------------------------------------------------------
    print("\n" + "="*115)
    print("TOP-K METRICS (Replicates Section 7.2.2)")
    print("="*115)
    # Added Top-3 Prec column
    print(f"{'Dist':<10} | {'Cnt':<5} | {'Weight':<10} | {'Top-1 Acc':<15} | {'Top-3 Prec':<15} | {'Top-5 Prec':<15} | {'Top-5 KTD':<15}")
    print("-" * 115)

    for dist in distributions:
        print(f"--- {dist.upper()} DISTRIBUTION ---")
        for n in contest_counts:
            for w in weights:
                subset = df[
                    (df['dist'] == dist) & 
                    (df['contests'] == n) & 
                    (df['weight'] == w)
                ]
                
                if len(subset) == 0:
                    continue

                # Top-1 is 0/1 accuracy, others are floats
                top1_str = format_stat(subset['top1'])
                top3p_str = format_stat(subset['top3p']) # New
                top5p_str = format_stat(subset['top5p'])
                top5k_str = format_stat(subset['top5_ktd'])
                
                print(f"{dist:<10} | {n:<5} | {w:<10} | {top1_str:<15} | {top3p_str:<15} | {top5p_str:<15} | {top5k_str:<15}")
        print("")

def analyze_statistical_significance(filename="synthetic_results_merged.csv"):
    """
    Perform statistical significance testing for top-k metrics.
    Uses paired Wilcoxon signed-rank tests (non-parametric) with Bonferroni correction.
    """
    print(f"\nLoading {filename}...")

    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: Could not find {filename}. Run synthetic_experiment_merged.py first.")
        return

    print(f"Loaded {len(df)} trials.")

    weights = ["uniform", "vigna", "quadratic"]
    weight_pairs = list(combinations(weights, 2))
    metrics = ["top1", "top3p", "top5p", "top5_ktd"]
    metric_labels = {
        "top1": "Top-1 Accuracy",
        "top3p": "Top-3 Precision",
        "top5p": "Top-5 Precision",
        "top5_ktd": "Top-5 KTD"
    }

    distributions = ["uniform", "skill_matched"]
    contest_counts = sorted(df['contests'].unique())

    # Bonferroni correction: 3 pairwise comparisons per metric
    n_comparisons = len(weight_pairs)
    alpha = 0.05
    bonferroni_alpha = alpha / n_comparisons

    print("\n" + "=" * 100)
    print("STATISTICAL SIGNIFICANCE TESTING FOR TOP-K METRICS")
    print("=" * 100)
    print(f"Test: Wilcoxon signed-rank (paired, non-parametric)")
    print(f"Alpha: {alpha}, Bonferroni-corrected alpha: {bonferroni_alpha:.4f} (for {n_comparisons} comparisons)")
    print("Higher values are better for precision metrics; lower is better for Top-5 KTD.")
    print("=" * 100)

    for dist in distributions:
        print(f"\n{'=' * 100}")
        print(f"DISTRIBUTION: {dist.upper()}")
        print("=" * 100)

        for n_contests in contest_counts:
            print(f"\n--- {n_contests} Contests ---")

            for metric in metrics:
                print(f"\n  {metric_labels[metric]}:")

                # Get data for each weight
                weight_data = {}
                for w in weights:
                    subset = df[
                        (df['dist'] == dist) &
                        (df['contests'] == n_contests) &
                        (df['weight'] == w)
                    ].sort_values('seed')
                    weight_data[w] = subset[metric].values

                # Check that we have paired data
                seeds_match = True
                for w in weights:
                    subset = df[
                        (df['dist'] == dist) &
                        (df['contests'] == n_contests) &
                        (df['weight'] == w)
                    ].sort_values('seed')
                    if len(subset) == 0:
                        seeds_match = False
                        break

                if not seeds_match:
                    print("    Insufficient data for comparison")
                    continue

                # Print means for reference
                for w in weights:
                    mean_val = weight_data[w].mean()
                    print(f"    {w}: mean = {mean_val:.4f}")

                # Pairwise comparisons
                print(f"    Pairwise comparisons:")
                for w1, w2 in weight_pairs:
                    data1 = weight_data[w1]
                    data2 = weight_data[w2]

                    # Check if arrays are identical (no variation)
                    diff = data1 - data2
                    if np.all(diff == 0):
                        print(f"      {w1} vs {w2}: No difference (identical values)")
                        continue

                    # Wilcoxon signed-rank test
                    try:
                        _, p_value = stats.wilcoxon(data1, data2, alternative='two-sided')

                        # Determine significance
                        if p_value < bonferroni_alpha:
                            sig = "***"
                        elif p_value < alpha:
                            sig = "*"
                        else:
                            sig = ""

                        # Direction of effect
                        mean_diff = data1.mean() - data2.mean()
                        if metric == "top5_ktd":
                            # Lower is better for KTD
                            better = w1 if mean_diff < 0 else w2
                        else:
                            # Higher is better for precision
                            better = w1 if mean_diff > 0 else w2

                        print(f"      {w1} vs {w2}: p = {p_value:.4f} {sig}")
                        if p_value < alpha:
                            print(f"        → {better} significantly better")
                    except ValueError as e:
                        print(f"      {w1} vs {w2}: Could not compute ({e})")

    print("\n" + "=" * 100)
    print("Legend: *** p < Bonferroni-corrected alpha, * p < 0.05")
    print("=" * 100)


def analyze_significance_summary(filename="synthetic_results_merged.csv"):
    """
    Generate a compact summary table of statistical significance results.
    """
    print(f"\nLoading {filename}...")

    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: Could not find {filename}.")
        return

    weights = ["uniform", "vigna", "quadratic"]
    weight_pairs = list(combinations(weights, 2))
    metrics = ["top1", "top3p", "top5p", "top5_ktd"]

    distributions = ["uniform", "skill_matched"]
    contest_counts = sorted(df['contests'].unique())

    n_comparisons = len(weight_pairs)
    alpha = 0.05
    bonferroni_alpha = alpha / n_comparisons

    print("\n" + "=" * 120)
    print("SUMMARY: STATISTICAL SIGNIFICANCE OF TOP-K METRICS")
    print("=" * 120)
    print(f"{'Dist':<14} | {'Contests':<8} | {'Comparison':<20} | {'Top-1':<12} | {'Top-3p':<12} | {'Top-5p':<12} | {'Top-5 KTD':<12}")
    print("-" * 120)

    for dist in distributions:
        for n_contests in contest_counts:
            for w1, w2 in weight_pairs:
                row = f"{dist:<14} | {n_contests:<8} | {w1} vs {w2:<10} |"

                for metric in metrics:
                    # Get paired data
                    data1 = df[(df['dist'] == dist) & (df['contests'] == n_contests) & (df['weight'] == w1)].sort_values('seed')[metric].values
                    data2 = df[(df['dist'] == dist) & (df['contests'] == n_contests) & (df['weight'] == w2)].sort_values('seed')[metric].values

                    diff = data1 - data2
                    if np.all(diff == 0):
                        cell = "="
                    else:
                        try:
                            _, p_value = stats.wilcoxon(data1, data2, alternative='two-sided')
                            mean_diff = data1.mean() - data2.mean()

                            if metric == "top5_ktd":
                                better = w1 if mean_diff < 0 else w2
                            else:
                                better = w1 if mean_diff > 0 else w2

                            if p_value < bonferroni_alpha:
                                cell = f"{better[:3]}***"
                            elif p_value < alpha:
                                cell = f"{better[:3]}*"
                            else:
                                cell = "n.s."
                        except ValueError:
                            cell = "N/A"

                    row += f" {cell:<12} |"

                print(row)
        print("-" * 120)

    print("\nLegend: *** p < Bonferroni-corrected, * p < 0.05, n.s. = not significant, = = identical")
    print("        Winner shown (uni=uniform, vig=vigna, qua=quadratic)")


if __name__ == "__main__":
    analyze_synthetic_results()
    print("\n\n")
    analyze_statistical_significance()
    print("\n\n")
    analyze_significance_summary()