"""
Analyze Synthetic Experiment Results
Replicates Tables in Section 7.2.1 and 7.2.2 of the paper.
Calculates Mean ± 95% Confidence Intervals.
"""

import pandas as pd
import numpy as np

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

if __name__ == "__main__":
    analyze_synthetic_results()