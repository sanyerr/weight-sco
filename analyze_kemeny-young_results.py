"""
Analyze Kemeny-Young Replication Results
Replicates the table in Section 7.1 of the paper.
Comparing: Standard (Uniform), Weighted (Vigna), and Quadratic SCO.
"""

import pandas as pd
import numpy as np

def analyze_replication_results(filename="replication_results_multi.csv"):
    print(f"Loading {filename}...")
    
    try:
        # Load data, treating "N/A" as NaN
        df = pd.read_csv(filename, na_values=["N/A", "None"])
    except FileNotFoundError:
        print(f"Error: Could not find {filename}. Run run_batch.py first.")
        return

    print(f"Loaded {len(df)} instances with computed Kemeny-Optimal rankings.")

    # ---------------------------------------------------------
    # CONFIGURATION
    # ---------------------------------------------------------
    variants = [
        ("Standard SCO", "Uni"),
        ("Weighted SCO", "Wei"),
        ("Quadratic SCO", "Quad")
    ]
    
    metrics = [
        ("KT to Kemeny (mean)", "{}_KT", "mean"),
        ("Perfect Match (%)",   "{}_KT", "perfect"), # Derived metric (KT == 0)
        ("Top-1 Accuracy (%)",  "{}_Top1", "mean"),
        ("Top-3 Overlap (%)",   "{}_Top3", "mean"),
        ("Top-5 Overlap (%)",   "{}_Top5", "mean")
    ]

    # ---------------------------------------------------------
    # PRINT RESULTS TABLE
    # ---------------------------------------------------------
    print("\n" + "="*85)
    print(f"{'Metric':<25} | {'Standard':<15} | {'Weighted':<15} | {'Quadratic':<15}")
    print("="*85)

    for metric_name, col_template, agg_type in metrics:
        row_values = []
        
        for _, prefix in variants:
            col = col_template.format(prefix)
            
            if col not in df.columns:
                row_values.append("N/A")
                continue
            
            # Get valid numeric data
            data = pd.to_numeric(df[col], errors='coerce').dropna()
            
            if len(data) == 0:
                row_values.append("-")
                continue

            # Calculate Statistic
            if agg_type == "mean":
                if "KT" in col:
                    # KT distance is a plain number (lower is better)
                    val = data.mean()
                    fmt_str = f"{val:.4f}"
                else:
                    # Overlaps/Accuracy are 0-1, convert to %
                    val = data.mean() * 100
                    fmt_str = f"{val:.1f}%"
                    
            elif agg_type == "perfect":
                # Percentage of instances where KT Distance is exactly 0
                perfect_count = (data == 0.0).sum()
                val = (perfect_count / len(data)) * 100
                fmt_str = f"{val:.1f}%"
                
            row_values.append(fmt_str)

        print(f"{metric_name:<25} | {row_values[0]:<15} | {row_values[1]:<15} | {row_values[2]:<15}")

    print("="*85)

    # ---------------------------------------------------------
    # CONDORCET SUBSET ANALYSIS (Sanity Check)
    # ---------------------------------------------------------
    # This checks Condorcet efficiency ONLY on this small subset of files (<=10 candidates)
    # to see if it aligns with the full dataset results.
    
    if "Condorcet_Winner" in df.columns:
        # Filter for rows where a Condorcet Winner actually exists
        cw_subset = df.dropna(subset=["Condorcet_Winner"])
        
        if len(cw_subset) > 0:
            print(f"\nSubset Analysis: Condorcet Efficiency on these {len(cw_subset)} small instances")
            print("-" * 60)
            
            for name, prefix in variants:
                col = f"{prefix}_CW_Found"
                if col in cw_subset.columns:
                    # 1 = Found, 0 = Missed, -1 = N/A
                    valid = cw_subset[cw_subset[col] != -1]
                    rate = valid[col].mean()
                    print(f"{name:<20}: {rate:.1%}")

if __name__ == "__main__":
    analyze_replication_results()