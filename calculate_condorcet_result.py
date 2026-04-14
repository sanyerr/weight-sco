"""
Simple Condorcet Statistics
Reads the mixed-format CSV and calculates efficiency rates.
Ignores files with no Condorcet Winner.
"""
import csv
import numpy as np

def calculate_stats():
    filename = "condorcet_efficiency_full.csv"
    print(f"Reading {filename}...")
    
    # Store success/failure (1/0) for each method
    uni_results = []
    wei_results = []
    quad_results = []
    log_results = []

    with open(filename, 'r') as f:
        reader = csv.reader(f)

        # Skip header if present
        try:
            header = next(reader)
        except StopIteration:
            print("File is empty.")
            return

        for row in reader:
            # Current format (11 columns): file, status, cw, uni_win, wei_win, quad_win, log_win, uni_ok, wei_ok, quad_ok, log_ok
            if len(row) == 11:
                status = row[1]
                if status == "OK":
                    try:
                        u = int(row[7])
                        w = int(row[8])
                        q = int(row[9])
                        l = int(row[10])

                        uni_results.append(u)
                        wei_results.append(w)
                        quad_results.append(q)
                        log_results.append(l)
                    except ValueError:
                        continue

            # Legacy format (9 columns): file, status, cw, uni_win, wei_win, quad_win, uni_ok, wei_ok, quad_ok
            elif len(row) == 9:
                status = row[1]
                if status == "OK":
                    try:
                        u = int(row[6])
                        w = int(row[7])
                        q = int(row[8])

                        uni_results.append(u)
                        wei_results.append(w)
                        quad_results.append(q)
                    except ValueError:
                        continue

    # --- Calculation & Printing ---
    total = len(uni_results)
    print(f"\nAnalyzed {total} elections with a Condorcet Winner.")
    
    if total == 0:
        return

    print("\n" + "="*60)
    print(f"{'Method':<25} | {'Efficiency':<10} | {'Std. Err':<10}")
    print("="*60)

    methods = [
        ("Standard SCO (Uniform)", uni_results),
        ("Weighted SCO (Vigna)", wei_results),
        ("Quadratic SCO", quad_results),
        ("Logarithmic SCO", log_results)
    ]

    for name, data in methods:
        mean = np.mean(data)
        stderr = np.sqrt((mean * (1 - mean)) / total)
        print(f"{name:<25} | {mean:.2%}   | ±{stderr:.2%}")
    
    print("="*60)

if __name__ == "__main__":
    calculate_stats()