"""
Full Dataset Condorcet Efficiency Experiment (Verbose & Resumable)
Replicates the Condorcet detection rates cited in Abstract and Section 7.1.

CHANGES:
- Logs EVERY file to CSV, even if no Condorcet Winner exists.
- Helps distinguish between "crashed" and "correctly skipped".
- Uses safe memory management.
"""

import os
import csv
import multiprocessing
import random
import gc
from tqdm import tqdm
from preflibtools.instances import OrdinalInstance

from ground_truth import find_condorcet_winner
from sco import update_ratings_batch

# CONFIGURATION
SAFE_NUM_WORKERS = 2  # Keep this low for stability

def run_custom_sco(dataset, num_candidates, iterations=10000):
    ratings = {i: 50.0 for i in range(1, num_candidates + 1)}
    for _ in range(iterations):
        batch = random.choices(dataset, k=32)
        update_ratings_batch(ratings, batch, lr=0.01, tau=1.0)
    winner = sorted(ratings, key=ratings.get, reverse=True)[0]
    return winner

def load_and_process_file(filepath):
    filename = os.path.basename(filepath)
    try:
        # 1. Check for Condorcet Winner
        cw = find_condorcet_winner(filepath)
        
        if cw is None:
            # RETURN A RESULT RECORDING "NO WINNER"
            return {
                "file": filename,
                "status": "NO_CW",
                "cw": "N/A",
                "uni_ok": "N/A", "wei_ok": "N/A", "quad_ok": "N/A"
            }
            
        # 2. Parse file
        instance = OrdinalInstance()
        instance.parse_file(filepath)
        num_candidates = instance.num_alternatives
        
        # Build datasets
        uniform_pairs = []
        weighted_pairs = []
        quadratic_pairs = []
        
        for ranking in instance.orders:
            count = instance.multiplicity[ranking]
            flat_ranking = []
            for item in ranking:
                if isinstance(item, (list, tuple, set)): flat_ranking.extend(item)
                else: flat_ranking.append(item)
            
            u_batch = []
            w_batch = []
            q_batch = []
            
            for i in range(len(flat_ranking)):
                for j in range(i + 1, len(flat_ranking)):
                    winner, loser = flat_ranking[i], flat_ranking[j]
                    u_batch.append((winner, loser, 1.0))
                    w_val = (1.0 / (i + 1)) + (1.0 / (j + 1))
                    w_batch.append((winner, loser, w_val))
                    q_val = (1.0 / ((i + 1)**2)) + (1.0 / ((j + 1)**2))
                    q_batch.append((winner, loser, q_val))
            
            for _ in range(count):
                uniform_pairs.extend(u_batch)
                weighted_pairs.extend(w_batch)
                quadratic_pairs.extend(q_batch)

        del instance
        
        if not uniform_pairs:
            return {
                "file": filename, "status": "EMPTY",
                "cw": "N/A", "uni_ok": "N/A", "wei_ok": "N/A", "quad_ok": "N/A"
            }

        # 3. Run SCO Models
        win_u = run_custom_sco(uniform_pairs, num_candidates)
        win_w = run_custom_sco(weighted_pairs, num_candidates)
        win_q = run_custom_sco(quadratic_pairs, num_candidates)
        
        del uniform_pairs, weighted_pairs, quadratic_pairs
        gc.collect()
        
        return {
            "file": filename,
            "status": "OK",
            "cw": cw,
            "uni_win": win_u, "wei_win": win_w, "quad_win": win_q,
            "uni_ok": 1 if win_u == cw else 0,
            "wei_ok": 1 if win_w == cw else 0,
            "quad_ok": 1 if win_q == cw else 0
        }
        
    except Exception as e:
        return {
            "file": filename,
            "status": "ERROR",
            "cw": "N/A", "uni_ok": "N/A", "wei_ok": "N/A", "quad_ok": "N/A"
        }

def get_processed_files(output_file):
    processed = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', newline='') as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
                for row in reader:
                    if row:
                        processed.add(row[0])
            except StopIteration:
                pass
    return processed

def main():
    DATA_DIR = "../Data/PrefLib-Data-main"
    OUTPUT_FILE = "condorcet_efficiency_full.csv"
    
    # 1. Find all files
    all_files = []
    print(f"Scanning {DATA_DIR}...")
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if f.endswith('.soc') or f.endswith('.soi'):
                all_files.append(os.path.join(root, f))
                
    # 2. Filter processed
    processed_files = get_processed_files(OUTPUT_FILE)
    files_to_run = [f for f in all_files if os.path.basename(f) not in processed_files]
    
    print(f"Total: {len(all_files)} | Processed: {len(processed_files)} | Remaining: {len(files_to_run)}")
    
    if not files_to_run:
        print("All files processed!")
        return

    # 3. Run
    mode = 'a' if os.path.exists(OUTPUT_FILE) else 'w'
    
    with open(OUTPUT_FILE, mode, newline='') as f:
        writer = csv.writer(f)
        if mode == 'w':
            writer.writerow(["file", "status", "cw", "uni_win", "wei_win", "quad_win", "uni_ok", "wei_ok", "quad_ok"])
        
        with multiprocessing.Pool(SAFE_NUM_WORKERS, maxtasksperchild=10) as pool:
            # We wrap the iterator in try-except to catch the IndexError gracefully
            try:
                iterator = pool.imap_unordered(load_and_process_file, files_to_run)
                for res in tqdm(iterator, total=len(files_to_run), desc="Processing"):
                    if res:
                        writer.writerow([
                            res["file"], res["status"], res["cw"],
                            res.get("uni_win", ""), res.get("wei_win", ""), res.get("quad_win", ""),
                            res["uni_ok"], res["wei_ok"], res["quad_ok"]
                        ])
                        f.flush()
            except Exception as e:
                print(f"\nPool finished with expected termination: {e}")

    print("Done.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()