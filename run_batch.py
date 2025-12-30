# experiments/run_batch.py
import os
import csv
import time
import multiprocessing
from tqdm import tqdm
from experiment import run_experiment_on_file
from ground_truth import (
    find_condorcet_winner, 
    find_kemeny_optimal,
    normalized_kendall_tau_distance,
    top_k_overlap,
    top_k_kendall_tau,
    top_1_match
)
from preflibtools.instances import OrdinalInstance

# Top-k values to evaluate
TOP_K_VALUES = [3, 5]

def find_soc_soi_files(root_dir):
    target_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith('.soc') or f.endswith('.soi'):
                target_files.append(os.path.join(dirpath, f))
    return target_files

def process_single_file(filepath):
    """
    Worker function for multiprocessing.
    Returns the row to write to CSV, or None if failed.
    """
    try:
        # 1. Ground Truths
        condorcet_winner = find_condorcet_winner(filepath)
        kemeny_ranking = find_kemeny_optimal(filepath)

        # 2. Run SCO
        start_time = time.time()
        result = run_experiment_on_file(filepath)
        elapsed = time.time() - start_time
        
        if result is None:
            return None
            
        num_candidates = result["num_candidates"]
        
        # Convert SCO ratings to ranking
        final_ratings = result["final_ratings"]
        sco_ranking = sorted(final_ratings, key=final_ratings.get, reverse=True)
        
        # --- METRICS (all vs Kemeny-Young) ---
        kt_to_kemeny = "N/A"
        top1_match = "N/A"
        
        if kemeny_ranking is not None:
            dist = normalized_kendall_tau_distance(sco_ranking, kemeny_ranking)
            kt_to_kemeny = f"{dist:.4f}" if dist is not None else "N/A"
            
            t1 = top_1_match(sco_ranking, kemeny_ranking)
            top1_match = str(t1) if t1 is not None else "N/A"
        
        # Top-k metrics
        top_k_metrics = {}
        for k in TOP_K_VALUES:
            if kemeny_ranking is not None and num_candidates >= k:
                overlap = top_k_overlap(sco_ranking, kemeny_ranking, k)
                kt_topk = top_k_kendall_tau(sco_ranking, kemeny_ranking, k)
                
                top_k_metrics[f"top{k}_overlap"] = f"{overlap:.4f}" if overlap is not None else "N/A"
                top_k_metrics[f"top{k}_kt"] = f"{kt_topk:.4f}" if kt_topk is not None else "N/A"
            else:
                top_k_metrics[f"top{k}_overlap"] = "N/A"
                top_k_metrics[f"top{k}_kt"] = "N/A"

        # Build row
        row = [
            os.path.basename(filepath),
            num_candidates,
            str(condorcet_winner) if condorcet_winner is not None else "None",  # Metadata
            kt_to_kemeny,
            top1_match,
        ]
        
        for k in TOP_K_VALUES:
            row.append(top_k_metrics[f"top{k}_overlap"])
            row.append(top_k_metrics[f"top{k}_kt"])
        
        row.append(f"{elapsed:.4f}")
        
        return row
        
    except Exception as e:
        print(f"Error processing {os.path.basename(filepath)}: {e}")
        return None

def get_candidate_count(filepath):
    """
    Safely parses the file using preflibtools to get the exact candidate count.
    """
    try:
        instance = OrdinalInstance()
        instance.parse_file(filepath)
        return instance.num_alternatives
    except Exception:
        return 999

def main():
    DATA_DIR = "../Data/PrefLib-Data-main"
    OUTPUT_FILE = "replication_results.csv"
    
    MAX_CANDIDATES = 10
    
    print(f"Scanning {DATA_DIR} for data files...")
    all_files = find_soc_soi_files(DATA_DIR)
    print(f"Found {len(all_files)} total files.")

    print(f"Filtering for files with <= {MAX_CANDIDATES} candidates...")
    valid_files = []
    
    for f in tqdm(all_files, desc="Checking file headers"):
        if get_candidate_count(f) <= MAX_CANDIDATES:
            valid_files.append(f)
            
    print(f"Found {len(valid_files)} files suitable for Kemeny calculation.")
    
    files_to_run = valid_files

    print(f"Starting run on {len(files_to_run)} files.")

    # Build header
    header = [
        "Filename", 
        "Num_Candidates", 
        "Condorcet_Winner",  # Metadata only
        "KT_to_Kemeny",
        "Top1_Match",
    ]
    
    for k in TOP_K_VALUES:
        header.append(f"Top{k}_Overlap")
        header.append(f"Top{k}_KT")
    
    header.append("Processing_Time")

    # Parallel execution
    num_workers = max(1, int(multiprocessing.cpu_count() * 0.75))
    
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        with multiprocessing.Pool(processes=num_workers) as pool:
            for result_row in tqdm(pool.imap_unordered(process_single_file, files_to_run), 
                                   total=len(files_to_run)):
                if result_row:
                    writer.writerow(result_row)
                    f.flush() 

    print(f"Batch processing complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()