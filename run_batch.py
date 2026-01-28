import os
import csv
import time
import multiprocessing
from tqdm import tqdm
from experiment import run_multi_experiment_on_file
from ground_truth import (
    find_condorcet_winner, 
    find_kemeny_optimal,
    normalized_kendall_tau_distance,
    top_k_overlap,
    top_1_match
)
from preflibtools.instances import OrdinalInstance

# Config
TOP_K_VALUES = [3, 5]
MAX_CANDIDATES = 10 

def process_single_file_multi(filepath):
    try:
        # --- 1. Ground Truths ---
        condorcet_winner = find_condorcet_winner(filepath)
        kemeny_ranking = find_kemeny_optimal(filepath) 

        if kemeny_ranking is None:
            return None

        # --- 2. Run SCO Models ---
        start_time = time.time()
        result = run_multi_experiment_on_file(filepath)
        elapsed = time.time() - start_time
        
        if result is None:
            return None
            
        ranking_uni = result["ranking_uniform"]
        ranking_wei = result["ranking_weighted"]
        ranking_quad = result["ranking_quadratic"]
        num_cands = result["num_candidates"]

        # --- 3. Compute Metrics ---
        
        def compute_metrics(predicted_ranking, true_ranking, cw):
            stats = {}
            if cw is not None:
                stats["cw_found"] = 1 if predicted_ranking[0] == cw else 0
            else:
                stats["cw_found"] = -1 

            dist = normalized_kendall_tau_distance(predicted_ranking, true_ranking)
            stats["kt_dist"] = f"{dist:.4f}"
            
            stats["top1_match"] = top_1_match(predicted_ranking, true_ranking)
            
            for k in TOP_K_VALUES:
                if num_cands >= k:
                    ov = top_k_overlap(predicted_ranking, true_ranking, k)
                    stats[f"top{k}_ov"] = f"{ov:.4f}"
                else:
                    stats[f"top{k}_ov"] = "N/A"
            return stats

        stats_uni = compute_metrics(ranking_uni, kemeny_ranking, condorcet_winner)
        stats_wei = compute_metrics(ranking_wei, kemeny_ranking, condorcet_winner)
        stats_quad = compute_metrics(ranking_quad, kemeny_ranking, condorcet_winner)

        # Build Row
        row = [
            os.path.basename(filepath),
            num_cands,
            str(condorcet_winner) if condorcet_winner is not None else "None",
            
            # Uniform
            stats_uni["cw_found"], stats_uni["kt_dist"], stats_uni["top1_match"],
            stats_uni.get("top3_ov", "N/A"), stats_uni.get("top5_ov", "N/A"),
            
            # Weighted (Vigna)
            stats_wei["cw_found"], stats_wei["kt_dist"], stats_wei["top1_match"],
            stats_wei.get("top3_ov", "N/A"), stats_wei.get("top5_ov", "N/A"),
            
            # Quadratic
            stats_quad["cw_found"], stats_quad["kt_dist"], stats_quad["top1_match"],
            stats_quad.get("top3_ov", "N/A"), stats_quad.get("top5_ov", "N/A"),
            
            f"{elapsed:.4f}"
        ]
        return row
        
    except Exception as e:
        return None

def get_candidate_count(filepath):
    try:
        instance = OrdinalInstance()
        instance.parse_file(filepath)
        return instance.num_alternatives
    except:
        return 999

def main():
    DATA_DIR = "../Data/PrefLib-Data-main"
    OUTPUT_FILE = "replication_results_multi.csv"
    
    print(f"Scanning {DATA_DIR}...")
    valid_files = []
    
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if f.endswith('.soc') or f.endswith('.soi'):
                full_path = os.path.join(root, f)
                if get_candidate_count(full_path) <= MAX_CANDIDATES:
                    valid_files.append(full_path)
            
    print(f"Found {len(valid_files)} files suitable for Kemeny calculation.")
    
    # CSV Header
    header = [
        "Filename", "Num_Candidates", "Condorcet_Winner",
        "Uni_CW_Found", "Uni_KT", "Uni_Top1", "Uni_Top3", "Uni_Top5",
        "Wei_CW_Found", "Wei_KT", "Wei_Top1", "Wei_Top3", "Wei_Top5",
        "Quad_CW_Found", "Quad_KT", "Quad_Top1", "Quad_Top3", "Quad_Top5",
        "Processing_Time"
    ]

    num_workers = max(1, int(multiprocessing.cpu_count() * 0.75))
    
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.imap_unordered(process_single_file_multi, valid_files)
            for row in tqdm(results, total=len(valid_files)):
                if row:
                    writer.writerow(row)
                    f.flush()

    print(f"Done! Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()