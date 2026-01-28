import os
import random
import multiprocessing
import gc
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import existing tools
from experiment import load_multi_datasets
from ground_truth import find_condorcet_winner
from sco import update_ratings_batch

# --- CONFIGURATION ---
DATA_DIR = "../Data/PrefLib-Data-main"
ITERATIONS = 3000
BATCH_SIZE = 32
LEARNING_RATE = 0.01

# --- EXPERIMENT SETTINGS ---
MAX_INSTANCES_PER_CAT = 500  # Target 500 files per category
NUM_WORKERS = 4              # Adjust based on your CPU
MAX_FILE_SIZE_MB = 2.0       # Skip huge files
CACHE_FILE = "convergence_results.npz"

def get_cw_rank(ratings, cw_id):
    ranked = sorted(ratings.keys(), key=lambda k: ratings[k], reverse=True)
    try:
        return ranked.index(cw_id)
    except ValueError:
        return len(ranked)

def train_single_run(pairs, num_cands, cw_id):
    ratings = {i: 50.0 for i in range(1, num_cands + 1)}
    history = np.zeros(ITERATIONS, dtype=np.int32)
    
    if not pairs:
        return history

    for t in range(ITERATIONS):
        batch = random.choices(pairs, k=BATCH_SIZE)
        update_ratings_batch(ratings, batch, lr=LEARNING_RATE, tau=1.0)
        history[t] = get_cw_rank(ratings, cw_id)
        
    return history

def process_file_safe(filepath):
    try:
        # Skip large files to prevent OOM
        if os.path.getsize(filepath) > (MAX_FILE_SIZE_MB * 1024 * 1024):
            return None

        cw = find_condorcet_winner(filepath)
        if cw is None:
            return None
        
        uni_data, wei_data, _, num_cands = load_multi_datasets(filepath)
        
        # Binary Categorization
        if num_cands <= 10:
            category = "Small"
        else:
            category = "Large"
        
        hist_std = train_single_run(uni_data, num_cands, cw)
        hist_wei = train_single_run(wei_data, num_cands, cw)
        
        # Explicit cleanup
        del uni_data, wei_data
        gc.collect()
        
        return (category, hist_std, hist_wei)

    except Exception:
        return None

def run_training():
    all_files = []
    print(f"Scanning {DATA_DIR}...")
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if f.endswith('.soc') or f.endswith('.soi'):
                all_files.append(os.path.join(root, f))
    
    random.shuffle(all_files)
    print(f"Found {len(all_files)} files. Targeting {MAX_INSTANCES_PER_CAT} per category...")

    results = {
        "Small": {"std": [], "wei": []},
        "Large": {"std": [], "wei": []}
    }
    
    counts = {"Small": 0, "Large": 0}

    with multiprocessing.Pool(NUM_WORKERS) as pool:
        iterator = pool.imap_unordered(process_file_safe, all_files)
        pbar = tqdm(total=MAX_INSTANCES_PER_CAT * 2) 
        
        for res in iterator:
            if res is not None:
                cat, h_std, h_wei = res
                
                if counts[cat] < MAX_INSTANCES_PER_CAT:
                    results[cat]["std"].append(h_std)
                    results[cat]["wei"].append(h_wei)
                    counts[cat] += 1
                    pbar.update(1)
            
            if counts["Small"] >= MAX_INSTANCES_PER_CAT and counts["Large"] >= MAX_INSTANCES_PER_CAT:
                print("\nReached target sample size!")
                pool.terminate()
                break
        
        pbar.close()
    
    return results

def save_results(results):
    # Flatten structure for saving
    np.savez(CACHE_FILE, 
             small_std=results["Small"]["std"],
             small_wei=results["Small"]["wei"],
             large_std=results["Large"]["std"],
             large_wei=results["Large"]["wei"])
    print(f"Data saved to {CACHE_FILE}")

def load_results():
    if not os.path.exists(CACHE_FILE):
        return None
    print(f"Loading cached data from {CACHE_FILE}...")
    data = np.load(CACHE_FILE)
    return {
        "Small": {"std": data["small_std"], "wei": data["small_wei"]},
        "Large": {"std": data["large_std"], "wei": data["large_wei"]}
    }

def plot_results(results):
    print("\nGenerating PDF Plot...")
    
    # Attempt to use a nice style, fallback to default if not found
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        plt.style.use('ggplot')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    categories = ["Small", "Large"]
    
    # Colors (Colorblind safe)
    COLOR_STD = "#377eb8" # Blue
    COLOR_WEI = "#e41a1c" # Red
    
    for idx, cat in enumerate(categories):
        ax = axes[idx]
        
        # Data might be a list or numpy array depending on load source
        mat_std = np.array(list(results[cat]["std"]))
        mat_wei = np.array(list(results[cat]["wei"]))
        data_count = len(mat_std)
        
        if data_count == 0:
            ax.text(0.5, 0.5, "No Data", ha='center')
            continue

        # Calculate Means
        mean_std = np.mean(mat_std, axis=0)
        mean_wei = np.mean(mat_wei, axis=0)
        
        x = np.arange(ITERATIONS)
        
        # Plotting
        ax.plot(x, mean_std, label='Standard SCO', color=COLOR_STD, linewidth=2.5, alpha=0.9)
        ax.plot(x, mean_wei, label='Weighted SCO', color=COLOR_WEI, linewidth=2.5, alpha=0.9)
        
        # Titles (Using standard <= to be safe)
        if cat == "Small":
            title = f"Small Instances (N <= 10)\n(Avg over {data_count} elections)"
        else:
            title = f"Large Instances (N > 10)\n(Avg over {data_count} elections)"

        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Training Iterations", fontsize=12)
        
        if idx == 0:
            ax.set_ylabel("Avg Rank of Condorcet Winner\n(Lower is Better)", fontsize=12)
        
        # Styling
        ax.legend(loc="upper right", frameon=True, framealpha=1.0, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Remove top/right borders for a cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        
        ax.set_ylim(bottom=-0.05)

    plt.tight_layout()
    plt.savefig("convergence_plot.pdf", format='pdf', bbox_inches='tight', dpi=300)
    print("Done! Saved to convergence_plot.pdf")
    # plt.show() # Uncomment if you want to see it pop up

def main():
    # 1. Try to load existing data
    results = load_results()
    
    # 2. If no data, run training
    if results is None:
        results = run_training()
        save_results(results)
    
    # 3. Plot
    plot_results(results)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()