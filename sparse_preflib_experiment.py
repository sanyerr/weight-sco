"""
Sparse PrefLib Experiment

Takes complete PrefLib ballots and subsamples candidates to simulate sparsity,
while preserving original global positions for Vigna weighting.
"""

import random
import os
import csv
import itertools
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable
from tqdm import tqdm
from preflibtools.instances import OrdinalInstance

from sco import update_ratings_batch
from ground_truth import find_condorcet_winner


# =============================================================================
# WEIGHT FUNCTIONS
# =============================================================================

def weight_uniform(i: int, j: int) -> float:
    return 1.0


def weight_vigna(i: int, j: int) -> float:
    return 1.0 / (i + 1) + 1.0 / (j + 1)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_preflib_ballots(filepath: str) -> Tuple[List[Tuple[List[int], int]], int]:
    """
    Load PrefLib file and return list of (ranking, count) tuples.
    """
    instance = OrdinalInstance()
    instance.parse_file(filepath)
    
    ballots = []
    for ranking in instance.orders:
        count = instance.multiplicity[ranking]
        
        # Flatten ties
        flat_ranking = []
        for item in ranking:
            if isinstance(item, (list, set, tuple)):
                flat_ranking.extend(item)
            else:
                flat_ranking.append(item)
        
        ballots.append((flat_ranking, count))
    
    return ballots, instance.num_alternatives


def subsample_ballot(ranking: List[int], sample_size: int) -> List[Tuple[int, int]]:
    """
    Subsample candidates from a ballot, keeping their original positions.
    Returns list of (candidate, original_position) tuples, sorted by position.
    """
    if sample_size >= len(ranking):
        return [(c, i) for i, c in enumerate(ranking)]
    
    # Sample indices, then sort to preserve order
    indices = sorted(random.sample(range(len(ranking)), sample_size))
    return [(ranking[i], i) for i in indices]


def ballot_to_pairs(sampled: List[Tuple[int, int]], weight_fn: Callable) -> List[Tuple[int, int, float]]:
    """
    Convert subsampled ballot to weighted pairs.
    sampled: list of (candidate, original_position)
    """
    pairs = []
    for i in range(len(sampled)):
        for j in range(i + 1, len(sampled)):
            winner, winner_pos = sampled[i]
            loser, loser_pos = sampled[j]
            weight = weight_fn(winner_pos, loser_pos)
            pairs.append((winner, loser, weight))
    return pairs


def create_sparse_dataset(ballots: List[Tuple[List[int], int]], 
                          sample_size: int,
                          weight_fn: Callable,
                          seed: int) -> List[Tuple[int, int, float]]:
    """
    Create sparse pairwise dataset from complete ballots.
    """
    random.seed(seed)
    all_pairs = []
    
    for ranking, count in ballots:
        for _ in range(count):
            sampled = subsample_ballot(ranking, sample_size)
            pairs = ballot_to_pairs(sampled, weight_fn)
            all_pairs.extend(pairs)
    
    return all_pairs


# =============================================================================
# TRAINING
# =============================================================================

def train_sco(pairs: List[Tuple[int, int, float]], num_candidates: int,
              iterations: int = 10000, batch_size: int = 32,
              lr: float = 0.01, tau: float = 1.0, seed: int = None) -> Dict[int, float]:
    if seed is not None:
        random.seed(seed)
    
    ratings = {i: 50.0 for i in range(1, num_candidates + 1)}
    
    for _ in range(iterations):
        batch = random.choices(pairs, k=min(batch_size, len(pairs)))
        update_ratings_batch(ratings, batch, lr=lr, tau=tau)
    
    return ratings


def ratings_to_ranking(ratings: Dict[int, float]) -> List[int]:
    return sorted(ratings.keys(), key=lambda a: ratings[a], reverse=True)


# =============================================================================
# METRICS
# =============================================================================

def kendall_tau_distance(ranking_a: List[int], ranking_b: List[int]) -> int:
    pos_a = {agent: i for i, agent in enumerate(ranking_a)}
    pos_b = {agent: i for i, agent in enumerate(ranking_b)}
    common = set(ranking_a) & set(ranking_b)
    
    disagreements = 0
    for i, j in itertools.combinations(common, 2):
        if (pos_a[i] < pos_a[j]) != (pos_b[i] < pos_b[j]):
            disagreements += 1
    return disagreements


def top_1_correct(predicted: List[int], true_ranking: List[int]) -> int:
    return 1 if predicted[0] == true_ranking[0] else 0


def top_k_precision(predicted: List[int], true_ranking: List[int], k: int) -> float:
    pred_top_k = set(predicted[:k])
    true_top_k = set(true_ranking[:k])
    return len(pred_top_k & true_top_k) / k


# =============================================================================
# EXPERIMENT
# =============================================================================

@dataclass
class Result:
    filename: str
    num_candidates: int
    sample_size: int
    weight_name: str
    seed: int
    condorcet_match: int  # 1 if top-1 matches Condorcet winner, 0 otherwise, -1 if no Condorcet
    top1_correct: int
    top5_precision: float
    ktd: int


def run_single_file(filepath: str, sample_size: int, weight_fn: Callable, 
                    weight_name: str, seed: int) -> Result:
    """Run experiment on a single file."""
    
    ballots, num_candidates = load_preflib_ballots(filepath)
    
    # Create sparse dataset
    pairs = create_sparse_dataset(ballots, sample_size, weight_fn, seed)
    
    # Train
    ratings = train_sco(pairs, num_candidates, seed=seed + 1000)
    sco_ranking = ratings_to_ranking(ratings)
    
    # Ground truth: full ranking from complete data (uniform weights)
    full_pairs = create_sparse_dataset(ballots, num_candidates, weight_uniform, seed=0)
    full_ratings = train_sco(full_pairs, num_candidates, seed=42)
    true_ranking = ratings_to_ranking(full_ratings)
    
    # Condorcet winner
    condorcet = find_condorcet_winner(filepath)
    if condorcet is not None:
        condorcet_match = 1 if sco_ranking[0] == condorcet else 0
    else:
        condorcet_match = -1
    
    return Result(
        filename=os.path.basename(filepath),
        num_candidates=num_candidates,
        sample_size=sample_size,
        weight_name=weight_name,
        seed=seed,
        condorcet_match=condorcet_match,
        top1_correct=top_1_correct(sco_ranking, true_ranking),
        top5_precision=top_k_precision(sco_ranking, true_ranking, k=min(5, num_candidates)),
        ktd=kendall_tau_distance(sco_ranking, true_ranking),
    )


def find_preflib_files(root_dir: str, max_candidates: int = 20) -> List[str]:
    """Find .soc and .soi files with <= max_candidates."""
    files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith('.soc') or f.endswith('.soi'):
                filepath = os.path.join(dirpath, f)
                try:
                    instance = OrdinalInstance()
                    instance.parse_file(filepath)
                    if instance.num_alternatives <= max_candidates:
                        files.append(filepath)
                except:
                    pass
    return files


def run_experiment(data_dir: str,
                   output_file: str = "sparse_preflib_results.csv",
                   sample_sizes: List[int] = None,
                   num_seeds: int = 5,
                   max_candidates: int = 20,
                   max_files: int = None):
    """
    Run sparse experiment on PrefLib data.
    """
    if sample_sizes is None:
        sample_sizes = [3, 4, 5, 6, 8, 10]
    
    weight_configs = [
        ("uniform", weight_uniform),
        ("vigna", weight_vigna),
    ]
    
    print(f"Finding PrefLib files in {data_dir}...")
    files = find_preflib_files(data_dir, max_candidates)
    print(f"Found {len(files)} files with <= {max_candidates} candidates")
    
    if max_files and len(files) > max_files:
        files = random.sample(files, max_files)
        print(f"Subsampled to {max_files} files")
    
    total = len(files) * len(sample_sizes) * len(weight_configs) * num_seeds
    print(f"Running {total} experiments...")
    
    results = []
    
    with tqdm(total=total) as pbar:
        for filepath in files:
            for sample_size in sample_sizes:
                for weight_name, weight_fn in weight_configs:
                    for seed in range(num_seeds):
                        try:
                            result = run_single_file(
                                filepath, sample_size, weight_fn, weight_name, seed
                            )
                            results.append(result)
                        except Exception as e:
                            pass  # Skip problematic files
                        pbar.update(1)
    
    # Save
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "num_candidates", "sample_size", "weight", "seed",
                         "condorcet_match", "top1_correct", "top5_precision", "ktd"])
        for r in results:
            writer.writerow([r.filename, r.num_candidates, r.sample_size, r.weight_name,
                             r.seed, r.condorcet_match, r.top1_correct, r.top5_precision, r.ktd])
    
    print(f"Saved to {output_file}")
    return results


def summarize_results(results: List[Result]):
    """Print summary grouped by sample_size and weight."""
    from collections import defaultdict
    import numpy as np
    
    grouped = defaultdict(list)
    for r in results:
        key = (r.sample_size, r.weight_name)
        grouped[key].append(r)
    
    print("\n" + "="*80)
    print("SPARSE PREFLIB RESULTS")
    print("="*80)
    print(f"{'sample':>8} {'weight':>10} {'condorcet%':>12} {'top1%':>10} {'top5_prec':>12} {'ktd':>10}")
    print("-" * 70)
    
    for key in sorted(grouped.keys()):
        sample_size, weight_name = key
        group = grouped[key]
        n = len(group)
        
        # Condorcet match rate (exclude -1s)
        condorcet_valid = [r.condorcet_match for r in group if r.condorcet_match >= 0]
        condorcet_rate = np.mean(condorcet_valid) if condorcet_valid else float('nan')
        
        top1 = np.mean([r.top1_correct for r in group])
        top5 = np.mean([r.top5_precision for r in group])
        ktd = np.mean([r.ktd for r in group])
        
        print(f"{sample_size:>8} {weight_name:>10} {condorcet_rate:>11.1%} {top1:>9.1%} {top5:>11.2f} {ktd:>10.2f}")


if __name__ == "__main__":
    # Example usage
    results = run_experiment(
        data_dir="../Data/PrefLib-Data-main",
        output_file="sparse_preflib_results.csv",
        sample_sizes=[3, 4, 5, 6],
        num_seeds=3,
        max_candidates=15,
        max_files=50  # Small test run
    )
    summarize_results(results)
