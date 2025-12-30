"""
Enhanced Sparse Data Regime Experiment with Top-k Metrics

Same setup as sparse_experiment.py, but evaluates using top-k metrics
where Vigna's weighted Kendall-tau should show its advantage.
"""

import random
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable
from tqdm import tqdm
import csv
import itertools

from sco import update_ratings_batch


# =============================================================================
# DATA GENERATION (same as sparse_experiment.py)
# =============================================================================

def generate_true_ratings(num_agents: int, mean: float = 100.0, std: float = 30.0) -> Dict[int, float]:
    return {i: np.random.normal(mean, std) for i in range(num_agents)}


def generate_contest_uniform(agents: List[int], contest_size: int = 4) -> List[int]:
    return random.sample(agents, min(contest_size, len(agents)))


def generate_contest_skill_matched(agents: List[int], true_ratings: Dict[int, float],
                                   contest_size: int = 4) -> List[int]:
    available = set(agents)
    contest = []
    
    first = random.choice(list(available))
    contest.append(first)
    available.remove(first)
    
    while len(contest) < contest_size and available:
        current_avg = np.mean([true_ratings[a] for a in contest])
        candidates = random.sample(list(available), min(3, len(available)))
        best = min(candidates, key=lambda a: abs(true_ratings[a] - current_avg))
        contest.append(best)
        available.remove(best)
    
    return contest


def simulate_contest_outcome(participants: List[int], true_ratings: Dict[int, float],
                             noise_std: float = 5.0) -> List[int]:
    performances = {
        agent: true_ratings[agent] + np.random.normal(0, noise_std)
        for agent in participants
    }
    return sorted(participants, key=lambda a: performances[a], reverse=True)


def ranking_to_pairs(ranking: List[int], weight_fn: Callable = None) -> List[Tuple[int, int, float]]:
    pairs = []
    for i in range(len(ranking)):
        for j in range(i + 1, len(ranking)):
            weight = weight_fn(i, j) if weight_fn else 1.0
            pairs.append((ranking[i], ranking[j], weight))
    return pairs


# =============================================================================
# WEIGHT FUNCTIONS
# =============================================================================

def weight_uniform(i: int, j: int) -> float:
    return 1.0


def weight_vigna(i: int, j: int) -> float:
    return 1.0 / (i + 1) + 1.0 / (j + 1)


# =============================================================================
# TOP-K METRICS
# =============================================================================

def top_k_precision(predicted: List[int], true_ranking: List[int], k: int) -> float:
    """Fraction of predicted top-k that are in true top-k."""
    pred_top_k = set(predicted[:k])
    true_top_k = set(true_ranking[:k])
    return len(pred_top_k & true_top_k) / k


def top_k_kendall_tau(predicted: List[int], true_ranking: List[int], k: int) -> int:
    """Kendall-tau distance restricted to the true top-k agents."""
    true_top_k = set(true_ranking[:k])
    
    # Filter both rankings to only include true top-k agents
    pred_filtered = [a for a in predicted if a in true_top_k]
    true_filtered = [a for a in true_ranking if a in true_top_k]
    
    pos_pred = {agent: i for i, agent in enumerate(pred_filtered)}
    pos_true = {agent: i for i, agent in enumerate(true_filtered)}
    
    disagreements = 0
    for i, j in itertools.combinations(true_top_k, 2):
        if (pos_pred[i] < pos_pred[j]) != (pos_true[i] < pos_true[j]):
            disagreements += 1
    return disagreements


def top_1_correct(predicted: List[int], true_ranking: List[int]) -> int:
    """1 if top-1 is correct, 0 otherwise."""
    return 1 if predicted[0] == true_ranking[0] else 0


# =============================================================================
# EXPERIMENT
# =============================================================================

def generate_tournament_data(num_agents: int, num_contests: int, 
                             distribution: str, weight_fn: Callable,
                             seed: int) -> Tuple[Dict[int, float], List[Tuple[int, int, float]]]:
    random.seed(seed)
    np.random.seed(seed)
    
    true_ratings = generate_true_ratings(num_agents)
    agents = list(range(num_agents))
    all_pairs = []
    
    for _ in range(num_contests):
        if distribution == "uniform":
            participants = generate_contest_uniform(agents)
        else:
            participants = generate_contest_skill_matched(agents, true_ratings)
        
        ranking = simulate_contest_outcome(participants, true_ratings)
        pairs = ranking_to_pairs(ranking, weight_fn)
        all_pairs.extend(pairs)
    
    return true_ratings, all_pairs


def train_sco(pairs: List[Tuple[int, int, float]], num_agents: int,
              iterations: int = 10000, batch_size: int = 16,
              lr: float = 0.01, tau: float = 1.0, seed: int = None) -> Dict[int, float]:
    if seed is not None:
        random.seed(seed)
    ratings = {i: 50.0 for i in range(num_agents)}
    
    for _ in range(iterations):
        batch = random.choices(pairs, k=min(batch_size, len(pairs)))
        update_ratings_batch(ratings, batch, lr=lr, tau=tau)
    
    return ratings


def ratings_to_ranking(ratings: Dict[int, float]) -> List[int]:
    return sorted(ratings.keys(), key=lambda a: ratings[a], reverse=True)


@dataclass
class Result:
    num_contests: int
    distribution: str
    weight_name: str
    seed: int
    top1_correct: int
    top3_precision: float
    top5_precision: float
    top5_ktd: int


def run_single(num_agents: int, num_contests: int, distribution: str,
               weight_fn: Callable, weight_name: str, seed: int) -> Result:
    
    true_ratings, pairs = generate_tournament_data(
        num_agents, num_contests, distribution, weight_fn, seed
    )
    
    final_ratings = train_sco(pairs, num_agents, seed=seed + 1000)
    
    sco_ranking = ratings_to_ranking(final_ratings)
    true_ranking = ratings_to_ranking(true_ratings)
    
    return Result(
        num_contests=num_contests,
        distribution=distribution,
        weight_name=weight_name,
        seed=seed,
        top1_correct=top_1_correct(sco_ranking, true_ranking),
        top3_precision=top_k_precision(sco_ranking, true_ranking, k=3),
        top5_precision=top_k_precision(sco_ranking, true_ranking, k=5),
        top5_ktd=top_k_kendall_tau(sco_ranking, true_ranking, k=5),
    )


def run_experiment(output_file: str = "topk_results.csv",
                   num_agents: int = 20,
                   contest_counts: List[int] = None,
                   num_seeds: int = 200):
    
    if contest_counts is None:
        contest_counts = [5, 10, 20, 30, 50, 75, 100, 200]
    
    distributions = ["uniform", "skill_matched"]
    weight_configs = [
        ("uniform", weight_uniform),
        ("vigna", weight_vigna),
    ]
    
    total = len(contest_counts) * len(distributions) * len(weight_configs) * num_seeds
    print(f"Running {total} experiments...")
    
    results = []
    
    with tqdm(total=total) as pbar:
        for n_contests in contest_counts:
            for dist in distributions:
                for weight_name, weight_fn in weight_configs:
                    for seed in range(num_seeds):
                        result = run_single(
                            num_agents, n_contests, dist,
                            weight_fn, weight_name, seed
                        )
                        results.append(result)
                        pbar.update(1)
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["num_contests", "distribution", "weight", "seed", 
                         "top1_correct", "top3_precision", "top5_precision", "top5_ktd"])
        for r in results:
            writer.writerow([r.num_contests, r.distribution, r.weight_name, r.seed,
                             r.top1_correct, r.top3_precision, r.top5_precision, r.top5_ktd])
    
    print(f"Saved to {output_file}")
    return results


def summarize_results(results: List[Result]):
    from collections import defaultdict
    
    grouped = defaultdict(list)
    for r in results:
        key = (r.num_contests, r.distribution, r.weight_name)
        grouped[key].append(r)
    
    print("\n" + "="*80)
    print("TOP-K RESULTS SUMMARY")
    print("="*80)
    
    for dist in ["uniform", "skill_matched"]:
        print(f"\n{dist.upper()} DISTRIBUTION")
        print("-" * 75)
        print(f"{'contests':>10} {'weight':>10} {'top1_acc':>12} {'top3_prec':>12} {'top5_prec':>12} {'top5_ktd':>12}")
        print("-" * 75)
        
        for key in sorted(grouped.keys()):
            n, d, w = key
            if d != dist:
                continue
            
            group = grouped[key]
            n_samples = len(group)
            
            top1 = np.mean([r.top1_correct for r in group])
            top3 = np.mean([r.top3_precision for r in group])
            top5 = np.mean([r.top5_precision for r in group])
            ktd5 = np.mean([r.top5_ktd for r in group])
            
            top1_ci = 1.96 * np.std([r.top1_correct for r in group]) / np.sqrt(n_samples)
            top3_ci = 1.96 * np.std([r.top3_precision for r in group]) / np.sqrt(n_samples)
            top5_ci = 1.96 * np.std([r.top5_precision for r in group]) / np.sqrt(n_samples)
            ktd5_ci = 1.96 * np.std([r.top5_ktd for r in group]) / np.sqrt(n_samples)
            
            print(f"{n:>10} {w:>10} {top1:>5.2f}±{top1_ci:<4.2f} {top3:>5.2f}±{top3_ci:<4.2f} {top5:>5.2f}±{top5_ci:<4.2f} {ktd5:>5.2f}±{ktd5_ci:<4.2f}")


if __name__ == "__main__":
    results = run_experiment(
        output_file="topk_results.csv",
        num_agents=20,
        contest_counts=[10, 30, 50, 100],
        num_seeds=50
    )
    summarize_results(results)