"""
Merged Synthetic Experiment
Replicates Section 7.2 of the paper.

Unified execution for:
1. Global Metrics (Table 7.2.1): KTD, MTRD
2. Top-k Metrics (Table 7.2.2): Top-1 Acc, Top-5 Prec, Top-5 KTD
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
# DATA GENERATION
# =============================================================================

def generate_true_ratings(num_agents: int, mean: float = 100.0, std: float = 30.0) -> Dict[int, float]:
    return {i: np.random.normal(mean, std) for i in range(num_agents)}

def generate_contest_uniform(agents: List[int], contest_size: int = 4) -> List[int]:
    return random.sample(agents, min(contest_size, len(agents)))

def generate_contest_skill_matched(agents: List[int], true_ratings: Dict[int, float], contest_size: int = 4) -> List[int]:
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

def simulate_contest_outcome(participants: List[int], true_ratings: Dict[int, float], noise_std: float = 5.0) -> List[int]:
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
    """Hyperbolic weights: 1/(i+1) + 1/(j+1)"""
    return 1.0 / (i + 1) + 1.0 / (j + 1)

def weight_quadratic(i: int, j: int) -> float:
    """Quadratic weights: 1/(i+1)^2 + 1/(j+1)^2"""
    return (1.0 / ((i + 1)**2)) + (1.0 / ((j + 1)**2))

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

def mean_true_rating_distance(predicted: List[int], true_ranking: List[int], true_ratings: Dict[int, float]) -> float:
    pos_pred = {agent: i for i, agent in enumerate(predicted)}
    pos_true = {agent: i for i, agent in enumerate(true_ranking)}
    common = set(predicted) & set(true_ranking)
    discordant = []
    for i, j in itertools.combinations(common, 2):
        if (pos_pred[i] < pos_pred[j]) != (pos_true[i] < pos_true[j]):
            discordant.append((i, j))
    if not discordant: return 0.0
    return sum(abs(true_ratings[i] - true_ratings[j]) for i, j in discordant) / len(discordant)

def top_k_precision(predicted: List[int], true_ranking: List[int], k: int) -> float:
    pred_top_k = set(predicted[:k])
    true_top_k = set(true_ranking[:k])
    return len(pred_top_k & true_top_k) / k

def top_k_kendall_tau(predicted: List[int], true_ranking: List[int], k: int) -> int:
    true_top_k = set(true_ranking[:k])
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
    return 1 if predicted[0] == true_ranking[0] else 0

# =============================================================================
# EXPERIMENT EXECUTION
# =============================================================================

@dataclass
class Result:
    num_contests: int
    distribution: str
    weight_name: str
    seed: int
    # Global Metrics
    ktd: int
    mtrd: float
    # Top-k Metrics
    top1_correct: int
    top3_precision: float
    top5_precision: float
    top5_ktd: int

def run_single(num_agents: int, num_contests: int, distribution: str,
               weight_fn: Callable, weight_name: str, seed: int) -> Result:
    
    true_ratings, pairs = generate_tournament_data(num_agents, num_contests, distribution, weight_fn, seed)
    final_ratings = train_sco(pairs, num_agents, seed=seed + 1000)
    
    sco_ranking = ratings_to_ranking(final_ratings)
    true_ranking = ratings_to_ranking(true_ratings)
    
    return Result(
        num_contests=num_contests,
        distribution=distribution,
        weight_name=weight_name,
        seed=seed,
        ktd=kendall_tau_distance(sco_ranking, true_ranking),
        mtrd=mean_true_rating_distance(sco_ranking, true_ranking, true_ratings),
        top1_correct=top_1_correct(sco_ranking, true_ranking),
        top3_precision=top_k_precision(sco_ranking, true_ranking, k=3),
        top5_precision=top_k_precision(sco_ranking, true_ranking, k=5),
        top5_ktd=top_k_kendall_tau(sco_ranking, true_ranking, k=5),
    )

def generate_tournament_data(num_agents, num_contests, distribution, weight_fn, seed):
    random.seed(seed)
    np.random.seed(seed)
    true_ratings = generate_true_ratings(num_agents)
    agents = list(range(num_agents))
    all_pairs = []
    for _ in range(num_contests):
        if distribution == "uniform": participants = generate_contest_uniform(agents)
        else: participants = generate_contest_skill_matched(agents, true_ratings)
        ranking = simulate_contest_outcome(participants, true_ratings)
        all_pairs.extend(ranking_to_pairs(ranking, weight_fn))
    return true_ratings, all_pairs

def train_sco(pairs, num_agents, iterations=10000, batch_size=16, lr=0.01, tau=1.0, seed=None):
    if seed is not None: random.seed(seed)
    ratings = {i: 50.0 for i in range(num_agents)}
    for _ in range(iterations):
        batch = random.choices(pairs, k=min(batch_size, len(pairs)))
        update_ratings_batch(ratings, batch, lr=lr, tau=tau)
    return ratings

def ratings_to_ranking(ratings):
    return sorted(ratings.keys(), key=lambda a: ratings[a], reverse=True)

def run_merged_experiment(output_file="synthetic_results_merged.csv", num_agents=20, contest_counts=None, num_seeds=200):
    if contest_counts is None: contest_counts = [10, 50, 100]
    distributions = ["uniform", "skill_matched"]
    
    # Updated configs to include quadratic
    weight_configs = [
        ("uniform", weight_uniform), 
        ("vigna", weight_vigna),
        ("quadratic", weight_quadratic)
    ]
    
    total = len(contest_counts) * len(distributions) * len(weight_configs) * num_seeds
    results = []
    
    with tqdm(total=total, desc="Running Experiments") as pbar:
        for n in contest_counts:
            for dist in distributions:
                for w_name, w_fn in weight_configs:
                    for seed in range(num_seeds):
                        results.append(run_single(num_agents, n, dist, w_fn, w_name, seed))
                        pbar.update(1)
                        
    # Save to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["contests", "dist", "weight", "seed", "ktd", "mtrd", "top1", "top3p", "top5p", "top5_ktd"])
        for r in results:
            writer.writerow([r.num_contests, r.distribution, r.weight_name, r.seed, 
                             r.ktd, r.mtrd, r.top1_correct, r.top3_precision, r.top5_precision, r.top5_ktd])
    
    print_summary(results)
    return results

def print_summary(results):
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results: grouped[(r.num_contests, r.distribution, r.weight_name)].append(r)
    
    print("\n" + "="*100)
    print(f"{'Contests':<8} {'Dist':<15} {'Weight':<10} | {'KTD (Global)':<12} | {'Top-1 Acc':<12} | {'Top-5 Prec':<12}")
    print("="*100)
    
    # Sort key to group by Contest -> Dist -> Weight
    for key in sorted(grouped.keys()):
        n, dist, w = key
        group = grouped[key]
        ktd = np.mean([r.ktd for r in group])
        top1 = np.mean([r.top1_correct for r in group])
        top5 = np.mean([r.top5_precision for r in group])
        
        print(f"{n:<8} {dist:<15} {w:<10} | {ktd:<12.2f} | {top1:<12.2f} | {top5:<12.2f}")

if __name__ == "__main__":
    run_merged_experiment(contest_counts=[10, 30, 50, 100], num_seeds=50)