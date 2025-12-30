"""
Sparse Data Regime Experiment
Replicates Section 4.3 of Lanctot et al. (2025) "Soft Condorcet Optimization"

Compares:
- Standard SCO (uniform weights)
- Weighted SCO (Vigna's weighted Kendall-tau)
"""

import random
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable
from tqdm import tqdm
import csv

from sco import update_ratings_batch


# =============================================================================
# DATA GENERATION (Section 4.3 of the paper)
# =============================================================================

def generate_true_ratings(num_agents: int, mean: float = 100.0, std: float = 30.0) -> Dict[int, float]:
    """
    Generate ground truth skill ratings.
    Paper: θ_i ~ N(100, 30)
    """
    return {i: np.random.normal(mean, std) for i in range(num_agents)}


def generate_contest_uniform(agents: List[int], contest_size: int = 4) -> List[int]:
    """Uniform distribution: sample agents uniformly at random."""
    return random.sample(agents, min(contest_size, len(agents)))


def generate_contest_skill_matched(agents: List[int], true_ratings: Dict[int, float],
                                   contest_size: int = 4) -> List[int]:
    """
    Skill-matched distribution: choose agents with similar skill levels.
    Paper: "draw 3 new candidates at random and choosing the one whose
    true rating is closest to the average of the set of agents so far"
    """
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
    """
    Simulate contest by adding noise to true ratings and sorting.
    Paper: P_i(c) = θ_i + ε_{c,i} where ε ~ N(0, 5.0)
    """
    performances = {
        agent: true_ratings[agent] + np.random.normal(0, noise_std)
        for agent in participants
    }
    return sorted(participants, key=lambda a: performances[a], reverse=True)


def ranking_to_pairs(ranking: List[int], weight_fn: Callable = None) -> List[Tuple[int, int, float]]:
    """
    Convert a ranking to (winner, loser, weight) tuples.
    """
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
    """Standard unweighted Kendall-tau."""
    return 1.0


def weight_vigna(i: int, j: int) -> float:
    """
    Vigna's weighted Kendall-tau: w(i,j) = 1/(i+1) + 1/(j+1)
    Emphasizes disagreements at the top of the ranking.
    """
    return 1.0 / (i + 1) + 1.0 / (j + 1)


# =============================================================================
# METRICS
# =============================================================================

def kendall_tau_distance(ranking_a: List[int], ranking_b: List[int]) -> int:
    """Count pairwise disagreements between two rankings."""
    import itertools
    
    pos_a = {agent: i for i, agent in enumerate(ranking_a)}
    pos_b = {agent: i for i, agent in enumerate(ranking_b)}
    common = set(ranking_a) & set(ranking_b)
    
    disagreements = 0
    for i, j in itertools.combinations(common, 2):
        if (pos_a[i] < pos_a[j]) != (pos_b[i] < pos_b[j]):
            disagreements += 1
    return disagreements


def mean_true_rating_distance(predicted: List[int], true_ranking: List[int], 
                               true_ratings: Dict[int, float]) -> float:
    """
    MTRD: Average |θ_i - θ_j| for misranked pairs.
    Paper's secondary metric.
    """
    import itertools
    
    pos_pred = {agent: i for i, agent in enumerate(predicted)}
    pos_true = {agent: i for i, agent in enumerate(true_ranking)}
    common = set(predicted) & set(true_ranking)
    
    discordant = []
    for i, j in itertools.combinations(common, 2):
        if (pos_pred[i] < pos_pred[j]) != (pos_true[i] < pos_true[j]):
            discordant.append((i, j))
    
    if not discordant:
        return 0.0
    return sum(abs(true_ratings[i] - true_ratings[j]) for i, j in discordant) / len(discordant)


# =============================================================================
# EXPERIMENT
# =============================================================================

def generate_tournament_data(num_agents: int, num_contests: int, 
                             distribution: str, weight_fn: Callable,
                             seed: int) -> Tuple[Dict[int, float], List[Tuple[int, int, float]]]:
    """Generate synthetic tournament data."""
    random.seed(seed)
    np.random.seed(seed)
    
    true_ratings = generate_true_ratings(num_agents)
    agents = list(range(num_agents))
    all_pairs = []
    
    for _ in range(num_contests):
        if distribution == "uniform":
            participants = generate_contest_uniform(agents)
        else:  # skill_matched
            participants = generate_contest_skill_matched(agents, true_ratings)
        
        ranking = simulate_contest_outcome(participants, true_ratings)
        pairs = ranking_to_pairs(ranking, weight_fn)
        all_pairs.extend(pairs)
    
    return true_ratings, all_pairs


def train_sco(pairs: List[Tuple[int, int, float]], num_agents: int,
              iterations: int = 10000, batch_size: int = 16,
              lr: float = 0.01, tau: float = 1.0, seed: int = None) -> Dict[int, float]:
    """Train SCO ratings."""
    if seed is not None:
        random.seed(seed)
    ratings = {i: 50.0 for i in range(num_agents)}
    
    for _ in range(iterations):
        batch = random.choices(pairs, k=min(batch_size, len(pairs)))
        update_ratings_batch(ratings, batch, lr=lr, tau=tau)
    
    return ratings


def ratings_to_ranking(ratings: Dict[int, float]) -> List[int]:
    """Convert ratings to ranking (highest first)."""
    return sorted(ratings.keys(), key=lambda a: ratings[a], reverse=True)


@dataclass
class Result:
    num_contests: int
    distribution: str
    weight_name: str
    seed: int
    ktd: int
    mtrd: float


def run_single(num_agents: int, num_contests: int, distribution: str,
               weight_fn: Callable, weight_name: str, seed: int) -> Result:
    """Run one experiment instance."""
    
    true_ratings, pairs = generate_tournament_data(
        num_agents, num_contests, distribution, weight_fn, seed
    )
    
    final_ratings = train_sco(pairs, num_agents, seed=seed + 1000)
    
    sco_ranking = ratings_to_ranking(final_ratings)
    true_ranking = ratings_to_ranking(true_ratings)
    
    ktd = kendall_tau_distance(sco_ranking, true_ranking)
    mtrd = mean_true_rating_distance(sco_ranking, true_ranking, true_ratings)
    
    return Result(num_contests, distribution, weight_name, seed, ktd, mtrd)


def run_experiment(output_file: str = "sparse_results.csv",
                   num_agents: int = 20,
                   contest_counts: List[int] = None,
                   num_seeds: int = 200):
    """
    Run the full experiment comparing uniform vs Vigna weighting.
    """
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
    
    # Save
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["num_contests", "distribution", "weight", "seed", "ktd", "mtrd"])
        for r in results:
            writer.writerow([r.num_contests, r.distribution, r.weight_name, r.seed, r.ktd, r.mtrd])
    
    print(f"Saved to {output_file}")
    return results


def summarize_results(results: List[Result]):
    """Print summary table."""
    from collections import defaultdict
    
    grouped = defaultdict(list)
    for r in results:
        key = (r.num_contests, r.distribution, r.weight_name)
        grouped[key].append(r)
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    for dist in ["uniform", "skill_matched"]:
        print(f"\n{dist.upper()} DISTRIBUTION")
        print("-" * 60)
        print(f"{'contests':>10} {'weight':>10} {'KTD':>15} {'MTRD':>15}")
        print("-" * 60)
        
        for key in sorted(grouped.keys()):
            n, d, w = key
            if d != dist:
                continue
            
            group = grouped[key]
            ktds = [r.ktd for r in group]
            mtrds = [r.mtrd for r in group]
            n_samples = len(ktds)
            
            ktd_mean = np.mean(ktds)
            ktd_ci = 1.96 * np.std(ktds) / np.sqrt(n_samples)
            mtrd_mean = np.mean(mtrds)
            mtrd_ci = 1.96 * np.std(mtrds) / np.sqrt(n_samples)
            
            print(f"{n:>10} {w:>10} {ktd_mean:>7.2f}±{ktd_ci:<5.2f} {mtrd_mean:>7.2f}±{mtrd_ci:<5.2f}")


if __name__ == "__main__":
    results = run_experiment(
        output_file="sparse_results.csv",
        num_agents=20,
        contest_counts=[10, 30, 50, 100],
        num_seeds=50
    )
    summarize_results(results)