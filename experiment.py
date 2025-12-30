import random
import numpy as np
from preflibtools.instances import OrdinalInstance
from sco import update_ratings_batch

def load_preflib_pairs(filename):
    """Parses a SOC/SOI file into a list of (winner, loser) tuples."""
    instance = OrdinalInstance()
    instance.parse_file(filename)
    
    all_pairs = []
    
    # Iterate over unique orders and their counts
    # preflibtools stores unique rankings in 'instance.orders'
    # and the count of each ranking in 'instance.multiplicity'
    for ranking in instance.orders:
        count = instance.multiplicity[ranking]
        
        # Flatten tuple if ties exist
        # Example ranking with ties: (1, (2, 3), 4) -> 1 beats {2,3} beats 4
        # We assume strict orders for simplicity, treating (2,3) as having no pair between them
        flat_ranking = []
        for item in ranking:
            if isinstance(item, (list, set, tuple)): 
                flat_ranking.extend(item)
            else: 
                flat_ranking.append(item)
            
        # Create all pairwise comparisons
        vote_pairs = []
        for i in range(len(flat_ranking)):
            for j in range(i + 1, len(flat_ranking)):
                #add weight based on positions
                weight = (1.0 / (i + 1)) + (1.0 / (j + 1))
                #weight = 1.0
                vote_pairs.append((flat_ranking[i], flat_ranking[j], weight))

        # Add to dataset 'count' times
        for _ in range(count):
            all_pairs.extend(vote_pairs)
            
    return all_pairs, instance.num_alternatives

def run_experiment_on_file(filename):
    """
    Runs SCO on a single file and returns the stats.
    """
    # 1. Hyperparameters (Same as paper)
    BATCH_SIZE = 32
    ITERATIONS = 10000
    LR = 0.01
    TAU = 1.0
    
    try:
        dataset, num_candidates = load_preflib_pairs(filename)
    except Exception as e:
        print(f"Skipping {filename}: {e}")
        return None

    if num_candidates < 2 or len(dataset) == 0:
        return None
    
    ratings = {i: 50.0 for i in range(1, num_candidates + 1)}
    
    # 2. Training Loop
    for t in range(ITERATIONS):
        batch = random.choices(dataset, k=BATCH_SIZE)
        update_ratings_batch(ratings, batch, lr=LR, tau=TAU)

    # 3. Return the results (instead of printing)
    # Sort candidates by rating (descending)
    sorted_ranking = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    top_candidate = sorted_ranking[0][0]
    
    return {
        "filename": filename,
        "num_candidates": num_candidates,
        "num_votes": len(dataset), # Approximation of total pairwise comparisons
        "top_candidate": top_candidate,
        "final_ratings": ratings
    }

if __name__ == "__main__":
    # Test on one file to make sure it still works
    res = run_experiment_on_file("../Data/PrefLib-Data-main/datasets/00001 - irish/00001-00000001.soi")
    print(res["top_candidate"])