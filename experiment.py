import random
from preflibtools.instances import OrdinalInstance
from sco import update_ratings_batch

def load_multi_datasets(filename):
    """
    Parses a SOC/SOI file and returns THREE datasets:
    1. Uniform pairs (weight = 1.0)
    2. Weighted pairs (Vigna weights)
    3. Quadratic pairs (Squared inverse weights)
    """
    instance = OrdinalInstance()
    instance.parse_file(filename)
    
    uniform_pairs = []
    weighted_pairs = []
    quadratic_pairs = []
    
    for ranking in instance.orders:
        count = instance.multiplicity[ranking]
        
        flat_ranking = []
        for item in ranking:
            if isinstance(item, (list, set, tuple)): 
                flat_ranking.extend(item)
            else: 
                flat_ranking.append(item)
            
        uni_batch = []
        wei_batch = []
        quad_batch = []
        
        for i in range(len(flat_ranking)):
            for j in range(i + 1, len(flat_ranking)):
                winner = flat_ranking[i]
                loser = flat_ranking[j]
                
                # 1. Uniform
                uni_batch.append((winner, loser, 1.0))
                
                # 2. Vigna Weighted
                w_val = (1.0 / (i + 1)) + (1.0 / (j + 1))
                wei_batch.append((winner, loser, w_val))
                
                # 3. Quadratic
                q_val = (1.0 / ((i + 1)**2)) + (1.0 / ((j + 1)**2))
                quad_batch.append((winner, loser, q_val))

        for _ in range(count):
            uniform_pairs.extend(uni_batch)
            weighted_pairs.extend(wei_batch)
            quadratic_pairs.extend(quad_batch)
            
    return uniform_pairs, weighted_pairs, quadratic_pairs, instance.num_alternatives

def train_model(dataset, num_candidates, iterations=10000, lr=0.01, tau=1.0):
    """Helper to train a single SCO model."""
    ratings = {i: 50.0 for i in range(1, num_candidates + 1)}
    batch_size = 32
    
    for _ in range(iterations):
        batch = random.choices(dataset, k=batch_size)
        update_ratings_batch(ratings, batch, lr=lr, tau=tau)
        
    return ratings

def run_multi_experiment_on_file(filename):
    """
    Runs Uniform, Weighted, and Quadratic SCO on the file.
    Returns results for all three.
    """
    try:
        uni_data, wei_data, quad_data, num_candidates = load_multi_datasets(filename)
    except Exception as e:
        return None

    if num_candidates < 2 or len(uni_data) == 0:
        return None
    
    # Train models
    ratings_uni = train_model(uni_data, num_candidates)
    ratings_wei = train_model(wei_data, num_candidates)
    ratings_quad = train_model(quad_data, num_candidates)
    
    # Sort rankings
    ranking_uni = sorted(ratings_uni.keys(), key=lambda x: ratings_uni[x], reverse=True)
    ranking_wei = sorted(ratings_wei.keys(), key=lambda x: ratings_wei[x], reverse=True)
    ranking_quad = sorted(ratings_quad.keys(), key=lambda x: ratings_quad[x], reverse=True)
    
    return {
        "filename": filename,
        "num_candidates": num_candidates,
        "ranking_uniform": ranking_uni,
        "ranking_weighted": ranking_wei,
        "ranking_quadratic": ranking_quad
    }