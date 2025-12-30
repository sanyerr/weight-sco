import itertools
from preflibtools.instances import OrdinalInstance

def get_pairwise_matrix(filepath):
    """
    Parses a PrefLib file and returns a dictionary of pairwise wins.
    """
    try:
        instance = OrdinalInstance()
        instance.parse_file(filepath)
    except Exception:
        return None, []

    candidates = list(range(1, instance.num_alternatives + 1))
    wins = {c: {other: 0 for other in candidates} for c in candidates}

    for ranking in instance.orders:
        try:
            count = instance.multiplicity[ranking]
        except KeyError:
            count = 1
        
        flat_ranking = []
        for item in ranking:
            if isinstance(item, (list, tuple, set)):
                flat_ranking.extend(item)
            else:
                flat_ranking.append(item)
            
        for idx_a in range(len(flat_ranking)):
            cand_a = flat_ranking[idx_a]
            
            if cand_a not in wins:
                continue

            for idx_b in range(idx_a + 1, len(flat_ranking)):
                cand_b = flat_ranking[idx_b]
                
                if cand_b in wins[cand_a]:
                    wins[cand_a][cand_b] += count

    return wins, candidates

def find_condorcet_winner(filepath):
    """
    Returns the ID of the Condorcet Winner, or None if none exists.
    """
    wins, candidates = get_pairwise_matrix(filepath)
    if wins is None or not candidates:
        return None

    for cand in candidates:
        is_winner = True
        for opponent in candidates:
            if cand == opponent:
                continue
            
            votes_for = wins.get(cand, {}).get(opponent, 0)
            votes_against = wins.get(opponent, {}).get(cand, 0)
            
            if votes_for <= votes_against:
                is_winner = False
                break
        
        if is_winner:
            return cand

    return None

def normalized_kendall_tau_distance(ranking_a, ranking_b):
    """
    Computes normalized Kendall-tau distance between two full rankings.
    """
    if not ranking_a or not ranking_b:
        return None

    common_candidates = set(ranking_a).intersection(set(ranking_b))
    if len(common_candidates) < 2:
        return 0.0

    r_a = [c for c in ranking_a if c in common_candidates]
    r_b = [c for c in ranking_b if c in common_candidates]

    rank_map_a = {cand: i for i, cand in enumerate(r_a)}
    rank_map_b = {cand: i for i, cand in enumerate(r_b)}
    
    disagreements = 0
    pairs = itertools.combinations(common_candidates, 2)
    n = len(common_candidates)
    
    for i, j in pairs:
        dir_a = rank_map_a[i] < rank_map_a[j]
        dir_b = rank_map_b[i] < rank_map_b[j]
        
        if dir_a != dir_b:
            disagreements += 1
            
    max_dist = (n * (n - 1)) / 2
    if max_dist == 0:
        return 0.0
    return disagreements / max_dist


# ============================================================
# Kemeny-Young Implementation
# ============================================================

def compute_kendall_tau_sum(ranking, wins):
    """
    Compute sum of Kendall-tau distances from a ranking to all votes.
    Uses the pairwise wins matrix.
    """
    total = 0
    for i, cand_a in enumerate(ranking):
        for j in range(i + 1, len(ranking)):
            cand_b = ranking[j]
            total += wins.get(cand_b, {}).get(cand_a, 0)
    return total

def find_kemeny_optimal(filepath):
    """
    Find the Kemeny-Young optimal ranking by brute force.
    Only feasible for small numbers of candidates (≤10).
    Returns the ranking as a list (best first).
    """
    wins, candidates = get_pairwise_matrix(filepath)
    if wins is None or not candidates:
        return None
    
    if len(candidates) > 10:
        return None
    
    best_ranking = None
    best_score = float('inf')
    
    for perm in itertools.permutations(candidates):
        score = compute_kendall_tau_sum(perm, wins)
        if score < best_score:
            best_score = score
            best_ranking = list(perm)
    
    return best_ranking


# ============================================================
# Top-k Metrics
# ============================================================

def top_k_overlap(ranking_a, ranking_b, k):
    """
    Fraction of top-k in ranking_a that also appear in top-k of ranking_b.
    Returns value in [0, 1].
    """
    if not ranking_a or not ranking_b:
        return None
    if len(ranking_a) < k or len(ranking_b) < k:
        return None
    
    set_a = set(ranking_a[:k])
    set_b = set(ranking_b[:k])
    return len(set_a & set_b) / k

def top_k_kendall_tau(ranking_a, ranking_b, k):
    """
    Kendall-tau distance restricted to candidates appearing in 
    top-k of BOTH rankings. Normalized to [0, 1].
    """
    if not ranking_a or not ranking_b:
        return None
    if len(ranking_a) < k or len(ranking_b) < k:
        return None
    
    top_k_a = ranking_a[:k]
    top_k_b = ranking_b[:k]
    
    common = set(top_k_a) & set(top_k_b)
    if len(common) < 2:
        return 0.0
    
    rank_map_a = {c: i for i, c in enumerate(top_k_a) if c in common}
    rank_map_b = {c: i for i, c in enumerate(top_k_b) if c in common}
    
    disagreements = 0
    for i, j in itertools.combinations(common, 2):
        dir_a = rank_map_a[i] < rank_map_a[j]
        dir_b = rank_map_b[i] < rank_map_b[j]
        if dir_a != dir_b:
            disagreements += 1
    
    n = len(common)
    max_dist = n * (n - 1) / 2
    return disagreements / max_dist if max_dist > 0 else 0.0

def top_1_match(ranking_a, ranking_b):
    """
    Check if the top candidate matches.
    Returns 1 if match, 0 otherwise.
    """
    if not ranking_a or not ranking_b:
        return None
    return 1 if ranking_a[0] == ranking_b[0] else 0