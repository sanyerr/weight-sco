"""
Quick tests for Kemeny-Young and top-k metrics.
"""
import sys
sys.path.insert(0, '/home/claude')

from ground_truth import (
    compute_kendall_tau_sum,
    find_kemeny_optimal,
    normalized_kendall_tau_distance,
    top_k_overlap,
    top_k_kendall_tau,
    top_1_match
)

def test_top_k_metrics():
    """Test top-k metric functions."""
    print("Testing top-k metrics...")
    
    # Two rankings over candidates 1-5
    ranking_a = [1, 2, 3, 4, 5]  # Ground truth
    ranking_b = [1, 3, 2, 4, 5]  # Swap positions 2 and 3
    
    # Top-1 should match
    assert top_1_match(ranking_a, ranking_b) == 1
    print("  top_1_match: PASS")
    
    # Top-3 overlap: {1,2,3} vs {1,3,2} -> all 3 match -> 1.0
    assert top_k_overlap(ranking_a, ranking_b, 3) == 1.0
    print("  top_k_overlap(k=3): PASS")
    
    # Top-3 Kendall-tau: 1 disagreement (2 vs 3), normalized by 3 pairs
    kt = top_k_kendall_tau(ranking_a, ranking_b, 3)
    assert abs(kt - 1/3) < 0.001, f"Expected ~0.333, got {kt}"
    print("  top_k_kendall_tau(k=3): PASS")
    
    # Different top elements
    ranking_c = [5, 4, 3, 2, 1]  # Reversed
    assert top_1_match(ranking_a, ranking_c) == 0
    print("  top_1_match (different): PASS")
    
    # Top-3 overlap: {1,2,3} vs {5,4,3} -> only 3 matches -> 1/3
    assert abs(top_k_overlap(ranking_a, ranking_c, 3) - 1/3) < 0.001
    print("  top_k_overlap (partial): PASS")
    
    print("All top-k tests passed!\n")

def test_kendall_tau_sum():
    """Test the Kendall-tau sum computation."""
    print("Testing Kendall-tau sum computation...")
    
    # Example from the paper (Table 1):
    # Votes: 1: A>B>C, 1: A>C>B, 2: C>A>B, 1: B>C>A
    # Using numeric IDs: A=1, B=2, C=3
    
    # Pairwise wins matrix:
    # wins[a][b] = number of voters preferring a over b
    wins = {
        1: {1: 0, 2: 4, 3: 2},  # A beats B 4 times, A beats C 2 times
        2: {1: 1, 2: 0, 3: 2},  # B beats A 1 time, B beats C 2 times
        3: {1: 3, 2: 3, 3: 0},  # C beats A 3 times, C beats B 3 times
    }
    
    # Optimal ranking should be C > A > B (from paper Figure 1)
    # Score = disagreements = wins[A][C] + wins[B][C] + wins[B][A] = 2 + 2 + 1 = 5
    optimal = [3, 1, 2]  # C, A, B
    score = compute_kendall_tau_sum(optimal, wins)
    assert score == 5, f"Expected 5, got {score}"
    print(f"  Optimal ranking [C,A,B] score: {score} (expected 5): PASS")
    
    # Non-optimal: A > B > C
    non_opt = [1, 2, 3]
    score2 = compute_kendall_tau_sum(non_opt, wins)
    assert score2 > score, f"Non-optimal should have higher score"
    print(f"  Non-optimal [A,B,C] score: {score2} > 5: PASS")
    
    print("Kendall-tau sum tests passed!\n")

def test_normalized_kendall_tau():
    """Test normalized Kendall-tau distance."""
    print("Testing normalized Kendall-tau distance...")
    
    # Identical rankings
    r1 = [1, 2, 3, 4, 5]
    assert normalized_kendall_tau_distance(r1, r1) == 0.0
    print("  Identical rankings -> 0.0: PASS")
    
    # Reversed rankings (maximum distance)
    r2 = [5, 4, 3, 2, 1]
    assert normalized_kendall_tau_distance(r1, r2) == 1.0
    print("  Reversed rankings -> 1.0: PASS")
    
    # One swap
    r3 = [1, 3, 2, 4, 5]
    # 1 disagreement out of C(5,2)=10 pairs -> 0.1
    dist = normalized_kendall_tau_distance(r1, r3)
    assert abs(dist - 0.1) < 0.001, f"Expected 0.1, got {dist}"
    print(f"  One swap -> {dist:.3f} (expected 0.1): PASS")
    
    print("Normalized Kendall-tau tests passed!\n")

if __name__ == "__main__":
    test_top_k_metrics()
    test_kendall_tau_sum()
    test_normalized_kendall_tau()
    print("=" * 50)
    print("All tests passed!")
