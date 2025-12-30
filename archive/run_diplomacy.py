import random
import numpy as np
from experiments.archive.diplomacy_loader import load_diplomacy
from sco import update_ratings_batch

def evaluate_accuracy(ratings, test_pairs):
    """
    Calculates how often the higher-rated player actually won in the test set.
    """
    correct = 0
    total = 0
    
    for winner, loser, weight in test_pairs:
        # Default rating is 50.0 if player was never seen in training
        r_winner = ratings.get(winner, 50.0)
        r_loser = ratings.get(loser, 50.0)
        
        if r_winner > r_loser:
            correct += 1
        elif r_winner == r_loser:
            correct += 0.5 # Coin toss for ties
            
        total += 1
        
    return correct / total if total > 0 else 0.0

def run_diplomacy_experiment():
    # 1. Load Data
    # Ensure 'standard_no_press.jsonl' is in the folder!
    print("Loading Diplomacy Data...")
    all_pairs, num_players = load_diplomacy("/home/santerikoivula/Programming/weighted-sco/Data/diplomacy_data.jsonl")
    
    if not all_pairs:
        print("No data found. Please download 'standard_no_press.jsonl'.")
        return

    # 2. Train/Test Split
    # We split by games, not by pairs, to avoid data leakage.
    # But since our loader returns a flat list of pairs, we'll just split that list for now.
    # (A strict split would require splitting the raw games first).
    random.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * 0.8)
    train_data = all_pairs[:split_idx]
    test_data = all_pairs[split_idx:]
    
    print(f"Split: {len(train_data)} training pairs, {len(test_data)} test pairs.")
    
    # 3. Train SCO (Weighted)
    print("Training SCO Model...")
    ratings = {i: 50.0 for i in range(1, num_players + 1)}
    
    # Hyperparameters
    BATCH_SIZE = 1024  # Larger batch for large dataset
    ITERATIONS = 50000 # More iterations needed for 50k players
    LR = 0.1
    
    for t in range(ITERATIONS):
        batch = random.choices(train_data, k=BATCH_SIZE)
        update_ratings_batch(ratings, batch, lr=LR, tau=1.0)
        
        if t % 5000 == 0:
            acc = evaluate_accuracy(ratings, test_pairs=test_data[:1000]) # Quick sample check
            print(f"Iter {t}: Test Accuracy ~{acc:.2%}")

    # 4. Final Evaluation
    final_acc = evaluate_accuracy(ratings, test_data)
    print(f"\nFinal Test Set Pairwise Accuracy: {final_acc:.2%}")
    
    # Optional: Print Top 5 Players
    sorted_players = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 5 Player IDs:", [p[0] for p in sorted_players[:5]])

if __name__ == "__main__":
    run_diplomacy_experiment()