from loss import sigmoid_loss_gradient

def clamp(n, minn=0.0, maxn=100.0):
    return max(min(maxn, n), minn)

def compute_gradient_for_pair(winner_rating, loser_rating, tau=1.0):
    """
    Returns (grad_winner, grad_loser)
    """
    g_winner = sigmoid_loss_gradient(winner_rating, loser_rating, tau)
    g_loser = -g_winner
    return g_winner, g_loser

def update_ratings_batch(ratings, batch_pairs, lr=0.01, tau=1.0):
    """
    Implements one step of Algorithm 1 (Batch SGD).
    Accumulates gradients for all pairs in the batch, then updates.
    """
    # 1. Initialize gradient accumulators
    grad_accum = {player: 0.0 for player in ratings}
    
    # 2. Sum gradients over the batch
    for winner, loser, weight in batch_pairs:
        # If a player in the batch isn't in our ratings yet, init them
        if winner not in ratings: ratings[winner] = 50.0
        if loser not in ratings: ratings[loser] = 50.0
        if winner not in grad_accum: grad_accum[winner] = 0.0
        if loser not in grad_accum: grad_accum[loser] = 0.0
        
        #include weights
        g_w, g_l = compute_gradient_for_pair(ratings[winner], ratings[loser], tau)
        grad_accum[winner] += g_w * weight
        grad_accum[loser] += g_l * weight

    # 3. Apply updates
    # Theta_new = Theta_old - alpha * gradient
    for player, gradient in grad_accum.items():
        if gradient != 0.0:
            ratings[player] -= lr * gradient
            ratings[player] = clamp(ratings[player])
            
    return ratings