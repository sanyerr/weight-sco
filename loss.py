import math

def sigmoid(x):
    """
    Standard sigmoid function: 1 / (1 + e^-x)
    """
    if x > 20:
        return 1.0
    elif x < -20:
        return 0.0
    return 1 / (1 + math.exp(-x))

def sigmoid_loss(winner_rating, loser_rating, tau=1.0):
    """
    Calculates the soft Kendall-tau loss for a single pair.
    We want to minimize sigmoid(loser - winner).
    """
    diff = loser_rating - winner_rating
    return sigmoid(diff / tau)

def sigmoid_loss_gradient(winner_rating, loser_rating, tau=1.0):
    """
    Returns the magnitude of the gradient for the winner.
    (The loser gets the negative of this).
    """
    diff = loser_rating - winner_rating
    
    # Derivative of sigmoid(x) is sig(x) * (1 - sig(x))
    val = sigmoid(diff / tau)
    derivative = val * (1 - val) / tau
    
    return -derivative