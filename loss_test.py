import unittest
from loss import sigmoid_loss_gradient, sigmoid_loss

class TestLoss(unittest.TestCase):
    def test_gradients(self):
        # These values correspond to the test_cases in loss_test.ts
        # Note: The TS formula was complex, but mathematically simplifies 
        # to the cleaner version used in loss.py. We verify the outputs match.
        test_cases = [
            {'winner': 0.5, 'loser': 0.5, 'tau': 1.0, 'expected': -0.25},
            {'winner': 0.75, 'loser': 0.25, 'tau': 0.9, 'expected': -0.2574},
            {'winner': 0.25, 'loser': 0.75, 'tau': 0.37, 'expected': -0.4415},
        ]
        
        for case in test_cases:
            grad = sigmoid_loss_gradient(case['winner'], case['loser'], case['tau'])
            self.assertAlmostEqual(grad, case['expected'], places=4)
            print(f"Verified: w={case['winner']}, l={case['loser']} -> grad={grad:.4f}")

if __name__ == '__main__':
    unittest.main()