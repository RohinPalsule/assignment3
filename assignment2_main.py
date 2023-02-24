import numpy as np
from scipy.stats import norm
import unittest
import matplotlib.pyplot as plt

class SignalDetection:
    def __init__(self, hits, misses, falseAlarms, correctRejections):
        self.hits = hits
        self.misses = misses
        self.falseAlarms = falseAlarms
        self.correctRejections = correctRejections
    def hitRate(self):
        return self.hits / (self.hits+self.misses)
    def falseAlarmRate(self):
        return self.falseAlarms / (self.falseAlarms + self.correctRejections)
    def __add__(self, other):
        return SignalDetection(self.hits + other.hits, self.misses + other.misses, self.falseAlarms + 
                other.falseAlarms, self.correctRejections + other.correctRejections)
    def __mul__(self, k):
        return SignalDetection(self.hits *k, self.misses *k, self.falseAlarms *k, self.correctRejections * k)
    def d_prime(self):
        return (norm.ppf(self.hitRate()) - norm.ppf(self.falseAlarmRate()))
    def criterion(self):
        return -0.5 * (norm.ppf(self.hitRate()) + norm.ppf(self.falseAlarmRate()))
    def plot_roc(self):
        plt.plot([0, self.falseAlarmRate(), 1], [0, self.hitRate(), 1])
        plt.xlabel('False Alarm Rate')
        plt.ylabel('Hit Rate')
        plt.title('ROC Plot')
        plt.show()
    def plot_sdt(self):
        plt.axvline((self.d_prime() / 2) + self.criterion(), color = 'yellow')
        plt.axhline(y = 0.4, color = 'g', xmin = 0.5, xmax = (self.d_prime() + 5)/10)
        x_axis = np.arange(-5, 5, 0.01)
        plt.plot(x_axis, norm.pdf(x_axis, 0, 1), color = 'r' ,label = "Noise")
        plt.plot(x_axis, norm.pdf(x_axis, self.d_prime(), 1), color = 'b', label = "Signal")
        plt.ylabel('Probability Density')
        plt.xlabel('Signal Strength')
        plt.title('Signal Detection Theory Plot')
        plt.legend(loc="upper left")
        plt.show()

class TestSignalDetection(unittest.TestCase):
    def test_d_prime_zero(self):
        sd   = SignalDetection(15, 5, 15, 5)
        expected = 0
        obtained = sd.d_prime()
        self.assertAlmostEqual(obtained, expected, places=6)
    def test_d_prime_nonzero(self):
        sd   = SignalDetection(15, 10, 15, 5)
        expected = -0.421142647060282
        obtained = sd.d_prime()
        self.assertAlmostEqual(obtained, expected, places=6)
    def test_criterion_zero(self):
        sd   = SignalDetection(5, 5, 5, 5)
        expected = 0
        obtained = sd.criterion()
        self.assertAlmostEqual(obtained, expected, places=6)
    def test_criterion_nonzero(self):
        sd   = SignalDetection(15, 10, 15, 5)
        expected = -0.463918426665941
        obtained = sd.criterion()
        self.assertAlmostEqual(obtained, expected, places=6)
    def test_addition(self):
        sd = SignalDetection(1, 1, 2, 1) + SignalDetection(2, 1, 1, 3)
        expected = SignalDetection(3, 2, 3, 4).criterion()
        obtained = sd.criterion()
        # Compare calculated and expected criterion
        self.assertEqual(obtained, expected)
    def test_multiplication(self):
        sd = SignalDetection(1, 2, 3, 1) * 4
        expected = SignalDetection(4, 8, 12, 4).criterion()
        obtained = sd.criterion()
        # Compare calculated and expected criterion
        self.assertEqual(obtained, expected)
    def test_corruption(self):
        sd = SignalDetection(2,5,5,5)
        expected = sd.criterion()
        sd.hits = 5
        sd.misses = 6
        sd.falseAlarms = 7
        sd.correctRejections = 8
        obtained = sd.criterion()
        # Compare criterion before and after changing the inputs
        self.assertNotEqual(obtained, expected)
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
