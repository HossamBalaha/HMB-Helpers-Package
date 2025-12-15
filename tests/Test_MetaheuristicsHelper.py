import unittest
import numpy as np
from HMB.MetaheuristicsHelper import MantaRayForagingOptimizer


class TestMetaheuristicsHelper(unittest.TestCase):
  '''
  Unit test for MantaRayForagingOptimizer on a simple sphere function.
  '''

  def setUp(self):
    self.fitness = lambda x: float(np.sum(np.square(x)))
    self.Ps = 20
    self.D = 5
    self.lb = np.ones(self.D) * -5.0
    self.ub = np.ones(self.D) * 5.0
    # Initialize population within bounds
    self.X = self.lb + (self.ub - self.lb) * np.random.random((self.Ps, self.D))
    self.X = np.clip(self.X, self.lb, self.ub)

  def test_mrfo_one_iteration(self):
    t, T = 1, 10
    Fs = [self.fitness(x) for x in self.X]
    newX, bestSol, bestFit = MantaRayForagingOptimizer(
      self.X, Fs, self.Ps, self.D, self.lb, self.ub, t, T, fitnessFunction=self.fitness
    )
    self.assertEqual(newX.shape, (self.Ps, self.D))
    self.assertEqual(bestSol.shape, (self.D,))
    self.assertTrue(np.isfinite(bestFit))


if __name__ == "__main__":
  unittest.main()
