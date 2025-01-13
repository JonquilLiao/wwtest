import unittest
import numpy as np
from wwtest import wwtest

class TestWWTest(unittest.TestCase):
    def test_wwtest_valid_input(self):
        # Create a symmetric matrix
        mat = np.array([[1, 2, 3], [2, 1, 1], [3, 1, 1]])
        result = wwtest(mat)
        # Check that the p-value is between 0 and 1
        self.assertTrue(0 <= result.p_value <= 1)

    def test_wwtest_non_symmetric(self):
        # Create a non-symmetric matrix
        mat = np.array([[1, 2], [3, 1]])
        with self.assertRaises(ValueError):
            wwtest(mat)

    def test_wwtest_nan_values(self):
        # Create a matrix with NaN values
        mat = np.array([[1, np.nan], [2, 1]])
        with self.assertRaises(ValueError):
            wwtest(mat)

if __name__ == "__main__":
    unittest.main()