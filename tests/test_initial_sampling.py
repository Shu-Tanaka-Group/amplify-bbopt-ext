import unittest
import warnings

import numpy as np

from amplify_bbopt_ext.initial_sampling import generate

class ContinuousInitialSamplingTest(unittest.TestCase):
    def test_lhs_is_reproducible_and_scaled(self) -> None:
        lower = np.array([-2.0, 10.0])
        upper = np.array([2.0, 20.0])
        first = generate("lhs", 8, lower_bounds=lower, upper_bounds=upper, seed=42)
        second = generate("lhs", 8, lower_bounds=lower, upper_bounds=upper, seed=42)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (8, 2))
        self.assertTrue(np.all(first >= lower))
        self.assertTrue(np.all(first <= upper))

    def test_sobol_is_reproducible(self) -> None:
        lower = np.array([0.0, -1.0])
        upper = np.array([1.0, 1.0])
        first = generate("sobol", 8, lower_bounds=lower, upper_bounds=upper, seed=7, scramble=True)
        second = generate("sobol", 8, lower_bounds=lower, upper_bounds=upper, seed=7, scramble=True)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (8, 2))

class DiscreteMappingTest(unittest.TestCase):
    def test_numeric_candidates_are_mapped(self) -> None:
        values = [[10, 20], [1, 2, 3]]
        mapped = generate("random", 10, variable_values=values, seed=1)
        
        self.assertEqual(mapped.shape, (10, 2))
        self.assertTrue(np.all(np.isin(mapped[:, 0], values[0])))
        self.assertTrue(np.all(np.isin(mapped[:, 1], values[1])))

    def test_mixed_candidates_preserve_python_values(self) -> None:
        values = [[1, 2], ["red", "blue"]]
        mapped = generate("random", 4, variable_values=values, seed=1)

        self.assertEqual(mapped.dtype, np.dtype(object))
        self.assertTrue(np.all(np.isin(mapped[:, 0], values[0])))
        self.assertTrue(np.all(np.isin(mapped[:, 1], values[1])))

    def test_incomplete_coverage_warns(self) -> None:
        values = [[0, 1], ["red", "blue"]]
        with self.assertWarns(UserWarning):
            generate("lhs", 1, variable_values=values, seed=1)

if __name__ == "__main__":
    unittest.main()
