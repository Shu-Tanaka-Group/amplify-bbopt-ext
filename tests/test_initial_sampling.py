import unittest
import warnings

import numpy as np

from amplify_bbopt_ext.initial_sampling import QRS

class ContinuousInitialSamplingTest(unittest.TestCase):
    def test_lhs_is_reproducible_and_scaled(self) -> None:
        lower = np.array([-2.0, 10.0])
        upper = np.array([2.0, 20.0])
        first = QRS("lhs", lower_bounds=lower, upper_bounds=upper, seed=42).generate(8)
        second = QRS("lhs", lower_bounds=lower, upper_bounds=upper, seed=42).generate(8)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (8, 2))
        self.assertTrue(np.all(first >= lower))
        self.assertTrue(np.all(first <= upper))

    def test_sobol_is_reproducible(self) -> None:
        lower = np.array([0.0, -1.0])
        upper = np.array([1.0, 1.0])
        first = QRS("sobol", lower_bounds=lower, upper_bounds=upper, seed=7, scramble=True).generate(8)
        second = QRS("sobol", lower_bounds=lower, upper_bounds=upper, seed=7, scramble=True).generate(8)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (8, 2))

    def test_successive_calls_continue_sequence(self) -> None:
        lower = np.array([0.0, -1.0])
        upper = np.array([1.0, 1.0])
        for scramble in (True, False):
            sampler = QRS("sobol", lower_bounds=lower, upper_bounds=upper, seed=7, scramble=scramble)
            first = sampler.generate(8)
            second = sampler.generate(8)
            whole = QRS("sobol", lower_bounds=lower, upper_bounds=upper, seed=7, scramble=scramble).generate(16)

            np.testing.assert_array_equal(np.vstack([first, second]), whole)

        sampler = QRS("lhs", lower_bounds=lower, upper_bounds=upper, seed=42)
        self.assertFalse(np.array_equal(sampler.generate(8), sampler.generate(8)))

    def test_sobol_unbalanced_warns(self) -> None:
        sampler = QRS("sobol", lower_bounds=[0.0], upper_bounds=[1.0], seed=0)
        with self.assertWarns(UserWarning):
            sampler.generate(6)

        sampler = QRS("sobol", lower_bounds=[0.0], upper_bounds=[1.0], seed=0)
        sampler.generate(8)
        with self.assertWarns(UserWarning):
            sampler.generate(16)

    def test_sobol_balanced_does_not_warn(self) -> None:
        sampler = QRS("sobol", lower_bounds=[0.0], upper_bounds=[1.0], seed=0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sampler.generate(8)
            sampler.generate(8)
            sampler.generate(16)

class ValidationTest(unittest.TestCase):
    def test_unknown_method_raises(self) -> None:
        for method in ("random", "sobl"):
            with self.assertRaises(ValueError):
                QRS(method, lower_bounds=[0.0], upper_bounds=[1.0])

    def test_method_is_case_insensitive(self) -> None:
        self.assertEqual(QRS("LHS", lower_bounds=[0.0], upper_bounds=[1.0]).method, "lhs")

    def test_missing_space_raises(self) -> None:
        with self.assertRaises(ValueError):
            QRS("lhs")

    def test_empty_candidates_raise(self) -> None:
        with self.assertRaises(ValueError):
            QRS("lhs", variable_values=[[0, 1], []])

    def test_non_positive_n_samples_raises(self) -> None:
        sampler = QRS("lhs", lower_bounds=[0.0], upper_bounds=[1.0])
        for n_samples in (0, -1):
            with self.assertRaises(ValueError):
                sampler.generate(n_samples)

class DiscreteMappingTest(unittest.TestCase):
    def test_numeric_candidates_are_mapped(self) -> None:
        values = [[10, 20], [1, 2, 3]]
        mapped = QRS("lhs", variable_values=values, seed=1).generate(6)

        self.assertEqual(mapped.shape, (6, 2))
        self.assertTrue(np.all(np.isin(mapped[:, 0], values[0])))
        self.assertTrue(np.all(np.isin(mapped[:, 1], values[1])))

    def test_mixed_candidates_preserve_python_values(self) -> None:
        values = [[1, 2], ["red", "blue"]]
        mapped = QRS("lhs", variable_values=values, seed=1).generate(4)

        self.assertEqual(mapped.dtype, np.dtype(object))
        self.assertTrue(np.all(np.isin(mapped[:, 0], values[0])))
        self.assertTrue(np.all(np.isin(mapped[:, 1], values[1])))

    def test_divisible_samples_cover_all_candidates(self) -> None:
        values = [[0, 1], ["a", "b", "c", "d"]]
        for method in ("lhs", "sobol"):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                mapped = QRS(method, variable_values=values, seed=1).generate(8)

            for column, candidates in enumerate(values):
                _, counts = np.unique(mapped[:, column], return_counts=True)
                self.assertEqual(len(counts), len(candidates))
                self.assertTrue(np.all(counts == 8 // len(candidates)))

    def test_incomplete_coverage_warns(self) -> None:
        values = [[0, 1], ["red", "blue"]]
        with self.assertWarns(UserWarning):
            QRS("lhs", variable_values=values, seed=1).generate(1)

    def test_incomplete_coverage_of_later_variable_warns(self) -> None:
        values = [[0, 1], [1, 2, 3]]
        with self.assertWarnsRegex(UserWarning, r"\[1\]"):
            QRS("lhs", variable_values=values, seed=1).generate(4)

if __name__ == "__main__":
    unittest.main()
