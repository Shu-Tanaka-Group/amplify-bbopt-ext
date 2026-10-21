import unittest
import warnings

import numpy as np
from amplify_bbopt import (
    BinaryVariable,
    DiscreteVariable,
    DiscretizationSpec,
    EncodingMethod,
    IntegerVariable,
    RealVariable,
    blackbox,
)

from amplify_bbopt_ext.initial_sampling import QRS, _covers_all_candidates

class ContinuousInitialSamplingTest(unittest.TestCase):
    def test_lhs_is_reproducible_and_scaled(self) -> None:
        bounds = [(-2.0, 2.0), (10.0, 20.0)]
        first = QRS("lhs", bounds, seed=42).generate(8)
        second = QRS("lhs", bounds, seed=42).generate(8)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (8, 2))
        self.assertTrue(np.all(first >= [-2.0, 10.0]))
        self.assertTrue(np.all(first <= [2.0, 20.0]))

    def test_sobol_is_reproducible(self) -> None:
        bounds = [(0.0, 1.0), (-1.0, 1.0)]
        first = QRS("sobol", bounds, seed=7, scramble=True).generate(8)
        second = QRS("sobol", bounds, seed=7, scramble=True).generate(8)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (8, 2))

    def test_successive_calls_continue_sequence(self) -> None:
        bounds = [(0.0, 1.0), (-1.0, 1.0)]
        for scramble in (True, False):
            sampler = QRS("sobol", bounds, seed=7, scramble=scramble)
            first = sampler.generate(8)
            second = sampler.generate(8)
            whole = QRS("sobol", bounds, seed=7, scramble=scramble).generate(16)

            np.testing.assert_array_equal(np.vstack([first, second]), whole)

        sampler = QRS("lhs", bounds, seed=42)
        self.assertFalse(np.array_equal(sampler.generate(8), sampler.generate(8)))

    def test_sobol_unbalanced_warns(self) -> None:
        sampler = QRS("sobol", [(0.0, 1.0)], seed=0)
        with self.assertWarns(UserWarning):
            sampler.generate(6)

        sampler = QRS("sobol", [(0.0, 1.0)], seed=0)
        sampler.generate(8)
        with self.assertWarns(UserWarning):
            sampler.generate(16)

    def test_sobol_balanced_does_not_warn(self) -> None:
        sampler = QRS("sobol", [(0.0, 1.0)], seed=0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sampler.generate(8)
            sampler.generate(8)
            sampler.generate(16)

class ValidationTest(unittest.TestCase):
    def test_unknown_method_raises(self) -> None:
        for method in ("random", "sobl"):
            with self.assertRaises(ValueError):
                QRS(method, [(0.0, 1.0)])

    def test_method_is_case_insensitive(self) -> None:
        self.assertEqual(QRS("LHS", [(0.0, 1.0)]).method, "lhs")

    def test_empty_possible_values_raises(self) -> None:
        with self.assertRaises(ValueError):
            QRS("lhs", [])

    def test_invalid_bounds_raise(self) -> None:
        for bounds in ((1.0, 0.0), (0.0, 1.0, 2.0)):
            with self.assertRaises(ValueError):
                QRS("lhs", [bounds])

    def test_empty_candidates_raise(self) -> None:
        with self.assertRaises(ValueError):
            QRS("lhs", [[0, 1], []])

    def test_non_positive_n_samples_raises(self) -> None:
        sampler = QRS("lhs", [(0.0, 1.0)])
        for n_samples in (0, -1):
            with self.assertRaises(ValueError):
                sampler.generate(n_samples)

class DiscreteMappingTest(unittest.TestCase):
    def test_mixed_variables_are_mapped(self) -> None:
        possible_values = [(0.0, 1.0), [10, 20], [1, 2, 3]]
        samples = QRS("lhs", possible_values, seed=1).generate(6)

        self.assertEqual(samples.shape, (6, 3))
        self.assertEqual(samples.dtype, np.float64)
        self.assertTrue(np.all((samples[:, 0] >= 0.0) & (samples[:, 0] <= 1.0)))
        self.assertTrue(np.all(np.isin(samples[:, 1], possible_values[1])))
        self.assertTrue(np.all(np.isin(samples[:, 2], possible_values[2])))

    def test_divisible_samples_cover_all_candidates_equally(self) -> None:
        possible_values = [[0, 1], [0.1, 0.2, 0.3, 0.4]]
        for method in ("lhs", "sobol"):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                samples = QRS(method, possible_values, seed=1).generate(8)

            for column, candidates in enumerate(possible_values):
                _, counts = np.unique(samples[:, column], return_counts=True)
                self.assertEqual(len(counts), len(candidates))
                self.assertTrue(np.all(counts == 8 // len(candidates)))

    def test_incomplete_coverage_warns(self) -> None:
        with self.assertWarns(UserWarning):
            QRS("lhs", [[0, 1], [0, 1]], seed=1).generate(1)

    def test_incomplete_coverage_of_later_variable_warns(self) -> None:
        with self.assertWarnsRegex(UserWarning, r"\[1\]"):
            QRS("lhs", [[0, 1], [1, 2, 3, 4, 5]], seed=1).generate(6)

    def test_guaranteed_coverage_without_divisibility_does_not_warn(self) -> None:
        # 6 is not a multiple of 4, but every candidate interval contains a full stratum.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            samples = QRS("lhs", [[1, 2, 3, 4]], seed=0).generate(6)
        self.assertEqual(set(samples[:, 0]), {1, 2, 3, 4})

    def test_covers_all_candidates(self) -> None:
        self.assertTrue(_covers_all_candidates(8, 4))
        self.assertTrue(_covers_all_candidates(6, 4))
        self.assertFalse(_covers_all_candidates(5, 4))
        self.assertFalse(_covers_all_candidates(3, 4))

class BlackBoxTest(unittest.TestCase):
    def test_from_blackbox_follows_variable_order_and_domains(self) -> None:
        @blackbox
        def bb_func(
            x: float = RealVariable((-1.0, 1.0), encoding=EncodingMethod.Amplify),  # type: ignore
            b: list[int] = [BinaryVariable(), BinaryVariable()],  # type: ignore  # noqa: B006
            n: int = IntegerVariable((0, 3)),  # type: ignore
            d: float = DiscreteVariable([0.5, 1.5]),  # type: ignore
        ) -> float:
            return 0.0

        samples = QRS.from_blackbox("sobol", bb_func, seed=0).generate(8)

        self.assertEqual(list(bb_func.flattened_variables_dict), ["x", "b[0]", "b[1]", "n", "d"])
        self.assertEqual(samples.shape, (8, 5))
        self.assertTrue(np.all((samples[:, 0] >= -1.0) & (samples[:, 0] <= 1.0)))
        self.assertTrue(np.all(np.isin(samples[:, 1:3], [0, 1])))
        self.assertTrue(np.all(np.isin(samples[:, 3], [0, 1, 2, 3])))
        self.assertTrue(np.all(np.isin(samples[:, 4], [0.5, 1.5])))

    def test_one_hot_bits_of_discretized_real_variable_are_covered(self) -> None:
        variables = [
            RealVariable((-3.0, 3.0), encoding=EncodingMethod.OneHot, discretization_spec=DiscretizationSpec(num_bins=32))
            for _ in range(2)
        ]

        @blackbox
        def bb_func(x: list[float] = variables) -> float:  # type: ignore  # noqa: B006
            return 0.0

        for method in ("lhs", "sobol"):
            samples = QRS.from_blackbox(method, bb_func, seed=42).generate(32)
            for column, var in enumerate(variables):
                one_hot = np.array([var.encoding.encode_value(value) for value in samples[:, column]])
                np.testing.assert_array_equal(one_hot.sum(axis=0), np.ones(32))

if __name__ == "__main__":
    unittest.main()
