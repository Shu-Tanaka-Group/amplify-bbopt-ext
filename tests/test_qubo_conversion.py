import unittest
from unittest.mock import patch

import numpy as np

from amplify_bbopt_ext.qubo_conversion import (
    QUBO_Dw_to_Oh,
    QUBO_Oh_to_Dw,
    _verify_qubo_conversion,
    X_Real_to_Dw,
    X_Real_to_Oh,
    run,
)


def qubo_energy(x: np.ndarray, qubo: np.ndarray) -> float:
    """Evaluate x.T @ Q @ x for an upper-triangular QUBO matrix."""
    return float(x @ qubo @ x)


class TestQuboConversion(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        self.lower = -3.0
        self.upper = 3.0

    def random_grid_inputs(
        self, num_variables: int, num_bins: int, num_samples: int = 100
    ) -> np.ndarray:
        grid = np.linspace(self.lower, self.upper, num_bins)
        indices = self.rng.integers(
            0, num_bins, size=(num_samples, num_variables)
        )
        return grid[indices]

    def test_one_hot_to_domain_wall_preserves_energy(self) -> None:
        for num_variables, num_bins in ((1, 3), (2, 5), (3, 7)):
            with self.subTest(
                num_variables=num_variables, num_bins=num_bins
            ):
                size = num_variables * num_bins
                one_hot_qubo = np.triu(
                    self.rng.normal(size=(size, size))
                )
                domain_wall_qubo, constant = QUBO_Oh_to_Dw(
                    num_variables, num_bins, one_hot_qubo
                )

                inputs = self.random_grid_inputs(
                    num_variables, num_bins
                )
                one_hot_inputs = X_Real_to_Oh(
                    num_variables,
                    self.lower,
                    self.upper,
                    num_bins,
                    inputs,
                )
                domain_wall_inputs = X_Real_to_Dw(
                    num_variables,
                    self.lower,
                    self.upper,
                    num_bins,
                    inputs,
                )

                for one_hot, domain_wall in zip(
                    one_hot_inputs, domain_wall_inputs, strict=True
                ):
                    energy_before = qubo_energy(
                        one_hot, one_hot_qubo
                    )
                    energy_after = (
                        qubo_energy(domain_wall, domain_wall_qubo)
                        + constant
                    )
                    self.assertAlmostEqual(
                        energy_before, energy_after, places=10
                    )

    def test_conversion_verification_logs_success(self) -> None:
        num_variables, num_bins = 2, 3
        size = num_variables * num_bins
        one_hot_qubo = np.triu(self.rng.normal(size=(size, size)))
        domain_wall_qubo, constant = QUBO_Oh_to_Dw(
            num_variables, num_bins, one_hot_qubo
        )

        with self.assertLogs("amplify_bbopt", level="INFO") as logs:
            states, max_error = _verify_qubo_conversion(
                num_variables,
                num_bins,
                one_hot_qubo,
                domain_wall_qubo,
                "Oh",
                constant,
            )

        self.assertEqual(states, num_bins**num_variables)
        self.assertLessEqual(max_error, 1e-10)
        self.assertIn("QUBO conversion verification passed", logs.output[0])

    def test_conversion_verification_raises_on_mismatch(self) -> None:
        one_hot_qubo = np.zeros((3, 3))
        domain_wall_qubo = np.ones((2, 2))

        with self.assertLogs("amplify_bbopt", level="ERROR") as logs:
            with self.assertRaises(RuntimeError):
                _verify_qubo_conversion(
                    1, 3, one_hot_qubo, domain_wall_qubo, "Oh"
                )

        self.assertIn("QUBO conversion verification failed", logs.output[0])

    @patch("amplify_bbopt_ext.qubo_conversion.EncodingSwitchOptimizer")
    def test_run_uses_robust_deduplication_trial_default(
        self, optimizer_class
    ) -> None:
        optimizer = optimizer_class.return_value

        result = run(
            client=object(),
            bb_func=lambda x: float(np.sum(x)),
            N=2,
            q=3,
            L=-1.0,
            M=1.0,
            encoding="Oh",
            k=2,
            n_iter=1,
            initial_data=np.array([[0.0, 0.0]]),
            epochs=1,
            optimizer_params={"lr": 0.01},
        )

        self.assertIs(result, optimizer)
        optimizer.optimize.assert_called_once_with(
            1, max_deduplication_trials=100
        )

    def test_domain_wall_to_one_hot_preserves_energy(self) -> None:
        for num_variables, num_bins in ((1, 3), (2, 5), (3, 7)):
            with self.subTest(
                num_variables=num_variables, num_bins=num_bins
            ):
                size = num_variables * (num_bins - 1)
                domain_wall_qubo = np.triu(
                    self.rng.normal(size=(size, size))
                )
                one_hot_qubo = QUBO_Dw_to_Oh(
                    num_variables, num_bins, domain_wall_qubo
                )

                inputs = self.random_grid_inputs(
                    num_variables, num_bins
                )
                domain_wall_inputs = X_Real_to_Dw(
                    num_variables,
                    self.lower,
                    self.upper,
                    num_bins,
                    inputs,
                )
                one_hot_inputs = X_Real_to_Oh(
                    num_variables,
                    self.lower,
                    self.upper,
                    num_bins,
                    inputs,
                )

                for domain_wall, one_hot in zip(
                    domain_wall_inputs, one_hot_inputs, strict=True
                ):
                    energy_before = qubo_energy(
                        domain_wall, domain_wall_qubo
                    )
                    energy_after = qubo_energy(
                        one_hot, one_hot_qubo
                    )
                    self.assertAlmostEqual(
                        energy_before, energy_after, places=10
                    )


if __name__ == "__main__":
    unittest.main()
