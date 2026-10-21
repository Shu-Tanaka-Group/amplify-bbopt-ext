import unittest
from itertools import product
from unittest.mock import patch

import numpy as np
import torch
from amplify import FixstarsClient, Poly
from amplify_bbopt import (
    Dataset,
    DiscretizationSpec,
    EncodingMethod,
    RealVariable,
    blackbox,
)
from amplify_bbopt.encoder import encode_input
from amplify_bbopt.trainer import TorchFM

from amplify_bbopt_ext.qubo_conversion import (
    EncodingSwitchOptimizer,
    QUBO_Dw_to_Oh,
    QUBO_Oh_to_Dw,
    _fm_qubo,
    _verify_qubo_conversion,
    X_Real_to_Dw,
    X_Real_to_Oh,
    run,
)
from amplify_bbopt_ext.training import BasicFMTrainer


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


class _FakeAmplifyModel:
    """探索モデルの目的関数を記録し，与えられた解をそのまま返す．"""

    solutions: list[np.ndarray] = []
    last: "_FakeAmplifyModel"

    def __init__(self, variables, client) -> None:
        self.variables = variables
        self.objective = Poly()
        _FakeAmplifyModel.last = self

    def solve(self, constraint_weight: float = 1.0, max_retries: int = 10):
        return list(_FakeAmplifyModel.solutions), None


class EncodingSwitchConsistencyTest(unittest.TestCase):
    """FM の学習時エンコーディングは amplify_bbopt の変数定義に従う．"""

    num_variables, num_bins, lower, upper = 2, 4, -1.0, 1.0

    def _optimizer(self, training_method: EncodingMethod, encoding: str):
        variables = [
            RealVariable(
                bounds=(self.lower, self.upper),
                encoding=training_method,
                discretization_spec=DiscretizationSpec(num_bins=self.num_bins),
            )
            for _ in range(self.num_variables)
        ]

        @blackbox
        def bb_func(x: list[float] = variables) -> float:  # type: ignore
            return float(np.sum(x))

        grid = np.linspace(self.lower, self.upper, self.num_bins)
        data_x = np.array([[grid[0], grid[1]], [grid[2], grid[3]]])
        trainer = BasicFMTrainer()
        num_inputs = len(encode_input(data_x[0], variables))
        torch.manual_seed(0)
        trainer._fm = TorchFM(num_inputs, 3)
        optimizer = EncodingSwitchOptimizer(
            blackbox=bb_func,
            trainer=trainer,
            client=FixstarsClient(),
            training_data=Dataset(data_x, np.zeros(len(data_x))),
            N=self.num_variables,
            L=self.lower,
            M=self.upper,
            q=self.num_bins,
            encoding=encoding,
        )
        return optimizer, variables, _fm_qubo(trainer)

    def _assert_consistent(
        self, training_method: EncodingMethod, encoding: str, search_states: np.ndarray
    ) -> None:
        optimizer, variables, training_qubo = self._optimizer(training_method, encoding)
        _FakeAmplifyModel.solutions = list(search_states)
        with patch("amplify_bbopt_ext.qubo_conversion.AmplifyModel", _FakeAmplifyModel):
            solutions, _ = optimizer.minimize_surrogate()

        self.assertEqual(len(solutions), len(search_states))
        model = _FakeAmplifyModel.last
        offsets = []
        for solution in solutions:
            # サロゲート側のベクトルは学習データと同じ規約で符号化されている
            np.testing.assert_array_equal(
                solution.surrogate, encode_input(solution.blackbox, variables)
            )
            # 探索の目的関数は，有効な状態上で FM の予測値と定数差しか違わない
            mapping = dict(zip(model.variables.tolist(), solution.amplify, strict=True))
            search_energy = float(model.objective.substitute(mapping))
            fm_energy = solution.surrogate @ training_qubo @ solution.surrogate
            offsets.append(search_energy - fm_energy)
        self.assertAlmostEqual(float(np.ptp(offsets)), 0.0, places=8)

    def _all_states(self) -> np.ndarray:
        grid = np.linspace(self.lower, self.upper, self.num_bins)
        return np.array(list(product(grid, repeat=self.num_variables)))

    def test_one_hot_training_domain_wall_search(self) -> None:
        states = X_Real_to_Dw(
            self.num_variables, self.lower, self.upper, self.num_bins, self._all_states()
        )
        self._assert_consistent(EncodingMethod.OneHot, "OhDw", states)

    def test_domain_wall_training_one_hot_search(self) -> None:
        states = X_Real_to_Oh(
            self.num_variables, self.lower, self.upper, self.num_bins, self._all_states()
        )
        self._assert_consistent(EncodingMethod.DomainWall, "DwOh", states)


if __name__ == "__main__":
    unittest.main()
