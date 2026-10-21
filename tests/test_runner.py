import pickle
import unittest
from functools import partial

import numpy as np
import torch
from amplify import FixstarsClient
from amplify_bbopt import BinaryVariable, Dataset, Optimizer, blackbox
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR

from amplify_bbopt_ext.runner import (
    _available_cpu_count,
    fm_trainer,
    latest_optimizer,
    plain_optimizer,
    run,
)
from amplify_bbopt_ext.training import (
    BasicFMTrainer,
    LatestDataset,
    LatestDatasetOptimizer,
)

_x_list = [BinaryVariable() for _ in range(3)]


@blackbox
def _bb_func(x: list[float] = _x_list) -> float:  # type: ignore
    return float(sum(x))


_initial_data = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=float)

_trainer_params = {"num_factors": 4, "epochs": 10, "optimizer_params": {"lr": 0.1}}


class _StubOptimizer:
    def __init__(self, blackbox, trainer, client, dataset) -> None:
        self.blackbox = blackbox
        self.trainer = trainer
        self.client = client
        self.dataset = dataset
        self.n_iter: int | None = None
        self.max_deduplication_trials: int | None = None

    def optimize(self, n_iter: int, max_deduplication_trials: int | None) -> None:
        self.n_iter = n_iter
        self.max_deduplication_trials = max_deduplication_trials


class FmTrainerTest(unittest.TestCase):
    def test_sets_hyperparameters(self) -> None:
        trainer = fm_trainer(**_trainer_params)

        self.assertIsInstance(trainer, BasicFMTrainer)
        self.assertEqual(trainer.num_factors, 4)
        self.assertEqual(trainer.epochs, 10)
        self.assertEqual(trainer.optimizer_params, {"lr": 0.1})
        self.assertIsNone(trainer.lr_scheduler_class)

    def test_sets_scheduler_and_params(self) -> None:
        trainer = fm_trainer(
            **_trainer_params,
            lr_scheduler_class=CosineAnnealingLR,
            lr_scheduler_params={"T_max": 10},
        )

        self.assertIs(trainer.lr_scheduler_class, CosineAnnealingLR)
        self.assertEqual(trainer.lr_scheduler_params, {"T_max": 10})

    def test_default_scheduler_params_are_not_inherited(self) -> None:
        # FMTrainer 既定の {"step_size": 100, "gamma": 0.8} が残らないこと
        trainer = fm_trainer(
            **_trainer_params,
            lr_scheduler_class=StepLR,
            lr_scheduler_params={"step_size": 5},
        )
        self.assertEqual(trainer.lr_scheduler_params, {"step_size": 5})

        trainer = fm_trainer(**_trainer_params)
        self.assertEqual(trainer.lr_scheduler_params, {})

    def test_params_without_scheduler_raises(self) -> None:
        with self.assertRaises(ValueError):
            fm_trainer(**_trainer_params, lr_scheduler_params={"T_max": 10})

    def test_invalid_scheduler_params_raises(self) -> None:
        with self.assertRaises(TypeError):
            fm_trainer(**_trainer_params, lr_scheduler_class=CosineAnnealingLR)

    def test_invalid_optimizer_params_raises(self) -> None:
        with self.assertRaises(TypeError):
            fm_trainer(num_factors=4, epochs=10, optimizer_params={"foo": 1})

    def test_reduce_lr_on_plateau_raises(self) -> None:
        with self.assertRaises(ValueError):
            fm_trainer(**_trainer_params, lr_scheduler_class=ReduceLROnPlateau)

    def test_trains_with_scheduler(self) -> None:
        trainer = fm_trainer(
            **_trainer_params,
            lr_scheduler_class=CosineAnnealingLR,
            lr_scheduler_params={"T_max": 10},
        )

        trainer.train(Dataset(_initial_data, np.array([1.0, 2.0, 3.0])))

        self.assertIsNotNone(trainer.model)


class OptimizerFactoryTest(unittest.TestCase):
    def _dataset(self) -> Dataset:
        return Dataset(_initial_data, np.array([1.0, 2.0, 3.0]))

    def test_plain_optimizer(self) -> None:
        optimizer = plain_optimizer(
            _bb_func, BasicFMTrainer(), FixstarsClient(), self._dataset()
        )

        self.assertIs(type(optimizer), Optimizer)
        np.testing.assert_array_equal(optimizer.training_data.x, _initial_data)

    def test_latest_optimizer_applies_size_limit(self) -> None:
        make_optimizer = partial(latest_optimizer, size_limit=2)

        optimizer = make_optimizer(
            _bb_func, BasicFMTrainer(), FixstarsClient(), self._dataset()
        )

        self.assertIsInstance(optimizer, LatestDatasetOptimizer)
        self.assertIsInstance(optimizer.training_data, LatestDataset)
        np.testing.assert_array_equal(optimizer.training_data.x, _initial_data[-2:])
        np.testing.assert_array_equal(optimizer.training_data.y, [2.0, 3.0])

    def test_latest_optimizer_partial_is_picklable(self) -> None:
        make_optimizer = partial(latest_optimizer, size_limit=2)

        restored = pickle.loads(pickle.dumps(make_optimizer))

        self.assertEqual(restored.keywords, {"size_limit": 2})


class RunTest(unittest.TestCase):
    def setUp(self) -> None:
        # torch のスレッド数はプロセス全体の設定なので，他のテストに影響しないよう戻す
        num_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, num_threads)

    def _run(self, **kwargs) -> _StubOptimizer:
        result = run(
            FixstarsClient(),
            _bb_func,
            n_iter=5,
            initial_data=_initial_data,
            **_trainer_params,
            make_optimizer=_StubOptimizer,  # pyright: ignore[reportArgumentType]
            **kwargs,
        )
        assert isinstance(result, _StubOptimizer)
        return result

    def test_num_threads_defaults_to_cpu_count(self) -> None:
        torch.set_num_threads(1)

        self._run()

        self.assertEqual(torch.get_num_threads(), _available_cpu_count())

    def test_sets_num_threads(self) -> None:
        self._run(num_threads=2)

        self.assertEqual(torch.get_num_threads(), 2)

    def test_wires_trainer_and_optimizer(self) -> None:
        client = FixstarsClient()

        result = run(
            client,
            _bb_func,
            n_iter=5,
            initial_data=_initial_data,
            **_trainer_params,
            lr_scheduler_class=StepLR,
            lr_scheduler_params={"step_size": 5},
            make_optimizer=_StubOptimizer,  # pyright: ignore[reportArgumentType]
        )

        assert isinstance(result, _StubOptimizer)
        self.assertIs(result.blackbox, _bb_func)
        self.assertIs(result.client, client)
        self.assertIsInstance(result.trainer, BasicFMTrainer)
        self.assertEqual(result.trainer.num_factors, 4)
        self.assertEqual(result.trainer.lr_scheduler_params, {"step_size": 5})
        np.testing.assert_array_equal(result.dataset.x, _initial_data)
        np.testing.assert_array_equal(result.dataset.y, [1.0, 2.0, 3.0])
        self.assertEqual(result.n_iter, 5)
        self.assertIsNone(result.max_deduplication_trials)

    def test_passes_max_deduplication_trials(self) -> None:
        result = self._run(max_deduplication_trials=7)

        self.assertEqual(result.max_deduplication_trials, 7)

    def test_invalid_params_fail_before_evaluating_blackbox(self) -> None:
        calls = []

        def counting_bb_func(x):
            calls.append(x)
            return 0.0

        with self.assertRaises(TypeError):
            run(
                FixstarsClient(),
                counting_bb_func,  # pyright: ignore[reportArgumentType]
                n_iter=5,
                initial_data=_initial_data,
                **_trainer_params,
                lr_scheduler_class=CosineAnnealingLR,
            )
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
