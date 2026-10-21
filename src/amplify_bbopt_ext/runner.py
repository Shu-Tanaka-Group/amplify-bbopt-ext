import os
import sys
from collections.abc import Callable

import numpy as np
import torch
from amplify import BaseClient
from amplify_bbopt import Dataset, Optimizer
from amplify_bbopt.blackbox import BlackBoxFuncBase
from amplify_bbopt.trainer import Trainer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from .training import BasicFMTrainer, LatestDataset, LatestDatasetOptimizer

# 並列実行（multiprocessing）で渡す場合は pickle 可能である必要があるため，
# lambda やクロージャではなくモジュールレベル関数の functools.partial を使う．
OptimizerFactory = Callable[
    [BlackBoxFuncBase, Trainer, BaseClient, Dataset, int], Optimizer
]


def fm_trainer(
    num_factors: int,
    epochs: int,
    optimizer_params: dict,
    lr_scheduler_class: type[LRScheduler] | None = None,
    lr_scheduler_params: dict | None = None,
    num_threads: int | None = 8,
) -> BasicFMTrainer:
    if lr_scheduler_class is None and lr_scheduler_params is not None:
        raise ValueError("lr_scheduler_params requires lr_scheduler_class")
    # BasicFMTrainer は scheduler.step() を引数なしで呼ぶため，metrics を要求するものは使えない
    if lr_scheduler_class is not None and issubclass(
        lr_scheduler_class, ReduceLROnPlateau
    ):
        raise ValueError("ReduceLROnPlateau is not supported")

    trainer = BasicFMTrainer(num_threads)
    trainer.num_factors = num_factors
    trainer.epochs = epochs
    trainer.optimizer_params = optimizer_params
    trainer.lr_scheduler_class = lr_scheduler_class
    # FMTrainer 既定の StepLR 用パラメータが他のスケジューラに渡らないよう必ず上書きする
    trainer.lr_scheduler_params = lr_scheduler_params or {}
    _check_training_params(trainer)
    return trainer


def _available_cpu_count() -> int:
    # ジョブスケジューラ等で使用できる CPU が制限されている場合（Linux）はその数を使う
    if sys.platform == "linux":
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _check_training_params(trainer: BasicFMTrainer) -> None:
    # 引数の誤りを学習時ではなく，ブラックボックス関数の評価前に検出するため，
    # ダミーのパラメータで optimizer と scheduler を一度構築する
    dummy = torch.nn.Parameter(torch.zeros(1))
    optimizer = trainer.optimizer_class([dummy], **trainer.optimizer_params)
    if trainer.lr_scheduler_class is not None:
        trainer.lr_scheduler_class(optimizer, **trainer.lr_scheduler_params)


def plain_optimizer(
    blackbox: BlackBoxFuncBase,
    trainer: Trainer,
    client: BaseClient,
    dataset: Dataset,
    seed: int,
) -> Optimizer:
    return Optimizer(blackbox, trainer, client, training_data=dataset, seed=seed)


def latest_optimizer(
    blackbox: BlackBoxFuncBase,
    trainer: Trainer,
    client: BaseClient,
    dataset: Dataset,
    seed: int,
    *,
    size_limit: int,
) -> LatestDatasetOptimizer:
    return LatestDatasetOptimizer(
        blackbox,
        trainer,
        client,
        training_data=LatestDataset(dataset.x, dataset.y, size_limit),
        seed=seed,
    )


def run(
    client: BaseClient,
    bb_func: BlackBoxFuncBase,
    n_iter: int,
    initial_data: np.ndarray,
    *,
    num_factors: int,
    epochs: int,
    optimizer_params: dict,
    lr_scheduler_class: type[LRScheduler] | None = None,
    lr_scheduler_params: dict | None = None,
    num_threads: int | None = None,
    make_optimizer: OptimizerFactory = plain_optimizer,
    max_deduplication_trials: int | None = None,
    seed: int = 0,
) -> Optimizer:
    # BasicFMTrainer の None（スレッド数を変更しない）とは異なり，None は CPU 数を意味する
    trainer = fm_trainer(
        num_factors,
        epochs,
        optimizer_params,
        lr_scheduler_class,
        lr_scheduler_params,
        num_threads if num_threads is not None else _available_cpu_count(),
    )
    dataset_x = np.asarray(initial_data)
    dataset_y = np.array([bb_func(x) for x in dataset_x])
    optimizer = make_optimizer(
        bb_func, trainer, client, Dataset(dataset_x, dataset_y), seed
    )
    optimizer.optimize(n_iter, max_deduplication_trials)
    return optimizer
