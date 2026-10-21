import matplotlib.pyplot as plt
import numpy as np

from logging import getLogger

import torch
from amplify_bbopt import Dataset, FMTrainer
from amplify_bbopt.bbopt_logging import AMPLIFY_BBOPT_LOGGER_NAME
from amplify_bbopt.trainer import TorchFM

logger = getLogger(AMPLIFY_BBOPT_LOGGER_NAME)


def plot_optimization_history(result, num_initial_data, *, figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    # 最適化履歴からこれまでの最小値（Best Objective Value）の推移を計算
    y_history = result.training_data._y
    best_y = np.minimum.accumulate(y_history)

    # 評価回数（x軸）は 1 からスタート
    x_evals = np.arange(1, len(best_y) + 1)

    # 全サンプリング点の散布図
    ax.scatter(x_evals, y_history, color='tab:blue', alpha=0.3, label='Sampled Points', s=20)

    # これまでの最小値の推移の折れ線
    ax.plot(x_evals, best_y, linestyle='-', color='tab:red', label='Best Objective Value', markersize=6)

    # フェーズの境界線
    ax.axvline(x=num_initial_data, color='k', linestyle='--', alpha=0.5, label='End of Initial Data')

    # 背景を色分けしてフェーズを区別
    ax.axvspan(0, num_initial_data, alpha=0.1, color='gray', label='Initial Sampling Phase')
    ax.axvspan(num_initial_data, len(best_y) + 1, alpha=0.05, color='green', label='FMQA Optimization Phase')

    ax.set_xlabel('Number of evaluations', fontsize=16)
    ax.set_ylabel('Objective value', fontsize=16)
    ax.set_title('Optimization History', fontsize=18)
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, linestyle=':', alpha=0.7)

    return fig, ax


class BasicFMTrainer(FMTrainer):
    def __init__(self, num_threads: int | None = 8) -> None:
        super().__init__()

    def train(self, dataset: Dataset) -> None:
        input_dataset_dim = 2

        if len(dataset.x) == 0 or len(dataset.y) == 0:
            raise ValueError("The dataset must not be empty.")
        if len(dataset.x) != len(dataset.y):
            raise ValueError(
                "The number of input values and output values must be the same."
            )
        if len(dataset.x.shape) != input_dataset_dim:
            raise ValueError("The input dataset must be 2D")

        self._fm = TorchFM(dataset.x.shape[1], self._num_factors)

        optimizer = self._optimizer(
            [self._fm.quadratic, self._fm.linear, self._fm.bias],
            **self._optimizer_params,
        )
        criterion = self._loss()
        scheduler = (
            self._lr_scheduler(optimizer, **self._lr_scheduler_params)
            if self._lr_scheduler is not None
            else None
        )
        x_tensor, y_tensor = (
            torch.from_numpy(dataset.x).float(),
            torch.from_numpy(dataset.y).float(),
        )
        self._fm.train()
        for _ in range(self._epochs):
            optimizer.zero_grad()
            out = self._fm(x_tensor)
            loss = criterion(out, y_tensor)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        # display training loss
        with torch.no_grad():
            if logger is not None:
                self._fm.eval()
                loss = criterion(self._fm(x_tensor), y_tensor)
                logger.info(f"training error: {loss.item():.3e}")
