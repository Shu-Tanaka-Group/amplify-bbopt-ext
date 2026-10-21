import numpy as np

import amplify_bbopt
from amplify_bbopt import Optimizer, FMTrainer
from amplify_bbopt.encoder import encode_input
from amplify_bbopt.trainer import Dataset, TorchFM
from amplify_bbopt.bbopt_logging import AMPLIFY_BBOPT_LOGGER_NAME

import torch
from numpy.typing import NDArray
from logging import getLogger



logger = getLogger(AMPLIFY_BBOPT_LOGGER_NAME)


class LatestDataset(Dataset):
    def __init__(
        self, x: np.ndarray, y: np.ndarray, size_limit: int | None = None
    ) -> None:
        if size_limit is None:
            self._size_limit = len(x)
        else:
            self._size_limit = size_limit
        self._x: NDArray[np.float64] = np.asarray(x)[-self._size_limit :]
        self._y: NDArray[np.float64] = np.asarray(y)[-self._size_limit :]
        self._check()

    def _check(self) -> None:
        super()._check()
        if self._size_limit <= 0:
            raise ValueError("Dataset size limit must be greater than zero")

    @property
    def size_limit(self) -> int:
        return self._size_limit

    @classmethod
    def empty(cls, num_variables: int, size_limit: int) -> Dataset:  # pyright: ignore[reportIncompatibleMethodOverride]
        return cls(np.empty((0, num_variables)), np.empty(0), size_limit)

    def append(self, x: np.ndarray, y: float | np.ndarray) -> None:
        self._x = np.vstack((self._x, x))[-self._size_limit :]
        self._y = np.append(self._y, y)[-self._size_limit :]
        self._check()


class LatestDatasetOptimizer(Optimizer):
    def __init__(
        self,
        blackbox,
        trainer,
        client,
        training_data: LatestDataset,
        *,
        constraints=None,
        pre_encoding: bool = True,
        seed: int = 0,
    ) -> None:
        if not isinstance(training_data, LatestDataset):
            raise TypeError("Dataset must be an subclass of LatestDataset")
        super().__init__(
            blackbox,
            trainer,
            client,
            training_data=training_data,
            constraints=constraints,
            pre_encoding=pre_encoding,
            seed=seed,
        )
        # make training dataset for surrogate model
        self._training_data = training_data
        if len(training_data.x) > 0:
            self._surrogate_training_data = (
                LatestDataset(
                    encode_input(self._training_data.x, self._blackbox_var_values),
                    self._training_data.y,
                    self._training_data.size_limit,
                )
                if pre_encoding
                else self._training_data
            )
        else:
            self._surrogate_training_data = (
                LatestDataset.empty(
                    len(self._enc_info.variables), training_data.size_limit
                )
                if pre_encoding
                else self._training_data
            )


class BasicFMTrainer(FMTrainer):
    def __init__(self, num_threads: int | None = 8) -> None:
        super().__init__()

    def train(self, dataset: amplify_bbopt.Dataset) -> None:
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
