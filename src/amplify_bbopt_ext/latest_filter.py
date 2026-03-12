import numpy as np
from amplify_bbopt import Optimizer
from amplify_bbopt.encoder import encode_input
from amplify_bbopt.trainer import Dataset
from numpy.typing import NDArray

from .utils import BasicFMTrainer


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


def run(
    client,
    bb_func,
    size_limit,
    k,
    n_iter,
    initial_data,
    epochs,
    optimizer_params,
    lr_scheduler_class=None,
):
    dataset_x = initial_data.copy()
    dataset_y = np.array([bb_func(x) for x in dataset_x])
    my_trainer = BasicFMTrainer()
    my_trainer.epochs = epochs
    my_trainer.optimizer_params = optimizer_params
    my_trainer.lr_scheduler_class = lr_scheduler_class
    my_trainer.num_factors = k
    optimizer = LatestDatasetOptimizer(
        blackbox=bb_func,
        trainer=my_trainer,
        client=client,
        training_data=LatestDataset(dataset_x, dataset_y, size_limit),
    )
    optimizer.optimize(n_iter)
    return optimizer
