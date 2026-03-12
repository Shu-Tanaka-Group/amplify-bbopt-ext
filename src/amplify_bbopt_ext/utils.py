from logging import getLogger

import torch
from amplify_bbopt import Dataset, FMTrainer
from amplify_bbopt.bbopt_logging import AMPLIFY_BBOPT_LOGGER_NAME
from amplify_bbopt.trainer import TorchFM

logger = getLogger(AMPLIFY_BBOPT_LOGGER_NAME)


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
