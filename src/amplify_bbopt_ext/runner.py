import numpy as np

from .utils import BasicFMTrainer
from .latest_filter import LatestDataset, LatestDatasetOptimizer

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
