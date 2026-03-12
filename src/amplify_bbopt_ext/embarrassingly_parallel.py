from amplify_bbopt import Optimizer

from .utils import BasicFMTrainer


def single_run(
    client,
    bb_func,
    k,
    n_iter,
    n_init_data,
    epochs,
    optimizer_params,
    lr_scheduler_class=None,
):
    my_trainer = BasicFMTrainer()
    my_trainer.epochs = epochs
    my_trainer.optimizer_params = optimizer_params
    my_trainer.lr_scheduler_class = lr_scheduler_class
    my_trainer.num_factors = k
    optimizer = Optimizer(
        blackbox=bb_func,
        trainer=my_trainer,
        client=client,
    )
    optimizer.add_random_training_data(num_data=n_init_data)
    optimizer.optimize(n_iter)
    return optimizer
