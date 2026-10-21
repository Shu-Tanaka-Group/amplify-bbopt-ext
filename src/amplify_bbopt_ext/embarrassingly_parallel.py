from amplify_bbopt import Optimizer

from .runner import fm_trainer


def single_run(
    client,
    bb_func,
    k,
    n_iter,
    n_init_data,
    epochs,
    optimizer_params,
    lr_scheduler_class=None,
    lr_scheduler_params=None,
):
    optimizer = Optimizer(
        blackbox=bb_func,
        trainer=fm_trainer(
            k, epochs, optimizer_params, lr_scheduler_class, lr_scheduler_params
        ),
        client=client,
    )
    optimizer.add_random_training_data(num_data=n_init_data)
    optimizer.optimize(n_iter)
    return optimizer
