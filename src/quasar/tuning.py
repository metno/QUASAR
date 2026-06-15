import functools

import numpy as np
import optuna

from quasar import training
from quasar.config import Config, Optimizer
from quasar.model import Activation


def objective(trial, config: Config):
    n_layers = trial.suggest_int("n_layers", 2, 64, step=2)
    n_neurons = trial.suggest_int("n_neurons", 64, 1024, step=64)
    lr = trial.suggest_float("lr", 1e-6, 5e-3, log=True)
    batch = trial.suggest_categorical("batch", [32, 64, 128, 256, 512, 1024])
    activation = trial.suggest_categorical(
        "activation", [Activation(act.value) for act in Activation]
    )
    optimizer = trial.suggest_categorical(
        "optimizer", [Optimizer(opt.value) for opt in Optimizer]
    )

    layers = [n_neurons for _ in range(n_layers)]
    config.model.sizes = layers
    config.optimizer.learning_rate = lr
    config.dataset.batch_size = batch
    config.model.activation = activation
    config.optimizer.name = optimizer

    model, loss, optimizer, dataloder = config.setup_training()
    best_loss = training.train(model, loss, optimizer, dataloder, config)
    return best_loss


def tune(config: Config):
    obj = functools.partial(objective, config=config)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler()
    )
    study.optimize(obj, n_trials=50)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
