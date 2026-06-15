import time

import numpy as np
import torch
from jaxtyping import Float
from torch import nn
from torch.optim import Optimizer

from quasar.config import Config
from quasar.dataset import DataLoader
from quasar.loss import QuantileLoss
from quasar.monitoring import Stats


def validate(
    model: nn.Module, loss_function: QuantileLoss, dataloader: DataLoader
) -> list[Float]:
    model.eval()

    losses = []
    for x, y in dataloader.validation_set:
        y_pred = model(x)
        losses.append(loss_function(y_pred, y).item())

    return losses


def train(
    model: nn.Module,
    loss_function: QuantileLoss,
    optimizer: Optimizer,
    dataloader: DataLoader,
    config: Config,
) -> float:
    best_loss = float("+inf")
    best_model = nn.Module()
    stats = Stats()

    try:
        for epoch in range(config.epochs):
            epoch_start = time.perf_counter_ns()

            model.train()
            train_losses = []
            for data in dataloader.train_set:
                optimizer.zero_grad()

                x, y = data
                y_pred = model(x)
                loss = loss_function(y_pred, y)

                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            with torch.no_grad():
                validation_losses = validate(model, loss_function, dataloader)

            train_loss = np.mean(train_losses).item()
            validation_loss = np.mean(validation_losses).item()

            epoch_duration = time.perf_counter_ns() - epoch_start
            secs = epoch_duration * 1e-9

            stats.update(epoch, secs, train_loss, validation_loss)

            if epoch % config.save_model_every_n_epoch == 0 and config.save_stats:
                torch.save(model.state_dict(), f"{config.output_dir}/model_{epoch}.pth")

            if validation_loss < best_loss:
                best_loss = validation_loss
                best_model = model

    except KeyboardInterrupt:
        if config.save_stats:
            print("\nCaught interrupt, saving data")
    finally:
        if config.save_stats:
            # Save losses
            stats.save(f"{config.output_dir}/stats.csv")

            # Save best model
            torch.save(best_model.state_dict(), f"{config.output_dir}/best.pth")
        return best_loss
