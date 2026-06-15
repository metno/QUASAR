import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from quasar.config import Config
from quasar.dataset import DataLoader
from quasar.loss import QuantileLoss


@torch.inference_mode()
def infer(
    model: nn.Module,
    loss: QuantileLoss,
    dataloder: DataLoader,
    metadata: pd.DataFrame,
    config: Config,
):
    losses = []
    forecasts = []
    pits = []
    quantiles = []

    for data in dataloder.test_set:
        x, y = data

        y_pred = model(x)
        qloss, batch_forecasts, batch_pits, batch_quantiles = loss.inference(y_pred, y)

        losses.append(qloss)
        pits.append(batch_pits)
        quantiles.append(batch_quantiles)
        forecasts.append(batch_forecasts)

    pits = torch.vstack([p.cpu() for p in pits]).cpu().numpy()
    quantiles = torch.vstack(quantiles).cpu().numpy()
    forecasts = torch.vstack(forecasts).cpu().numpy()
    data = np.hstack((forecasts, pits, quantiles))

    # output columns for the verif input file calculated in this function
    out_cols = ["fcst", "pit"] + [f"q{x:.4f}" for x in loss.submesh()]

    dataset = pd.DataFrame(data, columns=out_cols)
    dataset = pd.concat((metadata, dataset), axis=1)

    test_loss = np.mean(losses)
    print(f"Test loss: {test_loss}")

    outfile = f"{config.output_dir}/verif.input"
    dataset.to_csv(outfile, sep=" ", index=False)
    print(f'verif input saved to "{outfile}"')
