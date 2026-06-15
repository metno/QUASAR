# QUASAR

QUASAR: QUAntile-based neural network reconstruction for near-Surface Atmospheric vaRiables

A traditional geostatistical spatial reconstruction problem consists of
inferring the value of a variable at an unobserved location from observations
at neighboring locations.
Due to the inherent complexity of the task, including the filtering properties
of the predictor and the uncertainties associated with the observations, it is
often desirable to estimate not only a deterministic prediction but also its
associated uncertainty.

This repository explores the problem using the flexibility of machine learning
models.
Specifically, given the 20 nearest neighboring observations of a geophysical
field (e.g., temperature, precipitation, or other meteorological variables), a
multilayer perceptron (MLP) is trained to reconstruct the full probabilistic
distribution at a target location.
The model outputs a discretized quantile function, providing both a prediction
and an estimate of its uncertainty.


## How to Run

The model and training configurations are described in a config file.
Some examples can be found in the `configs` directory.

### Training

To train a model, run:

```bash
uv run quasar training config.toml
```

A new experiment directory will be created in the configured output location.
This directory contains the trained model checkpoints, the best-performing model, and training metadata.

### Inference

Suppose that the experiment directory created during training is located at `output/XXXX`.
To run inference on the test dataset, execute:

```bash
uv run quasar inference output/XXXX/config.toml
```

The configuration file stored in the experiment directory ensures that inference is performed using the same settings employed during training.

### Hyperparameter Optimization

To perform hyperparameter optimization using Optuna's Tree-structured Parzen Estimator (TPE) sampler, run:

```bash
uv run quasar optimize config.toml
```

This command launches an Optuna study and stores the optimization results in the configured output directory.
