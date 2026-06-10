# QUASAR

QUASAR: QUAntile-based neural network reconstruction for near-Surface Atmospheric vaRiables

## Scope

A traditional geostatistical spatial reconstruction problem consists of inferring the value of a variable at an unobserved location from observations at neighboring locations. Due to the inherent complexity of the task - including the filtering properties of the predictor and the uncertainties associated with the observations - it is often desirable to estimate not only a deterministic prediction but also its associated uncertainty.

This repository explores the problem using the flexibility of machine learning models. Specifically, given the 20 nearest neighboring observations of a geophysical field (e.g., temperature, precipitation, or other meteorological variables), a multilayer perceptron (MLP) is trained to reconstruct the full probabilistic distribution at a target location. The model outputs a discretized quantile function, providing both a prediction and an estimate of its uncertainty.


## How to Run

Assume your configuration file is located at `~/config/config.toml`.

### Training

To train a model, run:

```bash
uv run mlqc training ~/config/config.toml
```

A new experiment directory will be created in the configured output location. This directory contains the trained model checkpoints, the best-performing model, and training metadata.

### Inference

Suppose the experiment directory created during training is located at `~/mlqc/output/XXXX`. To run inference on the test dataset, execute:

```bash
uv run mlqc inference ~/mlqc/output/XXXX/config.toml
```

The configuration file stored in the experiment directory ensures that inference is performed using the same settings employed during training.

### Hyperparameter Optimization

To perform hyperparameter optimization using Optuna's Tree-structured Parzen Estimator (TPE) sampler, run:

```bash
uv run mlqc hyp_opt ~/config/config.toml
```

This command launches an Optuna study and stores the optimization results in the configured output directory.


## Repository structure

- `src`: directory containing the python source code
- `local_test_data`: directory with data for local testing, namely 813 lines of a synthetic dataset (gaussian random field).
- `design_docs`: directory with notes about requirements and the project design ideas.
- `config.toml`: file that specifies the training configuration.




