import enum

import msgspec
import torch
import torch.nn as nn


class ModelName(enum.StrEnum):
    BASIC = "basic"
    STATEFUL = "stateful"
    DROPOUT = "dropout"
    RESIDUAL = "residual"


class Activation(enum.StrEnum):
    ReLu = "relu"
    GELU = "gelu"
    LeakyReLU = "leaky_relu"
    Softplus = "softplus"

    def to_module(self) -> nn.Module:
        match self:
            case Activation.ReLu:
                return nn.ReLU()

            case Activation.GELU:
                return nn.GELU()

            case Activation.LeakyReLU:
                return nn.LeakyReLU()

            case Activation.Softplus:
                return nn.Softplus()


# Parameters that need to be specified in the config.toml
class ModelConfig(msgspec.Struct):
    name: ModelName

    # list of layer widths
    sizes: list[int]

    # Which activation function to use
    activation: Activation = Activation.ReLu

    def init(self, din: int, dout: int) -> nn.Module:
        act = self.activation.to_module()
        match self.name:
            case ModelName.BASIC:
                return NeuralNetwork(din, dout, act, self.sizes)

            case ModelName.STATEFUL:
                return StatefulNeuralNetwork(din, dout, act, self.sizes)

            case ModelName.DROPOUT:
                return BatchNormDropOutNN(din, dout, act, self.sizes)

            case ModelName.RESIDUAL:
                return ResNetCDF(din, dout, act, self.sizes)


# A simple neural network
class NeuralNetwork(nn.Module):
    def __init__(self, din: int, dout: int, act: nn.Module, sizes: list[int]):
        super().__init__()

        layers = [nn.Linear(din, sizes[0]), act]
        for i in range(1, len(sizes)):
            layers.append(nn.Linear(sizes[i - 1], sizes[i]))
            layers.append(act)

        self.layers = nn.Sequential(
            *layers,
            nn.Linear(sizes[-1], dout),
        )

    def forward(self, x):
        return self.layers(x)


# Like NeuralNetwork, but the inputs are normalized with a BatchNorm layer
class StatefulNeuralNetwork(nn.Module):
    def __init__(self, din: int, dout: int, act: nn.Module, sizes: list[int]) -> None:
        super().__init__()

        layers = [nn.Linear(din, sizes[0], bias=False), act]
        for i in range(1, len(sizes)):
            layers.append(nn.Linear(sizes[i - 1], sizes[i]))
            layers.append(act)

        self.layers = nn.Sequential(
            nn.BatchNorm1d(din),
            *layers,
            nn.Linear(sizes[-1], dout),
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class BatchDropoutMLP(nn.Module):
    def __init__(self, din: int, dout: int):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.BatchNorm1d(din),
            # Bias should not matter after batchnorm
            nn.Linear(din, dout, bias=False),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor):
        return self.mlp(x)


# Each "inner" layer is composed of [BatchNorm -> Linear -> Dropout]
class BatchNormDropOutNN(nn.Module):
    def __init__(self, din: int, dout: int, act: nn.Module, sizes: list[int]):
        super().__init__()

        layers = [BatchDropoutMLP(din, sizes[0]), act]
        for i in range(1, len(sizes)):
            layers.append(BatchDropoutMLP(sizes[i - 1], sizes[i]))
            layers.append(act)

        self.layers = nn.Sequential(
            *layers,
            nn.Linear(sizes[-1], dout),
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)


# A residual block for fully-connected inputs
class ResidualBlock(nn.Module):
    def __init__(self, width: int, act: nn.Module):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(width, width), act, nn.Linear(width, width)
        )

    def forward(self, x: torch.Tensor):
        # Apply the residual connection: x + block(x)
        return x + self.block(x)


# Residual MLP
class ResNetCDF(nn.Module):
    res_sizes: list[int] = [128, 128, 128, 128, 128, 128]

    def __init__(self, din: int, dout: int, act: nn.Module, sizes: list[int]):
        super().__init__()

        # first layer (projection to hidden width)
        self.input_layer = nn.Sequential(nn.Linear(din, self.res_sizes[0]), act)

        # residual blocks
        # TODO: couldn't this just be a sequential?
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(res_size, act) for res_size in self.res_sizes]
        )

        # MLP
        post_layers = []
        for i in range(1, len(sizes)):
            post_layers.append(nn.Linear(sizes[i - 1], sizes[i]))
            post_layers.append(act)

        self.post_layers = nn.Sequential(*post_layers)

        # output layer: predicts Bernstein coefficients
        self.output_layer = nn.Linear(sizes[-1], dout)

    def forward(self, x: torch.Tensor):
        x = self.input_layer(x)

        # apply residual blocks
        for block in self.res_blocks:
            x = block(x)

        x = self.post_layers(x)
        return self.output_layer(x)
