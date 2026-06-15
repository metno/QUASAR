import datetime
import enum
import os
import pickle
import tomllib
from typing import Self

import msgspec
import tomli_w
import torch

from quasar.dataset import DatasetConfig
from quasar.loss import QuantileLoss, QuantileLossConfig
from quasar.model import ModelConfig


class Optimizer(enum.StrEnum):
    Sgd = "sgd"
    Adam = "adam"
    AdamW = "adamw"
    AdaGrad = "adagrad"


class OptimizerConfig(msgspec.Struct):
    name: Optimizer
    learning_rate: float

    def instantiate(self, params) -> torch.optim.Optimizer:
        match self.name:
            case Optimizer.Adam:
                opt = torch.optim.Adam
            case Optimizer.AdamW:
                opt = torch.optim.AdamW
            case Optimizer.Sgd:
                opt = torch.optim.SGD
            case Optimizer.AdaGrad:
                opt = torch.optim.Adagrad

        return opt(params, lr=self.learning_rate)


class Config(msgspec.Struct, tag=True):
    dataset: DatasetConfig
    model: ModelConfig
    loss: QuantileLossConfig
    optimizer: OptimizerConfig

    epochs: int
    save_model_every_n_epoch: int = 1
    save_stats: bool = True

    # output directory where model checkpoints will be saved
    output_dir: str = "./"
    _inference_dir: str = ""

    # Seed for Torch RNG
    seed: None | int = torch.seed()
    _device: torch.device = torch.device("cpu")

    def __post_init__(self):
        if acc := torch.accelerator.current_accelerator(check_available=True):
            self._device = acc

        print(f"Using device: '{self._device}'")

    @classmethod
    def new(cls, file: str, inference: bool) -> Self:
        with open(file, "r") as infile:
            text = infile.read()

        config = msgspec.toml.decode(text, type=cls)

        if config.seed is not None:
            torch.manual_seed(config.seed)

        print(f"Using seed: {config.seed}")

        if inference:
            inference_dir, _ = os.path.split(os.path.abspath(file))
            # Save inference in a subdirectory of the training output dir
            config._inference_dir = os.path.abspath(inference_dir)
            config.output_dir = config._inference_dir

        config._set_and_create_output_dir()

        if not inference:
            config.save(file, text)

        return config

    def _set_and_create_output_dir(self):
        path = os.path.abspath(os.getcwd())
        now = datetime.datetime.now(datetime.timezone.utc)

        self.output_dir = os.path.join(
            path,
            self.output_dir,
            f"{now.year}{now.month:02}{now.day:02}_{now.hour:02}{now.minute:02}{now.second:02}",
        )

        os.makedirs(self.output_dir, exist_ok=True)

    def _init_model(self, model_file: str | None = None):
        input_size = self.dataset.get_input_size()
        output_size = self.loss.output_layer_size() * len(self.dataset.outputs)
        model = self.model.init(input_size, output_size)

        if model_file is not None:
            state_dict = torch.load(model_file, map_location=self._device)
            model.load_state_dict(state_dict)

        return model.to(self._device)

    def setup_training(self):
        dataloader = self.dataset.prepare_for_training(self._device)

        if self.save_stats:
            # Save scaling parameters for inference
            with open(f"{self.output_dir}/scaler.pickle", "wb") as outfile:
                pickle.dump(dataloader.scaler, outfile)

        # Initialize loss function
        loss_function = QuantileLoss(self.loss, self._device)

        # Initialize the model
        model = self._init_model()

        # Initialize optimizer
        optimizer = self.optimizer.instantiate(model.parameters())
        return model, loss_function, optimizer, dataloader

    def setup_inference(self):
        # Load scaler to normalize dataset
        with open(f"{self._inference_dir}/scaler.pickle", "rb") as infile:
            scaler = pickle.load(infile)

        dataloader, metadata, distances = self.dataset.prepare_for_inference(
            scaler, self._device
        )

        # Initialize loss function
        loss_function = QuantileLoss(self.loss, self._device, mesh_mul=4)

        # Load best model
        model_file = f"{self._inference_dir}/best.pth"
        model = self._init_model(model_file)
        return model, loss_function, dataloader, metadata, distances

    def save(self, file: str, original: str):
        toml = tomllib.loads(original)

        if "seed" not in toml:
            toml["seed"] = self.seed

        file = f"{self.output_dir}/config.toml"
        with open(file, "wb") as outfile:
            tomli_w.dump(toml, outfile)

        print(f'Input config backed up to "{file}"')
