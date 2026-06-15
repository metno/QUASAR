from argparse import ArgumentParser, Namespace, RawTextHelpFormatter

import quasar.inference as inference
import quasar.training as training
import quasar.tuning as tuning
from quasar.config import Config


class Args(Namespace):
    config: str
    action: str


def parse_args() -> Args:
    ap = ArgumentParser(formatter_class=RawTextHelpFormatter)
    ap.add_argument(
        "action",
        choices=["training", "inference", "optimize"],
        help="What action to perform\n\
  training: train a ML model based on the parameters provided in the config\n\
  inference: output a verif file based on the trained model and the test set specified in the config\n\
  optimize: optimize model and training parameters with optuna",
    )
    ap.add_argument(
        "config",
        metavar="config.toml",
        help="Config file in toml format. For inference, this should be located in the same directory the trained model params are.",
    )

    args = ap.parse_args(namespace=Args())
    return args


def main():
    args = parse_args()

    match args.action:
        case "training":
            config = Config.new(args.config, inference=False)
            model, loss_function, optimizer, dataloader = config.setup_training()
            training.train(model, loss_function, optimizer, dataloader, config)

        case "inference":
            config = Config.new(args.config, inference=True)
            model, loss_function, dataloader, metadata, distances = (
                config.setup_inference()
            )

            outfile = f"{config.output_dir}/distances.csv"
            distances.to_csv(outfile, index=False)
            print(f'Distances saved to "{outfile}"')

            inference.infer(model, loss_function, dataloader, metadata, config)

        case "optimize":
            config = Config.new(args.config, inference=False)
            config.save_stats = False
            tuning.tune(config)
