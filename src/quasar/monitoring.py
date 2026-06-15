# import pandas as pd


class Stats:
    columns = ["epoch", "time", "mean_train_loss", "mean_validation_loss"]

    def __init__(self):
        self.size = len(self.columns)
        self.data = []
        print("-" * 65)

    def update(
        self,
        epoch: int,
        time: float,
        train_loss: float,
        validation_loss: float,
    ):
        print(f"{epoch:5} {time=:7.4f} {train_loss=:9.6f} {validation_loss=:9.6f}")
        self.data.append((epoch, time, train_loss, validation_loss))

    def save(self, file: str):
        with open(file, "w") as out:
            out.write(",".join(self.columns) + "\n")

            for line in self.data:
                out.write(",".join((str(val) for val in line)) + "\n")
