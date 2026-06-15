import os
import warnings
from typing import Annotated as Float

import msgspec
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils import data

# suffixes added to feature names when encoding them
SINCOS_SUFFIX = ["_sin", "_cos"]
EARTH_RADIUS = 6371.137


def date_to_year(date: str) -> float:
    # date is a string with format YYYYMMDD
    year = date[:4]
    return float(year)


def calculate_haversine(
    target_lat: Float[np.ndarray, "N"],
    target_lon: Float[np.ndarray, "N"],
    others_lat: Float[np.ndarray, "neighbors N"],
    others_lon: Float[np.ndarray, "neighbors N"],
) -> np.ndarray:
    target_lat = np.radians(target_lat).reshape((-1, 1))
    target_lon = np.radians(target_lon).reshape((-1, 1))

    neigh_lats = np.radians(others_lat)
    neigh_lons = np.radians(others_lon)

    delta_lon = neigh_lons - target_lon
    delta_lat = neigh_lats - target_lat

    haversine = (
        np.sin(0.5 * delta_lat) ** 2
        + np.cos(target_lat) * np.cos(neigh_lats) * np.sin(delta_lon * 0.5) ** 2
    )

    distances = 2 * EARTH_RADIUS * np.arcsin(np.sqrt(haversine))
    return distances


def encode_cyclical(
    feature: Float[np.ndarray, "rows cols"], period: float
) -> Float[np.ndarray, "rows 2*cols"]:
    x = np.sin(2 * np.pi * feature / period)
    y = np.cos(2 * np.pi * feature / period)

    if feature.ndim == 1:
        # if the input feature is a row vector,
        # we need to vstack and transpose since we want (rows, cols)
        return np.vstack((x, y)).T

    return np.hstack((x, y))


# Wrap location encoder
def encode_positions(
    coords: Float[np.ndarray, "rows cols"],
) -> Float[np.ndarray, "rows 2*cols"]:
    # place (lat, lon) coordinates in the [-pi, pi) range
    coords = np.deg2rad(coords)
    return encode_cyclical(coords, 2 * np.pi)


def get_all_distances(test_set: pd.DataFrame):
    target = ("target_lat", "target_lon")

    n_neighs = 15
    latlons = [(f"n{i:02}_lat", f"n{i:02}_lon") for i in range(1, n_neighs + 1)]

    config = DistanceConfig(target=target, others=latlons)
    distances = config.calculate_dists(test_set)
    return distances


def extract_inference_metadata(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["target_lat", "target_lon", "target_elev"]
    new = df.copy()
    new["location"] = df.groupby(cols).ngroup()

    new = new.rename(
        columns={
            "target_lat": "lat",
            "target_lon": "lon",
            "target_elev": "altitude",
            "target_value": "obs",
        }
    )

    out_cols = ["date", "hour", "location", "lat", "lon", "altitude", "obs"]
    return new[out_cols]


class DistanceConfig(msgspec.Struct):
    # (lat, lon) column name for the target
    target: tuple[str, str]

    # (lat, lon) column names for the points for which to calculate the distances
    others: list[tuple[str, str]]

    # column names
    _cols: list[str] = []

    def __post_init__(self):
        self._cols = [f"n{i:02}_dist" for i in range(1, len(self.others) + 1)]

    def get_feat_number(self):
        return len(self.others)

    def calculate_dists(self, df: pd.DataFrame) -> pd.DataFrame:
        target_lat = df[[self.target[0]]].values
        target_lon = df[[self.target[1]]].values

        lats, lons = [], []
        for lat, lon in self.others:
            lats.append(lat)
            lons.append(lon)

        lats = df[lats].values
        lons = df[lons].values
        distances = calculate_haversine(target_lat, target_lon, lats, lons)
        return pd.DataFrame(distances, columns=self._cols)


class DatasetConfig(msgspec.Struct):
    ### File options
    # List of the train files
    train_files: list[str] = []

    # Optional list of files used for validation during training
    validation_files: list[str] = []

    # List of the test files used during inference
    test_files: list[str] = []

    # size of the train set if 'validation_files' is empty
    train_size: float = 0.8

    # Whether the train/set split should be random or in order
    random_split: bool = False

    # directory where to find dataset files
    directory: str = "."

    ### Dataloader options
    # Number of samples passed to the model at a time
    batch_size: int = 1

    # drop the last batch if its size is smaller than `batch_size`
    drop_last: bool = True

    # Shuffle the dataset on each epoch
    shuffle: bool = False

    # Pin memory on GPU
    # NOTE: only useful when loading batch by batch to GPU
    pin_memory: bool = False

    # Chunksize for loading dataset
    chunksize: int = 200_000

    # Number of workers for dataloader
    # NOTE: only useful when loading batch by batch to GPU
    num_workers: int = 0

    ### Dataset variables
    inputs: list[str] = []
    outputs: list[str] = []
    time_encoded_inputs: dict[str, float] = {}
    position_encoded_inputs: list[str] = []

    # Calculate neighbor distances and use them instead of neighbor positions
    distances: DistanceConfig | None = None

    def __post_init__(self):
        # Update dataset files by prepending input directory
        self.train_files = [os.path.join(self.directory, f) for f in self.train_files]
        self.validation_files = [
            os.path.join(self.directory, f) for f in self.validation_files
        ]
        self.test_files = [os.path.join(self.directory, f) for f in self.test_files]

    @staticmethod
    def _load_csvs(files: list[str]) -> pd.DataFrame:
        out = []
        for file in files:
            # Keep date as string so later we can extract year
            # and/or return it as is for inference
            df = pd.read_csv(file, dtype=np.float32, converters={"date": str})
            out.append(df)
        return pd.concat(out, ignore_index=True)

    def _create_dataloader(
        self, dataset: data.Dataset, shuffle: bool = False
    ) -> data.DataLoader:
        return data.DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def get_input_size(self) -> int:
        # number of resulting inputs if encoded
        embedding_size = 2

        input_size = len(self.inputs) + embedding_size * (
            len(self.position_encoded_inputs) + len(self.time_encoded_inputs)
        )

        if self.distances is not None:
            input_size += self.distances.get_feat_number()

        return input_size

    def prepare_for_training(self, device: torch.device) -> "DataLoader":
        if not self.train_files or not self.validation_files:
            raise ValueError(
                "training requires both 'dataset.train_files' and 'dataset.validation_files' to be defined"
            )

        dataloader = DataLoader(self, device)
        return dataloader

    def prepare_for_inference(
        self, scaler: StandardScaler, device: torch.device
    ) -> tuple["DataLoader", pd.DataFrame, pd.DataFrame]:
        if not self.test_files:
            raise ValueError("inference requires 'dataset.test_files' to be defined")

        test_set = self._load_csvs(self.test_files)
        metadata = extract_inference_metadata(test_set)
        distances = get_all_distances(test_set)
        del test_set

        self.drop_last = False
        dataloader = DataLoader(self, device, scaler)
        return dataloader, metadata, distances


class StreamDataset(data.IterableDataset):
    def __init__(
        self,
        files: list[str],
        device: torch.device,
        config: DatasetConfig,
        scaler: StandardScaler | None = None,
    ):
        # TODO: maybe extract these fields to a separate config class
        self.chunksize = config.chunksize
        self.features = config.inputs
        self.targets = config.outputs
        self.distances = config.distances
        self.time_encoded_features = config.time_encoded_inputs
        self.position_encoded_features = config.position_encoded_inputs

        self.device = device
        self.files = files

        if scaler is None:
            scaler = self._scaler()

        self.scaler = scaler

    # Precompute the scaling statistics over the whole dataset.
    # Returns the scaler which will then be used to scale data online during training and inference
    # TODO: add a way to reload a previously computed scaler instead of creating a new one every time
    def _scaler(self) -> StandardScaler:
        print("Scaling input features... ", end="")
        scaler = StandardScaler()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for file in self.files:
                for chunk in pd.read_csv(
                    file,
                    chunksize=self.chunksize,
                    dtype="float32",
                    converters={"date": str},
                ):
                    x_feature = self._extract_features(chunk)
                    scaler.partial_fit(x_feature)
        print("Done!")
        return scaler

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: could be more general
        if (feat := "date") in self.features:
            # Convert timestamp to year only
            df[feat] = df[feat].apply(date_to_year)

        dists = None
        if self.distances is not None:
            dists = self.distances.calculate_dists(df).reset_index(drop=True)

        encoded = self._encode_features(df).reset_index(drop=True)
        features = pd.concat(
            (df[self.features].reset_index(drop=True), encoded, dists), axis=1
        )
        return features

    def _encode_features(self, df: pd.DataFrame):
        out = []
        cols = []
        for feat, period in self.time_encoded_features.items():
            enc = encode_cyclical(df[feat].to_numpy(), period)
            cols += [feat + suff for suff in SINCOS_SUFFIX]
            out.append(enc)

        for feat in self.position_encoded_features:
            enc = encode_positions(df[feat].to_numpy())
            cols += [feat + suff for suff in SINCOS_SUFFIX]
            out.append(enc)

        out = np.hstack(out)
        return pd.DataFrame(out, columns=cols)

    def _extract_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        targets = df[self.targets]
        return targets

    def _preprocessing(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        x = self._extract_features(df)
        y = self._extract_targets(df)
        x[x.columns] = self.scaler.transform(x)
        return x, y

    def __iter__(self):
        for file in self.files:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for chunk in pd.read_csv(
                    file,
                    chunksize=self.chunksize,
                    dtype="float32",
                    converters={"date": str},
                ):
                    x = self._extract_features(chunk)
                    y = self._extract_targets(chunk)
                    x[x.columns] = self.scaler.transform(x)

                    # TODO: maybe use bfloat16 instead of float32?
                    features = torch.from_numpy(x.values).float().to(self.device)
                    targets = torch.from_numpy(y.values).float().to(self.device)

                    for i in range(len(chunk)):
                        yield features[i], targets[i]


class DataLoader:
    def __init__(
        self,
        config: DatasetConfig,
        device: torch.device,
        scaler: StandardScaler | None = None,
    ):
        stream_train = StreamDataset(config.train_files, device, config, scaler)

        if scaler is None:
            scaler = stream_train.scaler

        stream_val = StreamDataset(config.validation_files, device, config, scaler)
        stream_test = StreamDataset(config.test_files, device, config, scaler)

        self.scaler = scaler
        self.train_set = config._create_dataloader(stream_train)
        self.validation_set = config._create_dataloader(stream_val)
        self.test_set = config._create_dataloader(stream_test)
