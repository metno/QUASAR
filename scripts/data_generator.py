#!/usr/bin/env python3

"""
Generate synthetic spatial datasets using a Gaussian Random Field (GRF).

The script:

1. Creates N random neighbor locations around a target point located at (0, 0).
2. Computes an IDI-like metric (X_IDI) that quantifies how concentrated the
   neighbors are around the target.
3. Keeps only realizations whose X_IDI is below a user-defined threshold
   (currently hardcoded to 0.4).
4. Generates a Gaussian Random Field over the target and neighbor locations
   using a squared-exponential covariance function.
5. Stores the generated values and coordinates in CSV format.
6. Optionally produces a histogram of all accepted X_IDI values.

The generated CSV contains:
    - Target metadata
    - Target field value
    - Neighbor field values
    - Neighbor coordinates
    - Neighbor elevations

Each row corresponds to one accepted realization.
"""

import argparse
import csv
from typing import TextIO

import matplotlib.pyplot as plt
import numpy as np

CENTER = np.array([0.0, 0.0])
DATE = 20260101
HOUR = 13
DOY = "01"
ELEV = 0.0


def generate_points(center, N):
    """
    Generate N random points inside a random-radius disk.

    Parameters
    ----------
    center : ndarray
        Coordinates of the target point.
    N : int
        Number of neighbors.

    Returns
    -------
    points : ndarray
        Neighbor coordinates of shape (N, 2).
    r : ndarray
        Distance of each neighbor from the target.
    """
    radius = np.random.uniform(0.65, 1.4)

    theta = np.random.uniform(0, 2 * np.pi, N)
    r = radius * np.sqrt(np.random.uniform(0.0, 1, N))

    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)

    return np.column_stack([x, y]), r


def gaussian_field(points, D):
    """
    Generate a Gaussian Random Field at the specified locations.

    Covariance model:

        C_ij = exp(-r_ij^2 / (2 D^2))

    Parameters
    ----------
    points : ndarray
        Coordinates where the field is sampled.
    D : float
        Correlation length.

    Returns
    -------
    ndarray
        Random field values.
    """
    dist2 = np.sum(
        (points[:, None, :] - points[None, :, :]) ** 2,
        axis=2,
    )

    cov = np.exp(-dist2 / (2 * D**2))

    return np.random.multivariate_normal(
        mean=np.zeros(len(points)),
        cov=cov,
    )


def compute_D_IDI(N):
    """
    Correlation scale used in the IDI metric.
    """
    D = 1.0
    return D / np.sqrt(N)


def compute_X_IDI(r, N):
    """
    Compute the IDI metric from neighbor distances.
    """
    D_IDI = compute_D_IDI(N)
    return np.mean(np.exp(-(r**2) / (2 * D_IDI**2)))


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Generate synthetic spatial realizations using a Gaussian Random Field (GRF).

For each realization:

  1. Randomly generate N neighbors around a target point.
  2. Compute an IDI metric describing neighbor concentration.
  3. Accept only realizations with X_IDI < 0.4.
  4. Generate correlated Gaussian values using a GRF.
  5. Save the realization to a CSV file.

Output CSV structure:

  date,hour,day_of_the_year,
  target_lon,target_lat,target_elev,target_value,
  n01_value,...,nNN_value,
  n01_lon,...,nNN_lon,
  n01_lat,...,nNN_lat,
  n01_elev,...,nNN_elev

Examples
--------
Generate 1000 realizations with 20 neighbors:

    python data_generator.py \\
        --N 20 \\
        --n-realizations 1000 \\
        --output-file data.csv

Generate histogram as well:

    python data_generator.py \\
        --N 20 \\
        --n-realizations 1000 \\
        --output-file data.csv \\
        --histo \\
        --hist-output idi_hist.png
""",
    )

    parser.add_argument(
        "--N",
        type=int,
        default=15,
        help="Number of neighbor points per realization (default: %(default)s).",
    )

    parser.add_argument(
        "--n-realizations",
        type=int,
        default=813,
        help="Number of accepted realizations to generate (default: %(default)s).",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default="test.csv",
        help="Output CSV file path.",
    )

    parser.add_argument(
        "--hist-output",
        type=str,
        default="idi_histogram.png",
        help="Output histogram image path.",
    )

    parser.add_argument(
        "--histo",
        action="store_true",
        help="Generate and save a histogram of accepted X_IDI values.",
    )

    return parser.parse_args()


def get_header(n: int) -> list[str]:
    header = [
        "date",
        "hour",
        "day_of_the_year",
        "target_lon",
        "target_lat",
        "target_elev",
        "target_value",
    ]

    for i in range(1, n + 1):
        header.append(f"n{i:02d}_value")

    for i in range(1, n + 1):
        header.append(f"n{i:02d}_lon")

    for i in range(1, n + 1):
        header.append(f"n{i:02d}_lat")

    for i in range(1, n + 1):
        header.append(f"n{i:02d}_elev")

    return header


def write_csv(N: int, n_realizations: int, file: TextIO):
    writer = csv.writer(file)
    header = get_header(N)

    X_IDI_values = []
    D_field = compute_D_IDI(N)
    writer.writerow(header)

    realization_count = 0
    while realization_count < n_realizations:
        neighbors, r = generate_points(CENTER, N)

        Xidi = compute_X_IDI(r, N)

        if Xidi < 0.4:
            X_IDI_values.append(Xidi)

            all_points = np.vstack([CENTER, neighbors])

            field_values = gaussian_field(all_points, D_field)

            target_value = field_values[0]
            neighbor_values = field_values[1:]

            order = np.argsort(r)

            neighbors = neighbors[order]
            neighbor_values = neighbor_values[order]

            row = [DATE, HOUR, DOY, CENTER[0], CENTER[1], ELEV, target_value]

            row.extend(neighbor_values.tolist())
            row.extend(neighbors[:, 0].tolist())
            row.extend(neighbors[:, 1].tolist())
            row.extend([ELEV] * N)

            writer.writerow(row)

            realization_count += 1

    return X_IDI_values


def plot_histogram(X_IDI_values: np.ndarray, output: str):
    plt.figure(figsize=(7, 5))
    plt.hist(
        X_IDI_values,
        bins=40,
        density=True,
        alpha=0.7,
        edgecolor="k",
    )

    plt.xlabel("X_IDI")
    plt.ylabel("Probability density")

    plt.title(
        f"Histogram of IDI over random point generations\n"
        f"Mean IDI = {X_IDI_values.mean():.4f}\n"
        f"STD IDI = {X_IDI_values.std():.4f}"
    )

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()

    print(f"Histogram saved to: {output}")
    print(f"Mean IDI = {X_IDI_values.mean():.4f}")
    print(f"Std  IDI = {X_IDI_values.std():.4f}")


def main():
    args = parse_args()

    with open(args.output_file, "w", newline="") as f:
        X_IDI_values = write_csv(args.N, args.n_realizations, f)

    if args.histo:
        X_IDI_values = np.array(X_IDI_values)
        plot_histogram(X_IDI_values, args.hist_output)


if __name__ == "__main__":
    main()
