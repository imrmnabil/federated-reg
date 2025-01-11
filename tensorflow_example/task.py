"""tensorflow-example: A Flower / TensorFlow app."""

import json
import os
from datetime import datetime
from pathlib import Path
from turtledemo.sorting_animate import partition

import keras
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from keras import layers

from flwr.common.typing import UserConfig
from sklearn.ensemble import RandomForestRegressor

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# def load_model(learning_rate: float = 0.001):
#     # Define a simple CNN for FashionMNIST and set Adam optimizer
#     model = keras.Sequential(
#         [
#             keras.Input(shape=(28, 28, 1)),
#             layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
#             layers.MaxPooling2D(pool_size=(2, 2)),
#             layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
#             layers.MaxPooling2D(pool_size=(2, 2)),
#             layers.Flatten(),
#             layers.Dropout(0.5),
#             layers.Dense(10, activation="softmax"),
#         ]
#     )
#     optimizer = keras.optimizers.Adam(learning_rate)
#     model.compile(
#         optimizer=optimizer,
#         loss="sparse_categorical_crossentropy",
#         metrics=["accuracy"],
#     )
#     return model


def load_model(learning_rate: float = 0.001):
    # Define a simple regression model
    model = keras.Sequential(
        [
            layers.Dense(64, activation="relu", input_shape=(5,)),  # Input layer for 5 features
            layers.Dense(32, activation="relu"),  # Hidden layer with 32 units
            layers.Dense(16, activation="relu"),  # Hidden layer with 16 units
            layers.Dense(1, activation="linear"),  # Output layer for regression
        ]
    )

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mean_squared_error",  # Loss function for regression
        metrics=["mean_absolute_error"],  # Additional metric for evaluation
    )

    return model


fds = None  # Cache FederatedDataset


def load_data(partition_id, num_partitions):
    """Load partition FashionMNIST data."""
    # Download and partition dataset
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="diffuse flows",
            alpha=1.0,
            seed=42,
        )
        fds = FederatedDataset(
            dataset="naabiil/power_consumption",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id, "train")
    partition.set_format("numpy")

    # Divide data on each node: 80% train, 20% test
    partition = partition.train_test_split(test_size=0.2)
    # x_train, y_train = partition["train"]["image"] / 255.0, partition["train"]["label"]
    # x_test, y_test = partition["test"]["image"] / 255.0, partition["test"]["label"]

    features = [
        "Temperature",
        "Humidity",
        "Wind Speed",
        "general diffuse flows",
        "diffuse flows"
    ]

    target = "Zone 1 Power Consumption"

    x_train = partition["train"].to_pandas()[features]
    y_train = partition["train"].to_pandas()[target]
    x_test = partition["test"].to_pandas()[features]
    y_test = partition["test"].to_pandas()[target]

    return x_train.values, y_train.values, x_test.values, y_test.values


def create_run_dir(config: UserConfig) -> tuple[Path, str]:
    """Create a directory where to save results from this run."""
    # Create output directory given current timestamp
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    # Save path is based on the current directory
    save_path = Path.cwd() / f"outputs/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=False)

    # Save run config as json
    with open(f"{save_path}/run_config.json", "w", encoding="utf-8") as fp:
        json.dump(config, fp)

    return save_path, run_dir
