import logging
import os
import re
import time
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from featureEngineering.scripts.selection import CONFIG_PATH


class ErrorLoadingDatasets(Exception):
    pass


class ErrorReadingConfigFile(Exception):
    pass


def get_config_file() -> Path:
    try:
        base_dir = Path(__file__).resolve().parent.parent
        path = base_dir / "configs" / "cleaning.yaml"
        return path
    except NameError:  # if it is a jupyter file
        return Path("/training-app/configs/training.yaml")


def load_config(CONFIG_PATH):
    """
    Selects the current dataset's config file we are interest in.
    """
    with open(CONFIG_PATH, "r") as f:
        full_config = yaml.safe_load(f)

    try:
        current_dataset = full_config["CURRENT_DATASET"]
        logging.info(f"\nloading current dataset: {current_dataset}")
        if current_dataset not in full_config["DATASETS"]:
            raise ValueError(f"\nDataset {current_dataset} not found!")

        return full_config["DATASETS"][current_dataset]

    except Exception:
        raise ErrorReadingConfigFile(
            "Error while reading config file. Check its path and correctness"
        )


def load_datasets(CONFIG_PATH: Path) -> pd.DataFrame:
    """
    Loads the datasets and separetes them
    """

    dfs = load_config(CONFIG_PATH)

    df_active = pd.read_csv(dfs["ACTIVE_DATASET"])
    df_deactive = pd.read_csv(dfs["EVADED_DATASET"])
    df_history = pd.read_csv(dfs["HISTORY_DATASET"])

    return df_active, df_deactive, df_history


if __name__ == "__main__":
    """
    This is the main function. It defines and processes datasets
    """
    logging.basicConfig(level=logging.INFO)

    logging.info("\n\n[INFO]: Starting application...")

    try:
        CONFIG_PATH = get_config_file()
        logging.info("\n\n[INFO]: Loading datasets...")
        (
            df_active,
            df_deactive,
            df_history,
        ) = load_datasets(CONFIG_PATH)
    except ErrorLoadingDatasets:
        logging.error(
            "Error While loading datasets. Check their naming or if they exist."
        )
