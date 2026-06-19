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
from numpy._core.multiarray import dtype
from typing_extensions import Dict


class ErrorLoadingDatasets(Exception):
    pass


class ErrorReadingConfigFile(Exception):
    pass


class MissingColumnInDataFrame(Exception):
    pass


class LoadConfig:
    @staticmethod
    def get_config_file(filename: str) -> Path:
        try:
            base_dir = Path(__file__).resolve().parent.parent
            path = base_dir / "configs" / filename
            return path
        except NameError:  # if it is a jupyter file
            return Path("/training-app/configs/training.yaml")

    @staticmethod
    def load_config(CONFIG_PATH: Path) -> Any:
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

    def load_required_columns(self, CONFIG_PATH: Path) -> tuple[list, list]:
        with open(CONFIG_PATH, "r") as f:
            full_config = yaml.safe_load(f)

        required_columns_df_history = full_config["REQUIRED_COLUMNS_DF_HISTORY"]
        required_columns_df_active_df_deactive = full_config[
            "REQUIRED_COLUMNS_DF_ACTIVE_AND_DF_DEACTIVE"
        ]
        return required_columns_df_history, required_columns_df_active_df_deactive

    def load_datasets(
        self, CONFIG_PATH: Path
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads the datasets and separetes them
        """
        dfs = self.load_config(CONFIG_PATH)
        df_active = pd.read_csv(dfs["ACTIVE_DATASET"])
        df_deactive = pd.read_csv(dfs["EVADED_DATASET"])
        df_history = pd.read_csv(dfs["HISTORY_DATASET"])

        return df_active, df_deactive, df_history

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list, list]:

        CONFIG_PATH = self.get_config_file("preprocessing.yaml")
        logging.info("\n\n[INFO]: Loading datasets...")

        df_active, df_deactive, df_history = self.load_datasets(CONFIG_PATH)

        history_columns, non_history_columns = self.load_required_columns(CONFIG_PATH)

        return df_active, df_deactive, df_history, history_columns, non_history_columns


class Preprocessing:
    def __init__(
        self,
        df_active: pd.DataFrame,
        df_deactive: pd.DataFrame,
        df_history: pd.DataFrame,
        history_columns: list,
        non_history_columns: list,
    ):
        self.df_active = df_active
        self.df_deactive = df_deactive
        self.df_history = df_history
        self.history_columns = history_columns
        self.non_history_columns = non_history_columns

    @staticmethod
    def seeing_columns(df: pd.DataFrame) -> None:
        print(df.columns)

    def checking_presence_of_columns(self) -> None:

        missing_df_history = set(self.history_columns) - set(self.df_history.columns)
        missing_df_active = set(self.non_history_columns) - set(self.df_active.columns)
        missing_df_deactive = set(self.non_history_columns) - set(
            self.df_deactive.columns
        )

        if any([missing_df_active, missing_df_deactive, missing_df_history]):
            raise MissingColumnInDataFrame(
                "The datasets have missing or extra columns. Check for the config file and the datasets load"
            )

    def run(self) -> None:
        # Checking the presence of all columns
        self.checking_presence_of_columns()


if __name__ == "__main__":
    """
    This is the main function. It defines and processes datasets
    """
    logging.basicConfig(level=logging.INFO)

    logging.info("\n\n[INFO]: Starting application...")

    try:
        lc = LoadConfig()
        df_active, df_deactive, df_history, history_columns, non_history_columns = (
            lc.run()
        )

        preprocessor = Preprocessing(
            df_active, df_deactive, df_history, history_columns, non_history_columns
        )
        preprocessor.run()

    except RuntimeError as e:
        raise e
