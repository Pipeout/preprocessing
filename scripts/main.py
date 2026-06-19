import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


# error handlers
class ErrorLoadingDatasets(Exception):
    pass


class ErrorReadingConfigFile(Exception):
    pass


class MissingColumnInDataFrame(Exception):
    pass


class RequiredKeyNotUnique(Exception):
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

        return (
            required_columns_df_history,
            required_columns_df_active_df_deactive,
        )

    def load_datasets(
        self, CONFIG_PATH: Path
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str, str, str]:
        """
        Loads the datasets and separetes them
        """
        dfs = self.load_config(CONFIG_PATH)
        df_active = pd.read_csv(dfs["RAW_ACTIVE_DATASET"])
        df_deactive = pd.read_csv(dfs["RAW_DEACTIVE_DATASET"])
        df_history = pd.read_csv(dfs["RAW_HISTORY_DATASET"])
        period_range = dfs["PERIOD_RANGE"]
        preprocessed_active = dfs["PREPROCESSED_ACTIVE_DATASET"]
        preprocessed_deactive = dfs["PREPROCESSED_DEACTIVE_DATASET"]
        preprocessed_history = dfs["PREPROCESSED_HISTORY_DATASET"]

        return (
            df_active,
            df_deactive,
            df_history,
            period_range,
            preprocessed_active,
            preprocessed_deactive,
            preprocessed_history,
        )

    def run(
        self,
    ) -> tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, list, list, str, str, str, str
    ]:

        CONFIG_PATH = self.get_config_file("preprocessing.yaml")
        logging.info("\n\n[INFO]: Loading datasets...")

        (
            df_active,
            df_deactive,
            df_history,
            period_range,
            preprocessed_active,
            preprocessed_deactive,
            preprocessed_history,
        ) = self.load_datasets(CONFIG_PATH)

        (
            history_columns,
            non_history_columns,
        ) = self.load_required_columns(CONFIG_PATH)

        return (
            df_active,
            df_deactive,
            df_history,
            history_columns,
            non_history_columns,
            period_range,
            preprocessed_active,
            preprocessed_deactive,
            preprocessed_history,
        )


class Preprocessing:
    def __init__(
        self,
        df_active: pd.DataFrame,
        df_deactive: pd.DataFrame,
        df_history: pd.DataFrame,
        history_columns: list,
        non_history_columns: list,
        period_range: str,
        preprocessed_active: str,
        preprocessed_deactive: str,
        preprocessed_history: str,
    ):
        self.df_active = df_active
        self.df_deactive = df_deactive
        self.df_history = df_history
        self.history_columns = history_columns
        self.non_history_columns = non_history_columns
        self.period_range = period_range
        self.preprocessed_active = preprocessed_active
        self.preprocessed_deactive = preprocessed_deactive
        self.preprocessed_history = preprocessed_history

    def labeling_each_student(self) -> None:
        self.df_deactive["Class"] = "Inativo"
        self.df_active["Class"] = "Ativo"

    def concat_active_deactive(self) -> None:
        """
        Merge students to work with a single dataset
        """
        self.all_students = pd.concat([self.df_active, self.df_deactive], axis=0)

    def checking_presence_of_columns(self) -> None:

        missing_df_history = set(self.history_columns) - set(self.df_history.columns)
        missing_df_active = set(self.non_history_columns) - set(self.df_active.columns)
        missing_df_deactive = set(self.non_history_columns) - set(
            self.df_deactive.columns
        )

        if any([missing_df_active, missing_df_deactive, missing_df_history]):
            raise MissingColumnInDataFrame(
                f"Missing columns — active: {missing_df_active}, "
                f"deactive: {missing_df_deactive}, history: {missing_df_history}"
            )

    def drop_duplicates(self) -> None:
        self.df_history = self.df_history.drop_duplicates()

    def assert_unique(self, cols: list[str]) -> None:
        duplicated_active = self.df_active.duplicated(subset=cols, keep=False).any()
        duplicated_deactive = self.df_deactive.duplicated(subset=cols, keep=False).any()

        if any(
            [
                duplicated_active,
                duplicated_deactive,
            ]
        ):  # duplicated_history]):
            raise RequiredKeyNotUnique(f"Dataset is NOT unique on {cols}.")

    def standardize_column_text(self, col: str) -> None:
        self.df_history[col] = (
            self.df_history[col]
            .str.normalize("NFKD")
            .str.encode("ascii", errors="ignore")
            .str.decode("utf-8")
            .str.upper()
            .str.replace(r"\(OPTATIVA\)", "", regex=True)
            .str.strip()
        )

    def eliminating_duplicates_ap_ae(self) -> None:
        """
        AE and AP are duplicated in the dataset.
        Therefore it must be converted to one single row.
        """
        df = self.df_history

        ae_ap_pairs = df.groupby(["RGA_Anon", "Nome_Disciplina"]).filter(
            lambda g: {"AP", "AE"}.issubset(set(g["Situação"]))
        )

        df_adjust = df.copy()

        nota_ap = (
            ae_ap_pairs[ae_ap_pairs["Situação"] == "AP"]
            .groupby(["RGA_Anon", "Nome_Disciplina"])["Nota"]
            .max()
        )

        mask_ae = (df_adjust["Situação"] == "AE") & (
            df_adjust.set_index(["RGA_Anon", "Nome_Disciplina"]).index.isin(
                nota_ap.index
            )
        )

        df_adjust.loc[mask_ae, "Nota"] = (
            df_adjust.loc[mask_ae]
            .set_index(["RGA_Anon", "Nome_Disciplina"])
            .index.map(nota_ap)
        )

        mask_ap_to_drop = (df_adjust["Situação"] == "AP") & (
            df_adjust.set_index(["RGA_Anon", "Nome_Disciplina"]).index.isin(
                nota_ap.index
            )
        )

        self.df_history = df_adjust[~mask_ap_to_drop].copy()

    def setting_subject_failures(self) -> None:
        """
        Convert subject status into a binary outcome:
        - 1 = failure in a subject
        - 0 = non-failure
        Rows with 'MA' status are excluded.
        """
        df = self.df_history.copy()
        df = df[df["Situação"] != "MA"]
        failures = ["RMF", "RM", "RP", "RF"]

        df["Situação"] = np.where(df["Situação"].isin(failures), 1, 0)
        self.df_history = df

    def selecting_valid_period(self) -> None:
        """
        Anything before 2009.1 is inconsistent
        """
        self.df_history = self.df_history[self.df_history["AnoSem"] >= 2009.1].copy()

    def drop_columns(self, df_attr: str, cols: list[str]) -> None:
        """
        Generic column-dropper. df_attr is the attribute name as a string,
        e.g. 'df_history' or 'all_students'.
        """
        df = getattr(self, df_attr)
        setattr(self, df_attr, df.drop(columns=cols))

    def clean_student_demographics(self) -> None:
        """
        Type casting, date parsing, and scale fixes on the active/deactive
        student-level data.
        """

        self.df_active["Data Nascimento"] = pd.to_datetime(
            self.df_active["Data Nascimento"], format="%m/%d/%Y", errors="coerce"
        )
        self.df_active["Data ocorrência"] = pd.to_datetime(
            self.df_active["Data ocorrência"],
            format="%d/%m/%Y %H:%M:%S",
            errors="coerce",
        )
        self.df_deactive["Data Nascimento"] = pd.to_datetime(
            self.df_deactive["Data Nascimento"], format="%m/%d/%Y", errors="coerce"
        )
        self.df_deactive["Data ocorrência"] = pd.to_datetime(
            self.df_deactive["Data ocorrência"],
            format="%d/%m/%Y %H:%M:%S",
            errors="coerce",
        )

        self.df_deactive = self.df_deactive.rename(columns={"Período": "Periodo_Atual"})
        self.df_active = self.df_active.rename(columns={"Período": "Periodo_Atual"})

        self.df_deactive["Periodo_Atual"] = self.df_deactive["Periodo_Atual"] / 10
        self.df_active["Periodo_Atual"] = self.df_active["Periodo_Atual"] / 10

        self.df_deactive["Estrutura"] = self.df_deactive["Estrutura"].astype("int")
        self.df_active["Estrutura"] = self.df_active["Estrutura"].astype("int")

        self.df_deactive["Período ingresso"] = self.df_deactive["Período ingresso"] / 10
        self.df_active["Período ingresso"] = self.df_active["Período ingresso"] / 10

    def write_preprocessed_file(self) -> None:
        self.df_history.to_csv(self.preprocessed_history, index=False)
        self.df_deactive.to_csv(self.preprocessed_deactive, index=False)
        self.df_active.to_csv(self.preprocessed_active, index=False)

        pass

    def run(self) -> None:
        self.assert_unique(
            ["RGA_Anon"]
        )  # this is the main key. No student must have duplicated keys
        self.checking_presence_of_columns()
        self.drop_duplicates()
        self.standardize_column_text("Nome_Disciplina")
        self.eliminating_duplicates_ap_ae()
        self.setting_subject_failures()
        self.selecting_valid_period()
        self.labeling_each_student()
        self.clean_student_demographics()
        self.write_preprocessed_file()


if __name__ == "__main__":
    """
    This is the main function. It defines and processes datasets
    """
    logging.basicConfig(level=logging.INFO)

    logging.info("\n\nStarting application...")

    try:
        # loading the config file
        lc = LoadConfig()
        (
            df_active,
            df_deactive,
            df_history,
            history_columns,
            non_history_columns,
            period_range,
            preprocessed_active,
            preprocessed_deactive,
            preprocessed_history,
        ) = lc.run()
        # preprocessing data
        preprocessor = Preprocessing(
            df_active,
            df_deactive,
            df_history,
            history_columns,
            non_history_columns,
            period_range,
            preprocessed_active,
            preprocessed_deactive,
            preprocessed_history,
        )
        preprocessor.run()

    except RuntimeError as e:
        raise e
