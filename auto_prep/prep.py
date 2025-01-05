from typing import Callable

import pandas as pd
from sklearn.model_selection import train_test_split

from .raporting.eda import EdaRaport
from .raporting.overview import OverviewRaport
from .raporting.raport import raport
from .utils.config import config
from .utils.logging_config import setup_logger

logger = setup_logger(__name__)


format_column_name: Callable = lambda x: x.replace(".", "__")


class AutoPrep:
    """Main pipeline orchestrating the entire preprocessing process.

    This class handles the complete workflow from data preprocessing to
    report generation.
    """

    def __init__(self):
        self.overview_raport: OverviewRaport = OverviewRaport()
        self.eda_raport: EdaRaport = EdaRaport()

    def run(self, data: pd.DataFrame, target_column: str):
        """Run the complete pipeline on the provided dataset.

        Args:
            data (pd.DataFrame): Input dataset to process.
            target_column (str): Name of the target variable column.
        """
        logger.info(f"Starting pipeline run with target column: {target_column}")
        logger.debug(f"Input data shape: {data.shape}")

        # "." in names leads to os problems
        for col in data.columns:
            if "." in col:
                logger.warning(
                    f"Column '{col}' will be renamed to {format_column_name(col)}"
                    "renamed due to '.' in it's name, which leads to os problems."
                )
        data = data.rename(columns=lambda x: x.replace(".", "__"))

        self._run(data, target_column)

        self._generate_report()

    def _run(self, data: pd.DataFrame, target_column: str):
        """
        Performs all neccessary computations.

        Args:
            data (pd.DataFrame): Input dataset to process.
            target_column (str): Name of the target variable column.
        """

        logger.start_operation("Calculations.")

        try:
            """
            Split data
            """
            logger.start_operation("Spliting data.")

            # Split features and target
            X = data.drop(columns=[target_column])
            y = data[target_column]

            # Split data
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, train_size=config.train_size, random_state=config.random_state
            )
            X_valid, X_test, y_valid, y_test = train_test_split(
                X_temp,
                y_temp,
                test_size=config.test_size / (config.test_size + config.valid_size),
                random_state=config.random_state,
            )  # noqa: F841
            logger.end_operation()

            """
            Overview
            """
            self.overview_raport.run(X_train, y_train)

            """
            Eda
            """
            self.eda_raport.run(X_train, y_train)

        except Exception as e:
            logger.error(f"Pipeline run failed: {e}")
            raise

        logger.end_operation()

    def _generate_report(self):
        """Generates and saves raport."""

        logger.start_operation("Generate raport.")
        raport.add_header()

        """
        Overview
        """
        self.overview_raport.write_to_raport(raport)

        """
        Eda
        """
        self.eda_raport.write_to_raport(raport)

        raport.generate()
        logger.end_operation()
