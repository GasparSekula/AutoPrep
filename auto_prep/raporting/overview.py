import humanize
import numpy as np
import pandas as pd

from ..utils.logging_config import setup_logger
from ..utils.system import get_system_info
from .raport import Report

logger = setup_logger(__name__)


class OverviewRaport:
    def __init__(self):
        self.dataset_summary: dict = {}
        self.target_distibution: dict = {}
        self.missing_values: dict = {}
        self.features_details: dict = {}
        self.system_info: dict = {}

    def run(self, X: pd.DataFrame, y: pd.Series):
        """Performs dataset overview."""

        logger.start_operation("Overview.")

        try:
            self.system_info = get_system_info()
        except Exception as e:
            logger.error(f"Failed to gather system informations: {str(e)}")
            raise e

        try:
            numeric_features = X.select_dtypes(include=[np.number]).columns
            categorical_features = X.select_dtypes(exclude=[np.number]).columns

            # Basic statistics
            self.dataset_summary = {
                "Number of samples": len(X),
                "Number of features": len(X.columns),
                "Number of numerical features": len(numeric_features),
                "Number of categorical features": len(categorical_features),
            }

            value_counts = y.value_counts()
            normalized_counts = y.value_counts(normalize=True)
            self.target_distibution = list(
                zip(value_counts.index, value_counts.values, normalized_counts.values)
            )

            missing_value_counts = X.isnull().sum()
            normalized_missing_counts = X.isnull().sum() / len(X)
            self.missing_values = list(
                zip(
                    missing_value_counts.index,
                    missing_value_counts.values,
                    normalized_missing_counts.values,
                )
            )

            self.features_details = [
                (
                    feature,
                    "numerical" if feature in numeric_features else "categorical",
                    X[feature].dtype,
                    humanize.naturalsize(X[feature].memory_usage(deep=True)),
                )
                for feature in X.columns
            ]

            logger.debug(
                f"Found {len(numeric_features)} numeric and "
                f"{len(categorical_features)} categorical features"
            )

        except Exception as e:
            logger.error(f"Failed to gather overview statistics: {str(e)}")
            raise e

        logger.end_operation()

    def write_to_raport(self, raport: Report):
        """Writes overview section to a raport"""

        overview_section = raport.add_section("Overview")  # noqa: F841

        system_subsection = raport.add_subsection("System")  # noqa: F841
        raport.add_table(
            self.system_info,
            header=None,
            caption="System overview.",
        )

        dataset_subsection = raport.add_subsection("Dataset")  # noqa: F841
        raport.add_table(
            self.dataset_summary,
            header=None,
            caption="Dataset Summary.",
        )
        raport.add_table(
            self.target_distibution,
            caption="Target class distribution.",
            header=["class", "number of observations", "Percentage"],
        )
        raport.add_table(
            self.missing_values,
            caption="Missing values distribution.",
            header=["classgit", "number of observations", "Percentage"],
        )
        raport.add_table(
            self.features_details,
            header=["class", "type", "dtype", "space usage"],
            caption="Features description.",
        )

        return raport
