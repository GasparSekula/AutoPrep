from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..utils.config import config
from ..utils.logging_config import setup_logger
from ..utils.other import save_chart

logger = setup_logger(__name__)


class EdaVisualizer:
    """
    Contains methods that generate basic eda charts. Will
    be fed with entire original dataset. All methods
    will be called in order defined in :obj:`order`. Each method that would
    be called should return a tuple of (path_to_chart, chart title for latex) -
    if there is no need for chart generation should return ("", "").
    Charts should be saved via :obj:`save_chart`.
    """

    order = [
        "target_distribution_chart",
        "missing_values_chart",
    ]

    # for classification
    @staticmethod
    def target_distribution_chart(
        X: pd.DataFrame,  # noqa: F841
        y: pd.Series,
    ) -> Tuple[str, str]:
        """
        Generates a plot to visualize the distribution of the target variable.
        """
        logger.start_operation("Target distribution visualization.")
        try:
            y_df = y.to_frame(name="target")
            plt.figure(figsize=(10, 6))
            sns.countplot(
                data=y_df,
                x="target",
                palette=config.chart_settings["palette"],
                hue="target",
            )
            # add percent labels
            total = len(y)
            for p in plt.gca().patches:
                height = p.get_height()
                plt.gca().text(
                    p.get_x() + p.get_width() / 2,
                    height + 3,
                    f"{height / total:.2%}",
                    ha="center",
                )

            plt.title(f"Distribution of {y.name}")
            path = save_chart(name="target_distribution.png")
            return path, "Target distribution."
        except Exception as e:
            logger.error(f"Failed to generate target distribution plot: {str(e)}")
            raise e
        finally:
            logger.end_operation()

    @staticmethod
    def missing_values_chart(
        X: pd.DataFrame,
        y: pd.Series,  # noqa: F841
    ) -> Tuple[str, str]:
        """
        Generates a plot to visualize the percentage of missing values for each
        feature in the given DataFrame.
        """
        logger.start_operation("Missing values visualizations.")

        try:
            plt.figure(figsize=(10, 6))
            missing = X.isnull().sum() / len(X) * 100
            missing = missing[missing > 0].sort_values(ascending=False)

            if missing.empty:
                logger.debug("No missing values found in the dataset.")
                return "", ""
            logger.debug(
                f"Will create missing values chart for {list(missing.index)} columns."
            )

            sns.barplot(
                x=missing.index,
                y=missing.values,
                palette=config.chart_settings["palette"],
            )
            plt.xticks(rotation=45)
            plt.title("Percentage of Missing Values by Feature")
            path = save_chart(name="missing_values.png")

            return path, "Missing values."
        except Exception as e:
            logger.error(f"Failed to generate missing values plot: {str(e)}")
            raise e
        finally:
            logger.end_operation()
