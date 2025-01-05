from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..utils.config import config
from ..utils.logging_config import setup_logger
from ..utils.other import save_chart

logger = setup_logger(__name__)


class NumericalVisualizer:
    """
    Contains methods that generate eda charts for numerical data. Will
    be fed with just numerical columns from original dataset. All methods
    will be called in order defined in :obj:`order`. Each method that would
    be called should return a tuple of (path_to_chart, chart title for latex) -
    if there is no need for chart generation should return ("", "").
    Charts should be saved via :obj:`save_chart`.
    """

    order = [
        "numerical_distribution_chart",
        "correlation_heatmap_chart",
        "numerical_features_boxplot_chart",
    ]

    @staticmethod
    def numerical_distribution_chart(
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Tuple[str, str]:
        """
        Generates a plot to visualize the distribution of numerical features.
        """
        logger.start_operation("Numerical distribution visualizations.")

        try:
            df = pd.concat([X, y], axis=1)

            numerical_columns = df.select_dtypes(include="number").columns.tolist()
            if numerical_columns == []:
                logger.debug("No numerical features found in the dataset.")
                return "", ""
            logger.debug(
                "Will create numerical distribution visualisations"
                f"for {numerical_columns} columns."
            )

            num_columns = len(numerical_columns)
            num_rows = (num_columns + 1) // 2

            sns.set_theme(
                style=config.chart_settings["theme"],
                palette=config.chart_settings["palette"],
            )

            _, axes = plt.subplots(
                num_rows,
                2,
                figsize=(
                    config.chart_settings["plot_width"],
                    config.chart_settings["plot_height_per_row"] * num_rows,
                ),
            )
            axes = axes.flatten()

            for i, column in enumerate(numerical_columns):
                sns.histplot(data=df, x=column, ax=axes[i])
                axes[i].set_title(
                    f"Distribution of {column}",
                    fontsize=config.chart_settings["title_fontsize"],
                    fontweight=config.chart_settings["title_fontweight"],
                )
                axes[i].set_xlabel(
                    column, fontsize=config.chart_settings["xlabel_fontsize"]
                )
                axes[i].set_ylabel(
                    "Count", fontsize=config.chart_settings["ylabel_fontsize"]
                )
                axes[i].tick_params(
                    axis="x", rotation=config.chart_settings["tick_label_rotation"]
                )
                axes[i].tick_params(
                    axis="both",
                    which="major",
                    labelsize=config.chart_settings["xlabel_fontsize"],
                )

            axes[-1].axis("off")

            plt.tight_layout()
            path = save_chart(name="numerical_distribution.png")
            return path, "Numerical distribution."
        except Exception as e:
            logger.error(f"Failed to generate numerical distribution plot: {str(e)}")
            raise e
        finally:
            logger.end_operation()

    @staticmethod
    def correlation_heatmap_chart(
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Tuple[str, str]:
        """
        Generates a plot to visualize the correlation between features.
        """
        logger.start_operation("Correlation heatmap visualization.")

        try:
            df = pd.concat([X, y], axis=1)

            numerical_columns = df.select_dtypes(include="number").columns.tolist()
            if numerical_columns == []:
                logger.debug("No numerical features found in the dataset.")
                return "", ""
            logger.debug(f"Will create heatmap for {numerical_columns} columns.")

            plt.figure(
                figsize=(
                    config.chart_settings["plot_width"],
                    config.chart_settings["plot_height_per_row"] * 2.5,
                )
            )
            sns.heatmap(
                df[numerical_columns].corr(numeric_only=False),
                annot=True,
                cmap=config.chart_settings.get("heatmap_cmap", "coolwarm"),
                fmt=config.chart_settings.get("heatmap_fmt", ".2f"),
            )
            plt.title(
                "Correlation Heatmap",
                fontsize=config.chart_settings["title_fontsize"],
                fontweight=config.chart_settings["title_fontweight"],
            )
            path = save_chart(name="correlation_heatmap.png")

            return path, "Correlation heatmap."
        except Exception as e:
            logger.error(f"Failed to generate correlation heatmap plot: {str(e)}")
            raise e
        finally:
            logger.end_operation()

    @staticmethod
    def numerical_features_boxplot_chart(
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Tuple[str, str]:
        """
        Generates a boxplot for numerical features.
        """
        logger.start_operation("Boxplot visualization for numerical features.")

        try:
            df = pd.concat([X, y], axis=1)

            numerical_columns = df.select_dtypes(include="number").columns.tolist()
            if not numerical_columns:
                logger.debug("No numerical features found in the dataset.")
                return "", ""

            logger.debug(f"Will create boxplot for {numerical_columns} columns.")
            num_columns = len(numerical_columns)
            num_rows = (num_columns + 1) // 2

            _, axes = plt.subplots(
                num_rows,
                2,
                figsize=(
                    config.chart_settings["plot_width"],
                    config.chart_settings["plot_height_per_row"] * num_rows,
                ),
            )
            axes = axes.flatten()

            try:
                for i, column in enumerate(numerical_columns):
                    sns.boxplot(
                        data=df,
                        x=column,
                        ax=axes[i],
                        palette=config.chart_settings["palette"],
                    )
                    axes[i].set_title(
                        f"Boxplot of {column}",
                        fontsize=config.chart_settings["title_fontsize"],
                        fontweight=config.chart_settings["title_fontweight"],
                    )
                    axes[i].set_xlabel(
                        column, fontsize=config.chart_settings["xlabel_fontsize"]
                    )
                    axes[i].set_ylabel(
                        "Value", fontsize=config.chart_settings["ylabel_fontsize"]
                    )
                    axes[i].tick_params(
                        axis="x", rotation=config.chart_settings["tick_label_rotation"]
                    )
                    axes[i].tick_params(
                        axis="both",
                        which="major",
                        labelsize=config.chart_settings["xlabel_fontsize"],
                    )
            except Exception as e:
                raise Exception(f"Error in column {column}") from e

            for j in range(i + 1, len(axes)):
                axes[j].axis("off")

            plt.tight_layout()
            path = save_chart(name="boxplot.png")
            return path, "Boxplot."
        except Exception as e:
            logger.error(f"Failed to generate boxplot visualisations plot: {str(e)}")
            raise e
        finally:
            logger.end_operation()
