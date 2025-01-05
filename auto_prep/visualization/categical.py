from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..utils.config import config
from ..utils.logging_config import setup_logger
from ..utils.other import save_chart

logger = setup_logger(__name__)


class CategoricalVisualizer:
    """
    Contains methods that generate eda charts for categorical data. Will
    be fed with just categorical columns from original dataset. All methods
    will be called in order defined in :obj:`order`. Each method that would
    be called should return a tuple of (path_to_chart, chart title for latex) -
    if there is no need for chart generation should return ("", "").
    Charts should be saved via :obj:`save_chart`.
    """

    order = ["categorical_distribution_chart"]

    @staticmethod
    def categorical_distribution_chart(
        X: pd.DataFrame, y: pd.Series
    ) -> Tuple[str, str]:
        """
        Generates a plot to visualize the distribution of categorical features.
        """
        settings = config.chart_settings
        sns.set_theme(style=settings["theme"])

        logger.start_operation("Categorical distribution visualization.")

        categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
        if not categorical_columns:
            logger.debug("No categorical features found in the dataset.")
            return "", ""
        logger.debug(
            "Will create categorical distribution visualization chart"
            f"for {categorical_columns} columns."
        )

        num_columns = len(categorical_columns)
        num_rows = (num_columns + 1) // 2

        _, axes = plt.subplots(
            num_rows,
            2,
            figsize=(
                settings["plot_width"],
                settings["plot_height_per_row"] * num_rows,
            ),
        )
        axes = axes.flatten()

        plot_count = 0
        for i, column in enumerate(categorical_columns):
            if X[column].nunique() > 15:
                logger.debug(
                    f"Skipping column {column} with more than 15 unique values."
                )
                continue
            sns.countplot(
                data=X,
                y=column,
                order=X[column].value_counts().index,
                ax=axes[plot_count],
                palette=sns.color_palette(settings["palette"], X[column].nunique()),
                legend=False,
            )
            axes[plot_count].set_title(
                f"Distribution of {column}",
                fontsize=settings["title_fontsize"],
                fontweight=settings["title_fontweight"],
            )
            axes[plot_count].set_xlabel(column, fontsize=settings["xlabel_fontsize"])
            axes[plot_count].set_ylabel("Count", fontsize=settings["ylabel_fontsize"])
            axes[plot_count].tick_params(
                axis="x", rotation=settings["tick_label_rotation"]
            )

            for p in axes[plot_count].patches:
                width = p.get_width()
                axes[plot_count].text(
                    width + 0.2,
                    p.get_y() + p.get_height() / 2,
                    f"{int(width)}",
                    ha="center",
                    va="center",
                    fontsize=10,
                )

            plot_count += 1

        for j in range(plot_count, len(axes)):
            axes[j].axis("off")

        plt.suptitle(
            "Categorical Features Distribution",
            fontsize=settings["title_fontsize"],
            fontweight=settings["title_fontweight"],
            y=1.0,
        )
        plt.tight_layout(pad=2.0)

        path = save_chart(name="categorical_distribution.png")
        return path, "Categorical distribution."
