from typing import Dict, List, Tuple

import pandas as pd

from ..utils.logging_config import setup_logger
from ..visualization.categical import CategoricalVisualizer
from ..visualization.eda import EdaVisualizer
from ..visualization.numerical import NumericalVisualizer
from .raport import Report

logger = setup_logger(__name__)


class EdaRaport:
    visualizers: list = [EdaVisualizer, CategoricalVisualizer, NumericalVisualizer]

    def __init__(self):
        # for each visualizer_name - key, list of its charts as (path, caption)
        self.charts_dt: Dict[str, List[Tuple[str, str]]] = {}

    def run(self, X: pd.DataFrame, y: pd.Series):
        """Performs dataset overview."""

        logger.start_operation("Eda.")

        try:
            for visualiser_cls in EdaRaport.visualizers:
                logger.start_operation(f"{visualiser_cls.__name__} plot generation.")
                logger.debug(
                    f"Will call plots in following order: {visualiser_cls.order}"
                )
                self.charts_dt[visualiser_cls.__name__] = []

                for method_name in visualiser_cls.order:
                    method = getattr(visualiser_cls, method_name)

                    logger.debug(f"Calling {method}...")
                    chart_dt = method(X, y)
                    if chart_dt[0] != "":
                        self.charts_dt[visualiser_cls.__name__].append(chart_dt)

                logger.end_operation()

        except Exception as e:
            logger.error(f"Failed to gather eda statistics: {str(e)}")
            raise e

        logger.end_operation()

    def write_to_raport(self, raport: Report):
        """Writes eda section to a raport"""

        eda_section = raport.add_section("Eda")  # noqa: F841

        for visualizer_name, charts_dt in self.charts_dt.items():
            raport.add_subsection(visualizer_name[: -len("Visualizer")])

            for path, caption in charts_dt:
                raport.add_figure(path=path, caption=caption)

        return raport
