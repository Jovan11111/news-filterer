from pathlib import Path

import matplotlib.pyplot as plt

from src.constants.visualization import (
    COMPARISON_BAR_WIDTH,
    COMPARISON_SERIES,
    COMPARISON_SERIES_COLORS,
    COMPARISON_SERIES_LABELS,
    FIGURE_DPI,
)
from src.models.evaluation_result import EvaluationResult
from src.utils.enums.filter_mode import FilterMode
from src.utils.enums.price_horizon import PriceHorizon
from src.utils.logger import get_logger

plt.switch_backend("Agg")

logger = get_logger(__name__)


class ComparisonVisualizer:
    """Draws a filter's selection against the ideal/average/worst baseline.

    One figure holds a panel per horizon; within a panel every mode shows four
    bars (worst, average, filter, ideal) in the mode's oriented value, so the
    distance from filter to ideal and its lift over the random floor are visible
    at a glance.
    """

    def __init__(self, output_dir: Path):
        """Create the visualizer with the directory figures are written to.

        Args:
            output_dir (Path): Directory the comparison PNGs are saved in.
        """
        self.__output_dir = output_dir

    @property
    def output_dir(self) -> Path:
        """The directory the comparison PNGs are saved in."""
        return self.__output_dir

    def render(self, result: EvaluationResult, tag: str) -> Path:
        """Render result to a PNG named after the filter and tag.

        Args:
            result (EvaluationResult): The evaluation to visualize.
            tag (str): Suffix identifying the dataset in the file name and title.

        Returns:
            Path: Path to the written PNG figure.
        """
        title = f"Filter '{result.filter_name}' vs baseline ({tag}) - top {result.top_k} of {result.batch_size}"
        figure = self.build_comparison_figure(result, title)

        self.__output_dir.mkdir(parents=True, exist_ok=True)
        path = self.__output_dir.joinpath(f"{result.filter_name}_{tag}.png")
        figure.savefig(path, dpi=FIGURE_DPI)
        plt.close(figure)
        logger.info(f"Wrote comparison figure to {path}.")
        return path

    def build_comparison_figure(self, result: EvaluationResult, title: str):
        """Build (without saving) the per-mode/horizon comparison figure.

        Args:
            result (EvaluationResult): The evaluation to visualize.
            title (str): The figure's overall title.

        Returns:
            Figure: The assembled matplotlib figure.
        """
        horizons = list(PriceHorizon)
        modes = list(FilterMode)

        figure, axes = plt.subplots(1, len(horizons), figsize=(7 * len(horizons), 5), squeeze=False)
        for column, horizon in enumerate(horizons):
            self.__draw_panel(axes[0][column], result, modes, horizon)

        figure.suptitle(title)
        figure.tight_layout(rect=(0, 0, 1, 0.96))
        return figure


    def __draw_panel(self, axis, result: EvaluationResult, modes: list[FilterMode], horizon: PriceHorizon):
        """Draw the worst/average/filter/ideal bars for every mode on one horizon panel."""
        positions = list(range(len(modes)))
        offsets = [(index - (len(COMPARISON_SERIES) - 1) / 2) * COMPARISON_BAR_WIDTH for index in range(len(COMPARISON_SERIES))]

        for series, label, color, offset in zip(COMPARISON_SERIES, COMPARISON_SERIES_LABELS, COMPARISON_SERIES_COLORS, offsets):
            heights = [self.__value(result.score(mode, horizon), series) for mode in modes]
            axis.bar([position + offset for position in positions], heights, COMPARISON_BAR_WIDTH, label=label, color=color)

        axis.set_title(f"{horizon.value} horizon")
        axis.set_xticks(positions)
        axis.set_xticklabels([mode.value for mode in modes])
        axis.set_ylabel("oriented price change (%, higher is better)")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.legend(fontsize=8)

    @staticmethod
    def __value(score, field: str) -> float:
        """Return a score field as a plottable number, treating missing values as 0."""
        if score is None:
            return 0.0
        value = getattr(score, field)
        return value if value is not None else 0.0
