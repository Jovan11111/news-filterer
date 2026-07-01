from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from src.analysis.comparison_visualizer import ComparisonVisualizer  # noqa: E402
from src.models.filter_run import FilterRun  # noqa: E402
from src.utils.enums.filter_mode import FilterMode  # noqa: E402
from src.utils.enums.price_horizon import PriceHorizon  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

_MODE_COLORS = {FilterMode.BUY: "#27ae60", FilterMode.SELL: "#c0392b", FilterMode.AFFECT: "#2980b9"}


class ReportGenerator:
    """Turns a filter's batch-size sweep into PNG graphs and a single PDF report.

    Every figure is saved as a clearly named PNG under ``graphs_dir`` and also
    embedded into one multi-page PDF under ``reports_dir`` (assembled with
    matplotlib's ``PdfPages`` so it needs no external converter).
    """

    def __init__(self, graphs_dir: Path, reports_dir: Path):
        self.__graphs_dir = graphs_dir
        self.__reports_dir = reports_dir
        self.__comparison = ComparisonVisualizer(output_dir=graphs_dir)

    @property
    def graphs_dir(self) -> Path:
        return self.__graphs_dir

    @property
    def reports_dir(self) -> Path:
        return self.__reports_dir

    def generate(self, dataset_name: str, top_k: int, seed: int, runs: list[FilterRun], subtract_market: bool = False, standardize: bool = False) -> Path:
        """Build all graphs and the PDF for ``runs`` and return the PDF path."""
        if not runs:
            raise ValueError("Cannot generate a report from an empty list of runs.")

        self.__graphs_dir.mkdir(parents=True, exist_ok=True)
        self.__reports_dir.mkdir(parents=True, exist_ok=True)

        filter_name = runs[0].evaluation.filter_name
        suffix = f"{'_market_adjusted' if subtract_market else ''}{'_zscore' if standardize else ''}"
        prefix = f"{filter_name}_{dataset_name}{suffix}"
        pdf_path = self.__reports_dir.joinpath(f"{prefix}_report.pdf")

        with PdfPages(pdf_path) as pdf:
            self.__add(pdf, self.__title_figure(filter_name, dataset_name, top_k, seed, runs, subtract_market, standardize))
            self.__add(pdf, self.__overview_figure(runs))

            for horizon in PriceHorizon:
                figure = self.__scalability_figure(runs, horizon)
                self.__save_png(figure, f"{prefix}_scalability_{horizon.value}")
                self.__add(pdf, figure)

            for run in runs:
                evaluation = run.evaluation
                title = f"Filter '{filter_name}' vs baseline - batch {run.requested_size} (sample {evaluation.batch_size})"
                comparison = self.__comparison.build_comparison_figure(evaluation, title)
                self.__save_png(comparison, f"{prefix}_batch_{run.requested_size}_comparison")
                self.__add(pdf, comparison)
                self.__add(pdf, self.__price_table_figure(run))

            self.__add(pdf, self.__summary_figure(runs, "Normalized score (0 = average, 1 = ideal)", "normalized_score", _format_plain))
            self.__add(pdf, self.__summary_figure(runs, "Gap to ideal (lower is better)", "gap_to_ideal", _format_signed))
            self.__add(pdf, self.__summary_figure(runs, "Lift over average (higher is better)", "lift_over_average", _format_signed))

        logger.info(f"Wrote PDF report to {pdf_path}.")
        return pdf_path

    def __title_figure(self, filter_name: str, dataset_name: str, top_k: int, seed: int, runs: list[FilterRun], subtract_market: bool, standardize: bool):
        sizes = ", ".join(str(run.requested_size) for run in runs)
        returns_label = "market-adjusted (S&P 500 subtracted)" if subtract_market else "raw price change"
        if standardize:
            returns_label += ", standardized (z-score by 30-day volatility)"
        lines = [
            "Filter evaluation report",
            "",
            f"Filter: {filter_name}",
            f"Dataset: {dataset_name}",
            f"Selection size (top_k): {top_k}",
            f"Returns: {returns_label}",
            f"Random seed: {seed}",
            f"Batch sizes: {sizes}",
            f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
        ]
        figure = plt.figure(figsize=(11.7, 8.3))
        figure.text(0.5, 0.6, "\n".join(lines), ha="center", va="center", fontsize=14, family="monospace")
        return figure

    def __overview_figure(self, runs: list[FilterRun]):
        columns = ["requested batch", "sampled articles", "filter time (ms)"]
        rows = [
            [str(run.requested_size), str(run.evaluation.batch_size), f"{run.evaluation.elapsed_seconds * 1000:.2f}"]
            for run in runs
        ]
        return self.__table_figure("Run overview", columns, rows)

    def __scalability_figure(self, runs: list[FilterRun], horizon: PriceHorizon):
        sizes = [run.requested_size for run in runs]
        figure, axis = plt.subplots(figsize=(11.7, 6))
        for mode in FilterMode:
            values = [_score_value(run, mode, horizon, "normalized_score") for run in runs]
            axis.plot(sizes, values, marker="o", label=mode.value, color=_MODE_COLORS[mode])

        axis.axhline(0.0, color="#7f8c8d", linestyle="--", linewidth=1, label="average floor")
        axis.axhline(1.0, color="#27ae60", linestyle=":", linewidth=1, label="ideal")
        axis.set_xscale("log")
        axis.set_xticks(sizes)
        axis.set_xticklabels([str(size) for size in sizes])
        axis.set_xlabel("batch size (number of news)")
        axis.set_ylabel("normalized score")
        axis.set_title(f"Normalized score vs batch size - {horizon.value} horizon")
        axis.legend(fontsize=8)
        axis.grid(True, which="both", linewidth=0.3)
        figure.tight_layout()
        return figure

    def __price_table_figure(self, run: FilterRun):
        columns = ["mode", "horizon", "filter", "ideal", "average", "worst"]
        rows = []
        for mode in FilterMode:
            for horizon in PriceHorizon:
                score = run.evaluation.score(mode, horizon)
                rows.append([
                    mode.value, horizon.value,
                    _format_signed(getattr(score, "filter_value", None)),
                    _format_signed(getattr(score, "ideal_value", None)),
                    _format_signed(getattr(score, "average_value", None)),
                    _format_signed(getattr(score, "worst_value", None)),
                ])
        title = f"Oriented price change (%) - batch {run.requested_size} (sample {run.evaluation.batch_size})"
        return self.__table_figure(title, columns, rows)

    def __summary_figure(self, runs: list[FilterRun], title: str, field: str, formatter):
        columns = ["mode", "horizon"] + [str(run.requested_size) for run in runs]
        rows = []
        for mode in FilterMode:
            for horizon in PriceHorizon:
                cells = [formatter(_score_value(run, mode, horizon, field)) for run in runs]
                rows.append([mode.value, horizon.value] + cells)
        return self.__table_figure(title, columns, rows)

    def __table_figure(self, title: str, columns: list[str], rows: list[list[str]]):
        figure, axis = plt.subplots(figsize=(11.7, max(2.5, 0.5 * len(rows) + 1.5)))
        axis.axis("off")
        axis.set_title(title, fontsize=12, pad=12)
        table = axis.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        for column in range(len(columns)):
            table[0, column].set_facecolor("#34495e")
            table[0, column].set_text_props(color="white", fontweight="bold")
        figure.tight_layout()
        return figure

    def __save_png(self, figure, name: str):
        path = self.__graphs_dir.joinpath(f"{name}.png")
        figure.savefig(path, dpi=150)
        logger.info(f"Wrote graph to {path}.")

    @staticmethod
    def __add(pdf: PdfPages, figure):
        pdf.savefig(figure)
        plt.close(figure)


def _score_value(run: FilterRun, mode: FilterMode, horizon: PriceHorizon, field: str) -> float | None:
    score = run.evaluation.score(mode, horizon)
    return getattr(score, field, None) if score is not None else None


def _format_signed(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.3f}"


def _format_plain(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"
