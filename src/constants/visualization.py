"""Shared palette and layout for the graphs and the PDF report."""

from src.utils.enums.filter_mode import FilterMode

# --- Palette ------------------------------------------------------------------

# Semantic colors shared across every figure.
COLOR_BUY = "#27ae60"     # green: gains / ideal
COLOR_SELL = "#c0392b"    # red: losses / worst
COLOR_AFFECT = "#2980b9"  # blue: magnitude
COLOR_NEUTRAL = "#7f8c8d"  # grey: the average (random) floor
COLOR_TABLE_HEADER = "#34495e"  # dark slate: table header background

# Color used for each filter mode in charts.
MODE_COLORS = {
    FilterMode.BUY: COLOR_BUY,
    FilterMode.SELL: COLOR_SELL,
    FilterMode.AFFECT: COLOR_AFFECT,
}

# --- Comparison bars ----------------------------------------------------------

# The four baseline/filter series drawn per mode, their labels and their colors
# (kept in the same order so bar, label and color line up).
COMPARISON_SERIES = ("worst_value", "average_value", "filter_value", "ideal_value")
COMPARISON_SERIES_LABELS = ("worst", "average", "filter", "ideal")
COMPARISON_SERIES_COLORS = (COLOR_SELL, COLOR_NEUTRAL, COLOR_AFFECT, COLOR_BUY)

# Width of a single bar within a mode's group.
COMPARISON_BAR_WIDTH = 0.2

# --- Layout -------------------------------------------------------------------

# Resolution (dots per inch) every saved PNG is rendered at.
FIGURE_DPI = 150

# Landscape A4 page size (inches) used for the report's full-page figures.
PAGE_SIZE = (11.7, 8.3)
