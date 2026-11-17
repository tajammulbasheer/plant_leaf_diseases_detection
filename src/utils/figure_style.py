"""Shared palette for the README figures (slate text, blue/orange series, single-hue blue ramp)."""

from matplotlib.colors import LinearSegmentedColormap

INK = "#0f172a"        # titles, labels
INK_2 = "#1e293b"      # tick labels
MUTED = "#64748b"      # subtitles
BORDER = "#cbd5e1"     # spines, frames
LEGEND_BG, LEGEND_EDGE = "#f8fafc", "#e2e8f0"
BLUE_600, ORANGE_600 = "#2563eb", "#ea580c"
GREEN_BOX = (34 / 255, 197 / 255, 94 / 255)  # pseudo bounding boxes (same green as the Gradio app)

BLUE_RAMP = LinearSegmentedColormap.from_list(
    "slate_blue_ramp", ["#f8fafc", "#bfdbfe", "#3b82f6", "#1d4ed8", "#172554"]
)
