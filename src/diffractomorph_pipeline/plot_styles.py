"""
Lab figure style specification.
All rules are hardcoded. Students don't choose fonts, colors, or formatting.
Based on Tufte's data-ink ratio principle and PI preferences.
"""

import matplotlib
import matplotlib.pyplot as plt

# ── Typography ───────────────────────────────────────────────────────────────
FONT_FAMILY = 'Arial'
FONT_WEIGHT_LABELS = 'bold'
FONT_SIZE_TICK = 10
FONT_SIZE_AXIS_LABEL = 12
FONT_SIZE_ANNOTATION = 9
FONT_SIZE_PANEL_LABEL = 14
FONT_SIZE_LEGEND = 8.5

# ── Color palette (colorblind-safe, Okabe-Ito based) ────────────────────────
# Control is ALWAYS black. Conditions assigned in order.
COLORS = [
    '#2C2C2A',  # Control / baseline (black)
    '#378ADD',  # Condition 1 (blue)
    '#1D9E75',  # Condition 2 (teal)
    '#D85A30',  # Condition 3 (coral)
    '#7F77DD',  # Condition 4 (purple)
    '#D4537E',  # Condition 5 (pink)
]
REPLICATE_ALPHA = 0.22
MEAN_ALPHA = 1.0
REPLICATE_LINEWIDTH = 0.7
MEAN_LINEWIDTH = 2.0
REPLICATE_MARKERSIZE = 3.5
MEAN_MARKERSIZE = 5
DOT_PLOT_MARKERSIZE = 7
DOT_PLOT_ALPHA = 0.5
MEAN_BAR_WIDTH_FRAC = 0.08  # fraction of x-axis range for mean horizontal bar

# ── pH-study condition style (manuscript) ────────────────────────────────────
# ONE colour, line style, and marker per condition, shared across EVERY manuscript
# figure so a condition never silently changes appearance. Colours are Okabe-Ito
# (colourblind-safe); the distinct line styles + markers keep conditions separable
# in grayscale. Keyed by float pH so callers look up by condition value, not order.
PH_ORDER = [4.0, 4.5, 5.0]
PH_STYLE = {
    4.0: {"color": "#0072B2", "linestyle": "-",   "marker": "o", "label": "pH 4.0"},   # blue
    4.5: {"color": "#E69F00", "linestyle": "--",  "marker": "s", "label": "pH 4.5"},   # orange
    5.0: {"color": "#CC79A7", "linestyle": "-.",  "marker": "^", "label": "pH 5.0"},   # reddish-purple
}
BAND_ALPHA = 0.18            # between-date SD band
DATE_LINE_ALPHA = 0.45       # thin per-date trajectories
DATE_LINE_WIDTH = 0.8

# ── Axes and framing (Tufte) ─────────────────────────────────────────────────
SPINE_LINEWIDTH = 0.8
TICK_DIRECTION = 'in'
TICK_LENGTH_MAJOR = 5
TICK_LENGTH_MINOR = 3

# ── Output ───────────────────────────────────────────────────────────────────
DPI = 300
SINGLE_COL_WIDTH = 3.5   # inches
DOUBLE_COL_WIDTH = 7.0   # inches


def apply_lab_style() -> None:
    """Apply the lab figure style globally via rcParams."""
    matplotlib.rcParams.update({
        'font.family': FONT_FAMILY,
        'font.size': FONT_SIZE_TICK,
        'font.weight': FONT_WEIGHT_LABELS,
        'axes.labelsize': FONT_SIZE_AXIS_LABEL,
        'axes.labelweight': FONT_WEIGHT_LABELS,
        'axes.linewidth': SPINE_LINEWIDTH,
        'xtick.labelsize': FONT_SIZE_TICK,
        'ytick.labelsize': FONT_SIZE_TICK,
        'xtick.direction': TICK_DIRECTION,
        'ytick.direction': TICK_DIRECTION,
        'xtick.major.size': TICK_LENGTH_MAJOR,
        'ytick.major.size': TICK_LENGTH_MAJOR,
        'xtick.minor.size': TICK_LENGTH_MINOR,
        'ytick.minor.size': TICK_LENGTH_MINOR,
        'legend.fontsize': FONT_SIZE_LEGEND,
        'legend.frameon': False,
        'figure.dpi': DPI,
        'savefig.dpi': DPI,
        'savefig.bbox': 'tight',
        'pdf.fonttype': 42,   # TrueType fonts in PDF (editable in Illustrator)
        'ps.fonttype': 42,
    })


def setup_axes(ax, log_x: bool = False, log_y: bool = False,
               remove_top_right: bool = True) -> None:
    """Configure axes per lab standard."""
    if remove_top_right:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    else:
        # Keep all spines but make top/right lighter (for log-log plots)
        ax.spines['top'].set_linewidth(SPINE_LINEWIDTH * 0.5)
        ax.spines['right'].set_linewidth(SPINE_LINEWIDTH * 0.5)
        ax.tick_params(which='both', top=True, right=True)

    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')

    ax.tick_params(which='both', direction=TICK_DIRECTION)


def get_color(index: int) -> str:
    """Get color for condition index. 0 = control (black)."""
    return COLORS[index % len(COLORS)]


def ph_style(ph: float) -> dict:
    """Locked per-condition style dict (``color``/``linestyle``/``marker``/``label``) for a pH.

    Falls back to the ordered lab palette (solid, ``o``) for a pH outside the pre-registered
    trio, so an unexpected condition still renders rather than raising."""
    key = round(float(ph), 1)
    if key in PH_STYLE:
        return PH_STYLE[key]
    idx = 1 + (PH_ORDER.index(key) if key in PH_ORDER else 0)
    return {"color": get_color(idx), "linestyle": "-", "marker": "o", "label": f"pH {key}"}


def apply_manuscript_style() -> None:
    """Lab style + the extra manuscript requirements: white background, editable/embedded
    TrueType fonts in the PDF, and a floor on the smallest text so nothing prints below ~7 pt."""
    apply_lab_style()
    matplotlib.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "pdf.fonttype": 42,          # embedded editable TrueType (Illustrator-editable)
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
        "legend.fontsize": 7.5,
    })


def panel_label(ax, letter: str, *, x: float = -0.16, y: float = 1.06,
                fontsize: int = FONT_SIZE_PANEL_LABEL) -> None:
    """Bold panel tag (``A``, ``B``, …) in axes-fraction coords, placed outside the plot region
    at a consistent offset so tags align across a figure's panels."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=fontsize,
            fontweight="bold", va="bottom", ha="right")
