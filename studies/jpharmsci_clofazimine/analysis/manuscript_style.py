"""Shared figure style for the 2026 dissolution-diffraction manuscript figures.

Panel letters, type sizes, stroke widths, marker sizes, spine treatment and the ordered colour
ramp live here so the pH, loading and dissolution-medium figures read as one set. A figure module
imports this and adds only what is specific to its own panels.

**Colour is assigned by the job it does, not by taste.** The manuscript's varied factors — pH,
starting loading, medium polysorbate — are all ORDERED, so each gets the same single-hue
light-to-dark blue ramp (:data:`ORDERED_RAMP`), which stays legible under every common form of
colour-vision deficiency because it separates on lightness. A second identity inside one panel
(a model curve beside a measurement) takes :data:`VERMILLION`, the Okabe-Ito partner to the ramp's
dark end. Nothing here uses a rainbow, a dual axis, or hue alone to carry an ordered quantity.

Vector output is written with ``pdf.fonttype 42`` / ``svg.fonttype none`` so journal production
receives editable text rather than outlines.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402

# ── ordered ramp: one hue, light -> dark, for any factor the experiment stepped ──────────────
BLUE_PALE = "#9ECAE1"
BLUE_MID = "#4292C6"
BLUE = "#08519C"
ORDERED_RAMP = (BLUE_PALE, BLUE_MID, BLUE)

# ── second identity within a panel, and the neutral inks ─────────────────────────────────────
VERMILLION = "#D55E00"
DARK = "#1A1A1A"
GREY = "#7A7A7A"
PALE = "#C4C4C4"

FIG_W = 7.2                     # journal double-column width, inches

# Mark geometry, held constant across the figure set.
MS_MEAN = 4.4                   # condition / level mean marker
MS_UNIT = 2.6                   # individual replicate marker (open)
LW_MEAN = 1.0                   # line joining means
LW_UNIT = 0.8                   # thin preparation-level trajectory
ELINEWIDTH = 0.9
CAPSIZE = 2.4


def ramp_for(levels) -> dict:
    """Map ordered levels to the ramp, lightest first. Three levels is the manuscript's case."""
    levels = list(levels)
    if len(levels) > len(ORDERED_RAMP):
        raise ValueError(f"the ordered ramp carries {len(ORDERED_RAMP)} steps, "
                         f"asked for {len(levels)}; add a step deliberately rather than cycling")
    return dict(zip(levels, ORDERED_RAMP))


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8.0, "axes.labelsize": 8.0, "axes.titlesize": 8.2,
        "xtick.labelsize": 7.2, "ytick.labelsize": 7.2, "legend.fontsize": 6.6,
        "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.8)


def panel_tags(fig, axes_and_letters, dx_in: float = 0.46, dy: float = 0.030) -> None:
    """Bold panel letters at one shared offset in inches, so tags align across panel widths."""
    dx = dx_in / float(fig.get_figwidth())
    for ax, letter in axes_and_letters:
        box = ax.get_position()
        fig.text(box.x0 - dx, box.y1 + dy, letter, fontsize=9.6,
                 fontweight="bold", va="bottom", ha="left")


def save(fig, out_dir: Path, stem: str, formats) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, format=fmt, dpi=600 if fmt == "png" else None, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written
