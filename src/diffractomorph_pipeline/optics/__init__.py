"""Optical forward operator (Mie kernel) for the dissolution-optics pipeline.

A *shared dependency*, not a stage. Two consumers import it: the per-day
size-consistency QC and the forward Noyes–Whitney model. Used **forward only**
(PSD → pattern); never inverted.

- ``mie``       — loader + ``forward`` + ``channel_size_map`` + ``shape_consistency`` (use)
- ``mie_build`` — ``fit_geometry`` / ``fit_refractive_index`` / ``build_kernel`` /
                  ``resolve_kernel_for_day`` (build / calibration)
"""
from diffractomorph_pipeline.optics.mie import (
    MieKernel,
    channel_size_map,
    forward,
    load_kernel,
    shape_consistency,
)

__all__ = [
    "MieKernel",
    "load_kernel",
    "forward",
    "channel_size_map",
    "shape_consistency",
]
