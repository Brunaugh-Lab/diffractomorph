"""Generic, provenance-preserving processing operations."""

from .aggregate import (
    AggregateKWWConfig,
    AggregateKWWResult,
    aggregate_signal,
    fit_aggregate_kww,
    select_aggregate_start,
)
from .artifacts import ArtifactCorrectionConfig, ArtifactCorrectionResult, correct_artifacts
from .matched_extent import MatchedExtentConfig, matched_q3_extent

__all__ = [
    "AggregateKWWConfig", "AggregateKWWResult", "aggregate_signal", "fit_aggregate_kww",
    "select_aggregate_start",
    "ArtifactCorrectionConfig", "ArtifactCorrectionResult", "correct_artifacts",
    "MatchedExtentConfig", "matched_q3_extent",
]
