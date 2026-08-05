"""Explicit identities and inference boundaries for forward-model engines."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ForwardModelSpec:
    model_id: str
    domain: str
    status: str
    engine: str
    uses_detector_signal: bool
    uses_optical_operator: bool
    allowed_inference: str


@dataclass(frozen=True)
class NamedForwardResult:
    spec: ForwardModelSpec
    result: object


_MODELS = {
    "mass_surface_ph_nb_v1": ForwardModelSpec(
        model_id="mass_surface_ph_nb_v1",
        domain="mass",
        status="primary",
        engine="simulate_dissolution",
        uses_detector_signal=False,
        uses_optical_operator=False,
        allowed_inference="independently parameterized mass-domain dissolution trajectory",
    ),
    "mass_morphology_diagnostic_v1": ForwardModelSpec(
        model_id="mass_morphology_diagnostic_v1",
        domain="mass",
        status="exploratory",
        engine="simulate_morphology",
        uses_detector_signal=False,
        uses_optical_operator=False,
        allowed_inference="diagnostic sensitivity to named morphology/rate exponents",
    ),
    "optical_fitted_g_v1": ForwardModelSpec(
        model_id="optical_fitted_g_v1",
        domain="optical",
        status="legacy_exploratory",
        engine="simulate/fit_rate",
        uses_detector_signal=True,
        uses_optical_operator=True,
        allowed_inference="fitted detector-space description; not independent mass validation",
    ),
}


def model_spec(model_id: str) -> ForwardModelSpec:
    try:
        return _MODELS[model_id]
    except KeyError as exc:
        raise KeyError(f"unknown forward model {model_id!r}; available: {', '.join(_MODELS)}") from exc


def model_specs() -> tuple[ForwardModelSpec, ...]:
    return tuple(_MODELS.values())


def run_named_model(model_id: str, *args, **kwargs) -> NamedForwardResult:
    """Execute one explicitly selected engine and retain its inference identity."""
    spec = model_spec(model_id)
    if model_id == "mass_surface_ph_nb_v1":
        from diffractomorph_pipeline.forward.surface_ode import simulate
        result = simulate(*args, **kwargs)
    elif model_id == "mass_morphology_diagnostic_v1":
        from diffractomorph_pipeline.forward.morphology import simulate
        result = simulate(*args, **kwargs)
    elif model_id == "optical_fitted_g_v1":
        from diffractomorph_pipeline.forward.model import simulate
        result = simulate(*args, **kwargs)
    else:  # pragma: no cover - model_spec already rejects unknown ids
        raise AssertionError(model_id)
    return NamedForwardResult(spec=spec, result=result)
