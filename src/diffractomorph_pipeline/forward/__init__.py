"""Per-run forward dissolution models.

Two complementary engines:

**Mechanistic surface-pH Nernst–Brunner** (``params`` / ``buffer`` / ``surface`` /
``surface_ode``) — a parameterized surface-pH dissolution model (formulated by J. Al-Gousous;
see ``surface_ode`` for the physics and references). Predicts dissolution from chemistry with
**no fitted rate**: tune everything in :class:`Parameters`, swap the starting distribution with
:class:`PSD`, run :func:`simulate_dissolution`. ``predict`` supplies the measured Cs(pH) at
runtime (see ``solubility.py``). The exploratory independent-exponent morphology terms
(``a_area`` / ``b_size`` / ``rate_scale``; deprecated ``w_por`` / ``f_acc``) live separately in
``morphology`` (``MorphologyParams`` + ``simulate_morphology``), off by default.

    from diffractomorph_pipeline.forward import Parameters, PSD, simulate_dissolution
    run = simulate_dissolution(Parameters(s0_uM=0.6), psd, dose_mg=0.17, ph_bulk=4.5)

**Crude fitted-rate engine** (``model``) — size-resolved Noyes–Whitney with a simple
``1−C/Cs`` driving force that **fits a single transferable rate G** to observed scattering.
Reached via ``forward.simulate`` / ``forward.fit_rate``; also carries ``ensemble_beta``
(log-normal PSD width σ_g → KWW β). Superseded for prediction by the mechanistic model above.
"""
# ── mechanistic surface-pH model (primary) ───────────────────────────────────
from diffractomorph_pipeline.forward.params import Diffusivities, PSD, Parameters
from diffractomorph_pipeline.forward.surface_ode import (
    DissolutionRun,
    volume_psd,
)
from diffractomorph_pipeline.forward.surface_ode import simulate as simulate_dissolution
from diffractomorph_pipeline.forward.morphology import MorphologyParams
from diffractomorph_pipeline.forward.morphology import simulate as simulate_morphology
from diffractomorph_pipeline.forward.predict import predict, predict_from_snapshot
from diffractomorph_pipeline.forward.injected_mass import InjectedMass
from diffractomorph_pipeline.forward.psd_evolution import bucket_kinetics, compare_bucket_kinetics, q3_evolution
from diffractomorph_pipeline.forward.buffer import (
    solve_bulk_H,
    solve_bulk_pH,
    spectator_for_pH,
)
from diffractomorph_pipeline.forward.surface import (
    solve_surface_H,
    surface_solubility,
)
from diffractomorph_pipeline.forward.registry import (
    ForwardModelSpec, NamedForwardResult, model_spec, model_specs, run_named_model,
)

# ── crude fitted-rate engine (kept; different job) ───────────────────────────
from diffractomorph_pipeline.forward.model import (
    ForwardRun,
    ensemble_beta,
    fit_rate,
    residual,
    simulate,
)

__all__ = [
    # mechanistic surface-pH model
    "Parameters", "PSD", "Diffusivities", "simulate_dissolution", "predict",
    "predict_from_snapshot", "q3_evolution", "bucket_kinetics", "compare_bucket_kinetics",
    "InjectedMass", "MorphologyParams", "simulate_morphology",
    "DissolutionRun", "volume_psd", "solve_bulk_H", "solve_bulk_pH", "spectator_for_pH",
    "solve_surface_H", "surface_solubility",
    # crude fitted-rate engine
    "simulate", "ForwardRun", "fit_rate", "residual", "ensemble_beta",
    "ForwardModelSpec", "NamedForwardResult", "model_spec", "model_specs", "run_named_model",
]
