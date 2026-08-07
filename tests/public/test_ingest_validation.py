from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from diffractomorph_pipeline.ingest import extract_run
from diffractomorph_pipeline.io.paqxos_rtf import PaqxosRtfReader
from diffractomorph_pipeline.study.manifest import ProfileSpec, ProjectManifest, RunSpec


def _write_rtf(path: Path, frames: list[dict], *, newest_first: bool = True) -> Path:
    ordered = sorted(frames, key=lambda frame: frame["time"], reverse=newest_first)
    lines: list[str] = []
    for frame in ordered:
        lines.extend(
            [
                "Measurement Name: synthetic validation run",
                f"Measurement Time: {frame['time'].strftime('%Y-%m-%d %H:%M:%S')}",
                "Optical Concentration: "
                + ("" if frame.get("copt") is None else str(frame["copt"])),
                "Channel, Ref Value, Measured Value",
            ]
        )
        for channel, reference, measured in frame["rows"]:
            lines.append(f"{channel}, {reference}, {measured}")
    path.write_text("{\\rtf1\\ansi\n" + "\\par\n".join(lines) + "\\par\n}")
    return path


def _frame(index: int, channels=(1, 2, 3), *, copt=1.0) -> dict:
    started = datetime(2026, 1, 1, 12, 0, 0)
    return {
        "time": started + timedelta(seconds=30 * index),
        "copt": copt,
        "rows": [(channel, 0.1 * channel, 10 * index + channel) for channel in channels],
    }


def test_expected_channels_drop_same_length_wrong_identifier_set(tmp_path):
    source = _write_rtf(
        tmp_path / "run.rtf",
        [_frame(0), _frame(1, channels=(1, 2, 4)), _frame(2)],
    )

    run = extract_run(source, run_kind="measurement", expected_channel_ids=(1, 2, 3))

    assert run.channel_ids == ("1", "2", "3")
    assert run.signal.shape == (2, 3)
    assert run.flags["dropped_frames"] == 1
    assert run.flags["dropped_frame_reasons"] == {"channel_set_mismatch": 1}


def test_expected_channels_fail_when_no_frame_matches(tmp_path):
    source = _write_rtf(
        tmp_path / "run.rtf",
        [_frame(0, channels=(1, 2, 4)), _frame(1, channels=(1, 2, 4))],
    )

    with pytest.raises(ValueError, match="no structurally valid frames"):
        extract_run(source, expected_channel_ids=(1, 2, 3))


def test_expected_channels_must_use_numeric_detector_order(tmp_path):
    source = _write_rtf(tmp_path / "run.rtf", [_frame(0), _frame(1)])

    with pytest.raises(ValueError, match="strictly increasing order"):
        extract_run(source, expected_channel_ids=(2, 1, 3))


def test_inference_fails_closed_for_equally_complete_channel_sets(tmp_path):
    source = _write_rtf(
        tmp_path / "run.rtf",
        [_frame(0, channels=(1, 2, 3)), _frame(1, channels=(1, 2, 4))],
    )

    with pytest.raises(ValueError, match="ambiguous detector-channel sets"):
        extract_run(source)


def test_duplicate_channel_row_is_not_silently_overwritten(tmp_path):
    duplicate = _frame(1)
    duplicate["rows"].append((2, 99.0, 99.0))
    source = _write_rtf(tmp_path / "run.rtf", [_frame(0), duplicate, _frame(2)])

    run = extract_run(source, expected_channel_ids=(1, 2, 3))

    assert run.signal.shape == (2, 3)
    assert run.flags["dropped_frame_reasons"] == {"duplicate_channel_ids": 1}


def test_missing_copt_and_reference_variation_are_retained_as_flags(tmp_path):
    frames = [_frame(0, copt=None), _frame(1)]
    frames[1]["rows"][0] = (1, 0.5, 11.0)
    source = _write_rtf(tmp_path / "run.rtf", frames)

    with pytest.warns(UserWarning) as recorded:
        run = extract_run(source, expected_channel_ids=(1, 2, 3))

    assert len(recorded) == 2
    assert run.flags["copt_nan"] == 1
    assert run.flags["ref_static"] is False
    assert run.flags["dropped_frames"] == 0
    assert np.isnan(run.acquisition["copt"][0])


def test_profile_aware_reader_passes_declared_channels_to_parser(tmp_path):
    source = _write_rtf(
        tmp_path / "run.rtf",
        [_frame(0), _frame(1, channels=(1, 2, 4)), _frame(2)],
    )
    spec = SimpleNamespace(
        source=source,
        run_kind="measurement",
        run_id="run-1",
        sample_id="sample-1",
        independent_unit_id="prep-1",
        technical_replicate="1",
        instrument_id="instrument-1",
        metadata={},
    )

    run = PaqxosRtfReader().read_with_instrument_profile(
        spec, {"adapter": "paqxos_rtf", "channel_ids": [1, 2, 3]}
    )

    assert run.signal.shape == (2, 3)
    assert run.flags["expected_channel_ids_source"] == "instrument_profile"


def _project(source: Path, channel_ids) -> ProjectManifest:
    instrument_parameters = {"adapter": "paqxos_rtf"}
    if channel_ids is not None:
        instrument_parameters["channel_ids"] = channel_ids
    spec = RunSpec(
        run_id="run-1",
        source=source,
        adapter="paqxos_rtf",
        run_kind="measurement",
        sample_id="sample-1",
        independent_unit_id="prep-1",
    )
    return ProjectManifest(
        manifest_path=source.parent / "project.yaml",
        schema_version=1,
        project_id="validation-project",
        data_root=source.parent,
        independent_unit="preparation",
        profiles={
            "instrument": ProfileSpec(
                role="instrument",
                profile_id="instrument-1",
                parameters=instrument_parameters,
            )
        },
        runs=(spec,),
    )


def test_manifest_declared_channels_fail_when_no_frame_matches(tmp_path):
    source = _write_rtf(
        tmp_path / "run.rtf",
        [_frame(0, channels=(1, 2, 4)), _frame(1, channels=(1, 2, 4))],
    )

    with pytest.raises(ValueError, match="no structurally valid frames"):
        _project(source, [1, 2, 3]).read_run("run-1")


def test_manifest_without_declared_channels_fails_on_ambiguous_sets(tmp_path):
    source = _write_rtf(
        tmp_path / "run.rtf",
        [_frame(0, channels=(1, 2, 3)), _frame(1, channels=(1, 2, 4))],
    )

    with pytest.raises(ValueError, match="ambiguous detector-channel sets"):
        _project(source, None).read_run("run-1")
