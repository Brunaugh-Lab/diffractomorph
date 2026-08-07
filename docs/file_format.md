# Sympatec PAQXOS Export File Format Reference

This pipeline ingests the per-measurement exports produced by Sympatec's PAQXOS
laser-diffraction software. This document records the raw layout so `ingest`
(Step 0, `extract_run`) can be implemented and maintained without
depending on undocumented institutional knowledge. The layout is generated through
PAQXOS's normal Report Template Editor and parsed from the resulting RTF export. The
pipeline does not access PAQXOS's internal databases or executables.

The report-template body used to request optical concentration plus reference and
measured ring intensities is included at
[`templates/paqxos_raw_intensity_report_template.txt`](templates/paqxos_raw_intensity_report_template.txt).
It was reconstructed from the lab's archived print-to-PDF copy of the working template (SHA-256
`177fdf18f15c7bc42779b4509ec953a309ecad8b77544b2e019fe55ed562a34c`). Paste the text into an
empty PAQXOS report template and validate it with the editor's report preview before operational use.

PAQXOS 5.1 Operating Instructions document the Report Template Editor and `@I.REF(N)`.
The other substitutions in the bundled template are evidenced by the archived working template
and rendered exports. In particular, `@I.NORM(N)` was not found by text search of the PAQXOS 5.1
or WINDOX 5.6 command-reference PDFs reviewed for this release.

## Container

- Exports are **RTF** (`.rtf`); one file per run (measurement or blank).
- De-RTF'd to plain text via `striprtf.rtf_to_text`.

## Per-frame structure

A file is a sequence of frames. Each frame:

```
Measurement Name:  Compound X dissolution measurement
Measurement Time:   2026-01-01 15:49:38
Report Generated on: 2026-06-09 17:51:36
Optical Concentration: 1.70
Channel, Ref Value, Measured Value
1, 1.80, 1.40
2, 0.68, 0.64
... (31 channels)
```

| Marker / row | Meaning |
|---|---|
| `Measurement Name:` | run label (constant within a file) |
| `Measurement Time: YYYY-MM-DD HH:MM:SS` | frame timestamp |
| `Optical Concentration: <float>` | optical concentration (Copt), as reported by PAQXOS |
| `Channel, Ref Value, Measured Value` | column header |
| `<ch>, <ref>, <measured>` | per detector ring: index, **reference**, **measured** |

## Parser contracts

1. **Two intensity columns.** `Measured` is the per-frame signal (analysis
   target); `Ref` is the instrument's stored reference spectrum. **Capture
   both** — `Ref` feeds optional static-baseline subtraction (`I_bgsub = I − ref`)
   and the reference-variation check. The adapter retains the earliest valid
   frame's stored-reference vector and flags variation across retained frames;
   variation does not automatically exclude a frame.
2. **Frame structure is validated against detector identifiers.** A PAQXOS
   instrument profile may declare numeric `channel_ids` in strictly increasing
   order. A frame is retained only when its timestamp is parseable and every
   declared identifier occurs exactly once. Missing, additional, substituted,
   or duplicate identifiers are counted by exclusion reason. Without declared
   identifiers, the adapter infers a uniquely most frequent exact set and fails
   closed when equally supported sets are ambiguous.
3. **Frames may be listed newest-first.** The parser sorts by timestamp ascending
   and records whether reverse document order was detected.

`Measured` is **not** background-subtracted: in a blank,
`Measured ≈ Ref ≈ 0.3–0.6` (not zero). `Measured − Ref` is the drug-attributable
scattering, offered as a derived `I_bgsub`; `Measured` is the primary signal.

The current parser treats each new `Measurement Time:` marker as the start of a frame and
sorts parsed timestamps chronologically. It does not depend on `@PAR(1)` as a structural
separator. In the PAQXOS 5.1 manual, `@PAR(1)` means the first user-defined measurement
parameter; it is retained in the archived template body for fidelity but is not required by
the parser or by intensity extraction. Missing Copt, reference variation,
reverse document order, and interframe gaps are reported as run-level flags,
not additional automatic frame exclusions.

## Output contract

The PAQXOS adapter returns the instrument-neutral `Run` contract described in
[`data_contracts.md`](data_contracts.md). Existing analyses can still access the
legacy names `I`, `ref`, `copt`, `t_min`, `t0`, and `channels` through
compatibility properties. These names are not requirements for other adapters.

## CSV mirror

`dfm-ingest` / `extract_run(emit_csv=True)` writes one tidy CSV per run —
`frame, time_iso, t_min, copt, I_ch1 … I_ch31` — plus a `<run>_meta.json`
sidecar carrying the earliest retained frame's stored-reference vector and
run-level flags.

## Open questions

- Mapping from channel index → scattering angle / particle-size bin (µm), via the
  glass-bead-calibrated Mie kernel.
- Whether multiple lenses/configs appear across the dataset.
