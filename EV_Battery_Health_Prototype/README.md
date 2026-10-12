# EV Battery Health Lab

An upgraded Streamlit prototype based on the supplied project proposal,
design/methodology report and literature review. The core study is how data
quality affects SOH/RUL predictions on batteries excluded from training.

## Start on Windows

Install Python 3.10 or newer, extract the complete project, then double-click
`run_prototype.bat`. It changes into the project directory, installs missing
dependencies and starts Streamlit. An internet connection is needed for the
first dependency installation.

Alternatively, from the project directory:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Important data distinction

The supplied prototype contained generated signals, not NASA measurement files.
The default mode is therefore explicitly **Synthetic demonstration**. Its
signals are generated from simulated ageing; small errors cannot establish real
battery accuracy. The included experiment outputs are synthetic demonstration
results only. No real NASA or CALCE accuracy is claimed.

Use the dashboard's Data source selector to upload real data:

- **NASA MAT files:** original NASA MATLAB v5 cell structs containing `cycle`,
  `type` and `data`. The loader pairs a charge with its following discharge
  capacity and skips impedance records. Unpaired capacities can still define
  the initial reference and EOL; only paired records become model inputs.
  MATLAB v7.3/HDF5 files are not supported.
- **Normalized sensor CSV:** adapter for CALCE or another dataset after conversion
  to the schema below. This is not a native CALCE Excel parser. Keep different
  datasets/chemistries separate when evaluating transfer.

| Column | Meaning / unit |
|---|---|
| battery_id | Independent cell identifier, unique across uploaded files |
| cycle | Positive integer cycle number |
| time | Seconds within the matching charge segment |
| voltage | Measured charging voltage, V |
| current | Measured charging current, A; positive during charge |
| temperature | Cell temperature, °C |
| capacity_ah | Measured discharge capacity label, Ah; constant within a cycle |

At least two independent batteries and three distinct timestamps per input
cycle are required. Four or more batteries are preferable, but even four cells
provide only four independent experimental units. The CSV must include the
first observed capacity and subsequent capacity labels needed to identify EOL.

## What changed

- Explicit data provenance and deterministic demo generation, without unsafe
  dataset or benchmark pickle loading.
- SOH uses the first measured capacity as its initial reference. EOL is the first
  observed SOH ≤ 80%. RUL is unknown when EOL is not observed; no fabricated
  retirement cycle is substituted. Exact RUL targets and metrics exclude these
  censored batteries; their SOH remains usable.
- Independent controls for noise, rounding, random loss, a burst gap, reduced
  sampling and early/upper partial charge windows.
- Unavailable features remain missing. Imputation and scaling fit only on
  training batteries. Long gaps are excluded from charge integration.
- Four LOBO models, clean versus augmented RF, repeated sensor seeds, per-battery
  errors, held-out permutation importance and feature-group removal experiments.
- Quality limits and break-even levels derive from measured curves. No hard-coded
  performance claims or thresholds. Tolerances are chosen by the user.
- Dataset and code fingerprints prevent loading stale experiment outputs.
- JSON/CSV exports, documented setup and automated pipeline tests.

## Dashboard

1. **Live prediction:** choose held-out cell/cycle, model and degradation. Inspect
   original/degraded signals, features, SOH/RUL and project-defined health grade.
2. **Robustness experiments:** choose tolerances and repeat count, then run the
   four isolated factor sweeps. Display error, seed variation, last passing grid
   level, first failure and first level not beating the cycle baseline.
3. **Cross-battery results:** compare all four models and inspect every LOBO fold.
4. **Features & ablation:** current feature availability, held-out permutation
   importance and retraining without each sensor-feature group.
5. **Data & methodology:** source, independent battery count, sample interval,
   initial capacity, observed EOL and metric definitions.

Live prediction and experimental sweeps are separate: the live sliders do not
change precomputed curves. Sweeps train on clean inputs except the explicit
augmented RF. They isolate factors; V/I/T rounding and burst gaps are live controls,
not mixed into the noise or missing-rate sweeps. Sampling steps mean every Nth
reading, not N seconds. Error bars are seed standard deviations, not confidence
intervals. The feature-group ablation compares clean and 50% early charging.

## Recompute experiments

```bash
python precompute_benchmarks.py --source synthetic --repeats 3
python precompute_benchmarks.py --source nasa --data-dir data/raw --repeats 3
python precompute_benchmarks.py --source csv --data-dir data/raw --repeats 3
```

Use `--quick` for a grid containing only clean and severe endpoints, or
`--output-dir results/nasa` for a separate result directory. The dashboard loads
`data/benchmark_summary.json` only if its data/code fingerprints match; it can
also run experiments on uploaded data and export JSON. Full sweeps can take
several minutes on an ordinary laptop. Models are fixed before testing; this
prototype does not optimize hyperparameters on held-out cells.

Primary metrics are equal-weight means of battery-fold MAE/RMSE. SOH errors are
percentage points, RUL errors are cycles. Limits are descriptive bounds on the
sampled grid, with no interpolation or extrapolation. The first failure remains
the boundary even if a later grid point improves. A missing failure means the
tested range had no crossing, not unlimited robustness.

```bash
python -m pytest -q test_pipeline.py
```

## Files

- `app.py`: dashboard
- `data_loader.py`: demo generator, MAT/CSV loading and labels
- `simulator.py`: controlled input degradation
- `features.py`: voltage, timing, thermal, integrated-charge and cycle features
- `models.py`: models, battery-disjoint fitting/evaluation and grid boundaries
- `precompute_benchmarks.py`: experiment runner and export metadata
- `test_pipeline.py`: numerical, ingestion and leakage regression tests
- `PROJECT_REPORT.md`: implementation status and report traceability
- `data/`: reproducible synthetic results
- `references/`: three supplied source reports in the downloadable package

This is a cell-level research prototype. It does not model battery packs, certify
safety, or establish transfer across chemistry or real-vehicle conditions.
Prediction uncertainty intervals, native CALCE ingestion, large benchmark
datasets and field validation remain future work.
