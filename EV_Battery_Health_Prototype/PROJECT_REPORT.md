# EV Battery Health Prediction: Updated Prototype Report

## Purpose and evidence

The prototype implements a controlled stress test of battery-health predictions
using the three supplied reports: the Special Project Proposal, Design and
Methodology Report, and Literature Review. The reports are requirements and
research context; their cited paper results are not results of this prototype.

The original code generated synthetic signals while labeling them NASA data and
included unsupported fixed accuracy/quality claims. This update removes those
claims. No public raw dataset was included in the selected project folder.
Default demonstrations and shipped experiment outputs are therefore synthetic.
They verify the implementation, not predictive performance on real EVs.

## Requirements implemented

| Report requirement | Implementation / output |
|---|---|
| Public-data pipeline | Original NASA v5 MAT ingestion; normalized CSV adapter for CALCE/other data |
| Fixed SOH definition | 100 × discharge capacity / first measured discharge capacity |
| Fixed RUL / retirement | First observed SOH ≤ 80%; censored cells have unknown RUL |
| Four degradation factors | Reading decimation, V/I/T noise, random missing readings, partial charge |
| Sensor rounding / gaps | Separate live V/I/T rounding and consecutive gap controls |
| Charge features | Voltage statistics, timing, thermal rise/max, integrated observed charge, cycle count |
| Cycle-only baseline | Linear regression using only cycle index |
| Classical ML | Ridge, Random Forest and scikit-learn Gradient Boosting |
| Unseen-battery evaluation | Leave-one-battery-out; train-only imputation and scaling |
| Error reporting | SOH and RUL MAE/RMSE, per-battery folds and equal-weight battery averages |
| Robustness curves | Multiple severity levels, repeated independent sensor seeds |
| Minimum quality / break-even | Measured first crossing on sampled grid; user-defined tolerance |
| Degraded-data training | Clean versus clean-plus-degraded Random Forest on training batteries only |
| Robust feature groups | Group-removal retraining on clean / 50% charge data |
| Interpretation | Permutation importance on held-out cells, not training-set impurity importance |
| Streamlit demo | Five tabs, provenance audit, current features and JSON/CSV export |
| Reproducibility | Stable seeds, input/code fingerprints, dependency specification and regression tests |

## Data and target handling

NASA import advances the cycle index on discharges and pairs each discharge with
the preceding charge. The initial reference and EOL are computed from all valid
capacity records, including those without usable charge inputs. Such unpaired
records are excluded from model features, and a pending charge cannot be reused
for a later discharge. Other data must be converted to the documented common
CSV schema; native CALCE spreadsheets are not directly supported.

RUL is exact only if retirement was observed. Cells that never reach 80% SOH in
their recorded data are right-censored; no arbitrary retirement cycle is added.
Their RUL targets stay missing. Direct RUL models fit on observed targets from
training cells and RUL evaluation excludes unavailable labels. If no training
cell has an observed RUL target, the model returns unknown RUL. This treatment
does not implement a survival model; censoring can limit the representativeness
of the remaining direct-RUL sample.

SOH is not clipped to hide model extrapolation. RUL predictions are clipped at
zero. Project-defined health grades use ≥90% Good, 80–90% Moderate, and <80%
Below 80%. Grades are descriptive and do not constitute service instructions.

## Features and degradation

Inputs are charge time, voltage, current and temperature. Discharge capacity is
used only as a label. Integrated charge represents the retained charge segment,
not the discharge capacity. Missing voltage transitions remain missing; missing
values are imputed from training-cell medians. All-empty training columns use
the imputer's explicit zero fallback. Intervals longer than five median sample
intervals are excluded from charge integration; shorter gaps use trapezoidal
interpolation. This heuristic should itself be assessed on imported data.

Live controls permit early and upper time windows, rounded V/I/T measurements, random loss
and burst loss. Experimental sweeps isolate the four core factors. The noise
sweep links voltage/current/temperature standard deviations using declared
scales; rounding and bursts are not silently mixed into it. Decimation applies
by index, with native sample intervals disclosed in the audit.

## Experimental protocol

Four fixed model families are evaluated. Each fold holds out an entire battery.
Every preprocessing statistic and model uses only the remaining batteries.
Sensor degradations operate on inputs after labels are fixed. Models are reused
across severity levels rather than repeatedly refitted on identical clean data.
Augmented RF uses a reproducible degraded copy of training batteries alongside
their clean inputs. It never augments the held-out cell into the training set.

The primary reported metric is the equal-weight mean of per-battery metrics;
pooled metrics are also available from the evaluation API. SOH uses percentage
points and RUL uses cycles. Sensor-seed variation is not variation across new
batteries. All fold counts and the number of scored RUL labels are exported.

The minimum-quality summary records the last passing severity level before the
first tolerance failure, and the first tested level where error is at least the
cycle-only baseline error. It does not interpolate an exact continuous boundary.
Non-monotonic later improvements do not erase an earlier failure. Tolerances
must be justified before real experiments; default 3 SOH points and 15 cycles
are example project settings, not established reliability standards.

Feature-group ablation retrains RF after removing each group and evaluates both
clean and 50% early-charge input. Held-out permutation importance increases a
feature's disorder on test batteries and measures the SOH MAE increase. Feature
correlation can obscure importance; these analyses are descriptive.

## Reproducible deliverables

The experiment runner saves provenance and parameters in
`data/benchmark_summary.json`, clean fold metrics in `clean_folds.csv`, clean
out-of-fold predictions in `clean_predictions.csv`, repeated robustness results
in `robustness_repeats.csv`, and all robustness folds in `robustness_folds.csv`.
The JSON also includes augmentation, ablation and permutation results. Dataset
and source-code fingerprints guard against stale results in the dashboard.

Recompute from real measurements before reporting research accuracy. Synthetic
signals explicitly encode the ageing curve and are expected to be easier than
real measurements. No synthetic performance should be cited as NASA evidence.

## Validation and remaining work

Automated checks cover stable data generation, fixed labels, censored RUL,
MAT pairing, normalized CSV validation, numerical charge integration, missing
features, repeatable sensor degradation, nonmutation of raw records, battery
isolation, fold-local imputation, sampled boundaries and small-data rejection.
Streamlit interaction checks cover model changes and degradation controls.
The Windows batch script is prepared for Python 3.10+ but must be exercised on
a Windows machine; the execution environment used here is Linux.

Remaining research work: acquire and audit real NASA data, convert and evaluate
CALCE as a separate dataset, justify tolerances, explore richer degradation
grids and feature experiments, and collect more independent cells. Uncertainty
intervals, SHAP, LSTM and field/pack validation remain optional future extensions.
The three provided reports remain unchanged and are bundled as references.
