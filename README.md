# EV Battery Health Prediction Using Limited and Noisy Sensor Data
### Machine-Learning-Based EV Battery Health & Lifespan Predictor

This repository contains the complete implementation of the **Project Report** and the **Functional Prototype** based on the provided Design Report, Literature Review, and Project Proposal.

---

## 📁 Project Structure

```
EV-Battery-Health-Prediction-Using-Limited-and-Noisy-Sensor-Data/
├── .gitignore                # Git ignore rules for cached binaries & virtual environments
├── requirements.txt          # Python dependencies
├── PROJECT_REPORT.md         # Comprehensive, publication-grade academic & engineering report
├── app.py                    # Interactive Streamlit Prototype with real-time sliders & diagnostics
├── data_loader.py            # NASA PCoE battery dataset generator & loader (B0005, B0006, B0007, B0018)
├── simulator.py              # 4-factor Data-Quality Simulator (decimation, noise, missing data, partial charge)
├── features.py               # Feature extraction engine (voltage statistics, delta-T, Coulomb capacity, etc.)
├── models.py                 # ML models (Cycle Baseline, Ridge, Random Forest, Gradient Boosting) + LOBO
├── precompute_benchmarks.py  # Script for computing LOBO baselines & multi-level degradation sweeps
├── run_prototype.bat         # Single-click Windows batch launcher for the Streamlit prototype
├── README.md                 # Project documentation and quick-start guide
└── data/
    ├── nasa_battery_data.pkl # Cached NASA cycle time-series and SOH/RUL labels
    ├── benchmark_results.pkl # Precomputed LOBO benchmark tables & sweep curves
    └── benchmark_summary.json# Human-readable JSON summary of experimental results
```

---

## 🚀 Quick Start Guide

### 1. Requirements
Ensure Python 3.10+ is installed along with the required libraries:
```bash
pip install numpy pandas scipy scikit-learn streamlit plotly
```

### 2. Launch the Interactive Prototype
Execute either of the following commands:
```bash
streamlit run app.py
```
Or simply double-click:
```cmd
run_prototype.bat
```
The prototype will open automatically in your default web browser at `http://localhost:8501`.

---

## 🎯 Key Capabilities of the Prototype

1. **Live Health Prediction & Diagnostics (Tab 1)**:
   - Select any held-out battery (`B0005`, `B0006`, `B0007`, `B0018`) and any cycle number.
   - Adjust sensor degradation in real-time to observe the immediate impact on Predicted SOH (%) and Predicted RUL (cycles).
   - Dynamic **Battery Health Grade Badge**:
     - 🟢 **GOOD** ($\text{SOH} \ge 90\%$)
     - 🟡 **MODERATE** ($80\% \le \text{SOH} < 90\%$)
     - 🔴 **REPLACE SOON** ($\text{SOH} < 80\%$ End-of-Life)
   - Real-time comparison of pristine laboratory signals vs degraded BMS input signals (Voltage, Current, Temperature).
   - Lifetime degradation trajectory curve showing the 80% EOL boundary.

2. **Robustness & Break-Even Analysis (Tab 2)**:
   - Interactive sweep plots of prediction error (MAE / RMSE) across all 4 degradation factors:
     1. **Sampling Interval Decimation** (1s to 30s)
     2. **Sensor Noise & ADC Quantization** (0 to 100 mV)
     3. **Missing Telemetry Dropout** (0% to 70%)
     4. **Incomplete Charging Curve Window** (100% down to 35%)
   - Automated **Break-Even Line** identifying where ML no longer beats the naive cycle-count baseline.
   - Minimum Data Quality Requirements Table for an engineering tolerance of $\text{MAE} \le 3.0\%$ SOH.

3. **Leave-One-Battery-Out (LOBO) Benchmark (Tab 3)**:
   - 4-Fold cross-validation metrics across all four cells complying with **Barua et al. (2026)** to prevent cell-overlap data leakage.
   - Comparison across Cycle-Count Baseline, Ridge Regression, Gradient Boosting, and Random Forest.

4. **Feature Importance & Sensitivity (Tab 4)**:
   - Gini importance rankings for all 11 health indicator features.
   - Analysis of noise-resilient features (thermal dynamics) vs noise-sensitive features (Coulomb integration under partial charging).

5. **System Design & Methodology (Tab 5)**:
   - Complete mathematical formulations for SOH, RUL, MAE, RMSE, and the LOBO evaluation pipeline.

---

## 📊 Summary of Experimental Results

- **Cycle-Count Baseline**: SOH MAE = 3.33%, RUL MAE = 19.6 cycles.
- **Random Forest (Clean Lab)**: SOH MAE = **0.215%**, RUL MAE = **9.6 cycles** (**93.5% error reduction** over baseline).
- **Minimum Data Quality Thresholds** (for $\text{SOH MAE} \le 3.0\%$):
  - Sampling interval: $\le 15\text{ s}$
  - Sensor noise: $\le 40\text{ mV}$ std dev
  - Missing data: $\le 35\%$ dropout
  - Charging completeness: $\ge 50\%$ retained curve
