"""
Precompute Benchmarks & Robustness Sweeps
Runs LOBO evaluation on clean data for all 4 models and sweeps all 4 degradation factors.
Saves precomputed data to data/benchmark_results.pkl and json for fast dashboard loading.
"""

import os
import json
import pickle
import pandas as pd
from data_loader import load_battery_data
from simulator import DataQualitySimulator
from models import evaluate_lobo, BatteryHealthModelWrapper, ML_MODEL_REGISTRY
from features import FEATURE_NAMES, extract_features_for_battery

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BENCHMARK_FILE = os.path.join(DATA_DIR, "benchmark_results.pkl")

def run_all_benchmarks():
    print("Loading battery dataset...")
    dataset = load_battery_data()

    # 1. Clean LOBO Baseline across all models
    print("Evaluating Clean LOBO Baselines across models...")
    clean_benchmarks = {}
    for label, m_key in ML_MODEL_REGISTRY.items():
        print(f"  Evaluating {label} ({m_key})...")
        res = evaluate_lobo(dataset, model_type=m_key, test_simulator=None)
        clean_benchmarks[label] = {
            "soh_mae": float(res["soh_mae"]),
            "soh_rmse": float(res["soh_rmse"]),
            "rul_mae": float(res["rul_mae"]),
            "rul_rmse": float(res["rul_rmse"]),
            "folds": [
                {
                    "test_battery": f["test_battery"],
                    "soh_mae": float(f["soh_mae"]),
                    "soh_rmse": float(f["soh_rmse"]),
                    "rul_mae": float(f["rul_mae"]),
                    "rul_rmse": float(f["rul_rmse"])
                }
                for f in res["folds"]
            ]
        }

    # 2. Robustness Sweeps across the 4 factors
    print("Running Robustness Degradation Sweeps...")
    sweeps = {}

    # Factor 1: Sampling Interval (Decimation)
    print("  Sweeping Sampling Interval...")
    sampling_steps = [1, 2, 5, 10, 20, 30]
    sweep_samp = []
    for step in sampling_steps:
        sim = DataQualitySimulator(sampling_step=step)
        eval_rf = evaluate_lobo(dataset, model_type="rf", test_simulator=sim)
        eval_ridge = evaluate_lobo(dataset, model_type="ridge", test_simulator=sim)
        sweep_samp.append({
            "step": step,
            "label": f"{step}s interval ({step}x decimation)",
            "rf_soh_mae": float(eval_rf["soh_mae"]),
            "rf_rul_mae": float(eval_rf["rul_mae"]),
            "ridge_soh_mae": float(eval_ridge["soh_mae"]),
            "ridge_rul_mae": float(eval_ridge["rul_mae"]),
            "cycle_baseline_soh_mae": clean_benchmarks["Cycle Count Baseline"]["soh_mae"],
            "cycle_baseline_rul_mae": clean_benchmarks["Cycle Count Baseline"]["rul_mae"]
        })
    sweeps["sampling_interval"] = sweep_samp

    # Factor 2: Sensor Noise & Rounding
    print("  Sweeping Sensor Noise & Precision...")
    noise_levels = [0.0, 0.005, 0.015, 0.030, 0.050, 0.100] # Volts
    sweep_noise = []
    for nv in noise_levels:
        sim = DataQualitySimulator(
            noise_std_voltage=nv,
            noise_std_current=nv * 2.0,
            noise_std_temp=nv * 25.0,
            voltage_precision=nv if nv > 0 else None
        )
        eval_rf = evaluate_lobo(dataset, model_type="rf", test_simulator=sim)
        eval_ridge = evaluate_lobo(dataset, model_type="ridge", test_simulator=sim)
        sweep_noise.append({
            "noise_v_std": nv,
            "label": f"{nv*1000:.0f} mV noise",
            "rf_soh_mae": float(eval_rf["soh_mae"]),
            "rf_rul_mae": float(eval_rf["rul_mae"]),
            "ridge_soh_mae": float(eval_ridge["soh_mae"]),
            "ridge_rul_mae": float(eval_ridge["rul_mae"]),
            "cycle_baseline_soh_mae": clean_benchmarks["Cycle Count Baseline"]["soh_mae"],
            "cycle_baseline_rul_mae": clean_benchmarks["Cycle Count Baseline"]["rul_mae"]
        })
    sweeps["sensor_noise"] = sweep_noise

    # Factor 3: Missing Data & Packet Drop
    print("  Sweeping Missing Data Dropout...")
    missing_rates = [0.0, 0.05, 0.15, 0.30, 0.50, 0.70]
    sweep_miss = []
    for mr in missing_rates:
        sim = DataQualitySimulator(missing_rate=mr, burst_drop_gap=(mr >= 0.30))
        eval_rf = evaluate_lobo(dataset, model_type="rf", test_simulator=sim)
        eval_ridge = evaluate_lobo(dataset, model_type="ridge", test_simulator=sim)
        sweep_miss.append({
            "missing_rate": mr,
            "label": f"{int(mr*100)}% missing",
            "rf_soh_mae": float(eval_rf["soh_mae"]),
            "rf_rul_mae": float(eval_rf["rul_mae"]),
            "ridge_soh_mae": float(eval_ridge["soh_mae"]),
            "ridge_rul_mae": float(eval_ridge["rul_mae"]),
            "cycle_baseline_soh_mae": clean_benchmarks["Cycle Count Baseline"]["soh_mae"],
            "cycle_baseline_rul_mae": clean_benchmarks["Cycle Count Baseline"]["rul_mae"]
        })
    sweeps["missing_data"] = sweep_miss

    # Factor 4: Incomplete Charging Window
    print("  Sweeping Charging Completeness...")
    charge_windows = [1.0, 0.85, 0.70, 0.50, 0.35]
    sweep_charge = []
    for cw in charge_windows:
        sim = DataQualitySimulator(charge_window_pct=cw)
        eval_rf = evaluate_lobo(dataset, model_type="rf", test_simulator=sim)
        eval_ridge = evaluate_lobo(dataset, model_type="ridge", test_simulator=sim)
        sweep_charge.append({
            "charge_window": cw,
            "label": f"{int(cw*100)}% complete",
            "rf_soh_mae": float(eval_rf["soh_mae"]),
            "rf_rul_mae": float(eval_rf["rul_mae"]),
            "ridge_soh_mae": float(eval_ridge["soh_mae"]),
            "ridge_rul_mae": float(eval_ridge["rul_mae"]),
            "cycle_baseline_soh_mae": clean_benchmarks["Cycle Count Baseline"]["soh_mae"],
            "cycle_baseline_rul_mae": clean_benchmarks["Cycle Count Baseline"]["rul_mae"]
        })
    sweeps["charging_completeness"] = sweep_charge

    # 3. Feature Importances from RF
    print("Computing Feature Importances...")
    # Train RF on all clean data to get feature importance
    all_clean = pd.concat([extract_features_for_battery(dataset[b]) for b in dataset.keys()], ignore_index=True)
    rf_model = BatteryHealthModelWrapper(model_type="rf")
    rf_model.fit(all_clean, all_clean["soh"].values, all_clean["rul"].values)
    feat_names = rf_model.feature_cols
    importances_soh = rf_model.model_soh.feature_importances_
    importances_rul = rf_model.model_rul.feature_importances_

    feature_rankings = []
    for f_name, imp_s, imp_r in zip(feat_names, importances_soh, importances_rul):
        feature_rankings.append({
            "feature": f_name,
            "importance_soh": float(imp_s),
            "importance_rul": float(imp_r)
        })
    feature_rankings.sort(key=lambda x: x["importance_soh"], reverse=True)

    results = {
        "clean_benchmarks": clean_benchmarks,
        "sweeps": sweeps,
        "feature_rankings": feature_rankings
    }

    with open(BENCHMARK_FILE, "wb") as f:
        pickle.dump(results, f)
    with open(os.path.join(DATA_DIR, "benchmark_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("Benchmarks and Robustness sweeps completed successfully!")
    return results

if __name__ == "__main__":
    run_all_benchmarks()
