"""
Machine Learning Models & Leave-One-Battery-Out (LOBO) Validation Engine
Implements:
1. Cycle-Count-Only Predictor (Naive baseline)
2. Ridge Regression (L2 regularized linear model)
3. Random Forest Regressor (Nonlinear ensemble)
4. Gradient Boosting Regressor (Boosting ensemble)
5. Leave-One-Battery-Out (LOBO) Cross-Validation
6. Robustness Stress-Testing Sweeps across Degradation Factors
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from features import FEATURE_NAMES, extract_features_for_battery
from simulator import DataQualitySimulator

ML_MODEL_REGISTRY = {
    "Cycle Count Baseline": "cycle_only",
    "Ridge Regression": "ridge",
    "Random Forest": "rf",
    "Gradient Boosting": "gb"
}

class BatteryHealthModelWrapper:
    def __init__(self, model_type: str = "rf"):
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.feature_cols = [c for c in FEATURE_NAMES if c != "cycle"] if model_type != "cycle_only" else ["cycle"]
        
        if model_type == "cycle_only":
            self.model_soh = LinearRegression()
            self.model_rul = LinearRegression()
        elif model_type == "ridge":
            self.model_soh = Ridge(alpha=1.0)
            self.model_rul = Ridge(alpha=1.0)
        elif model_type == "rf":
            self.model_soh = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
            self.model_rul = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
        elif model_type == "gb":
            self.model_soh = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42)
            self.model_rul = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def fit(self, X: pd.DataFrame, y_soh: np.ndarray, y_rul: np.ndarray):
        X_sub = X[self.feature_cols].copy()
        if self.model_type != "cycle_only":
            X_scaled = self.scaler.fit_transform(X_sub)
        else:
            X_scaled = X_sub.values

        self.model_soh.fit(X_scaled, y_soh)
        self.model_rul.fit(X_scaled, y_rul)
        return self

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X_sub = X[self.feature_cols].copy()
        if self.model_type != "cycle_only":
            X_scaled = self.scaler.transform(X_sub)
        else:
            X_scaled = X_sub.values

        pred_soh = self.model_soh.predict(X_scaled)
        pred_rul = self.model_rul.predict(X_scaled)
        # RUL cannot be negative
        pred_rul = np.maximum(0.0, pred_rul)
        return pred_soh, pred_rul


def evaluate_lobo(dataset: Dict[str, Dict], model_type: str = "rf", test_simulator: DataQualitySimulator = None) -> Dict:
    """
    Executes 4-fold Leave-One-Battery-Out (LOBO) cross validation.
    Training is performed on clean data from 3 batteries, testing on the 4th held-out battery
    (optionally subjected to degradation via test_simulator).
    """
    batteries = list(dataset.keys()) # ['B0005', 'B0006', 'B0007', 'B0018']
    
    # 1. Pre-extract clean features for each battery
    clean_dfs = {}
    for b_id in batteries:
        clean_dfs[b_id] = extract_features_for_battery(dataset[b_id], simulator=None)

    fold_results = []
    all_y_soh_true = []
    all_y_soh_pred = []
    all_y_rul_true = []
    all_y_rul_pred = []

    for test_b_id in batteries:
        train_b_ids = [b for b in batteries if b != test_b_id]
        
        # Training set is always clean lab baseline
        train_df = pd.concat([clean_dfs[b] for b in train_b_ids], ignore_index=True)
        
        # Test set is extracted with test_simulator (clean or degraded)
        test_df = extract_features_for_battery(dataset[test_b_id], simulator=test_simulator)
        
        model = BatteryHealthModelWrapper(model_type=model_type)
        model.fit(
            train_df,
            train_df["soh"].values,
            train_df["rul"].values
        )
        
        pred_soh, pred_rul = model.predict(test_df)
        
        soh_mae = mean_absolute_error(test_df["soh"], pred_soh)
        soh_rmse = root_mean_squared_error(test_df["soh"], pred_soh)
        rul_mae = mean_absolute_error(test_df["rul"], pred_rul)
        rul_rmse = root_mean_squared_error(test_df["rul"], pred_rul)
        
        fold_results.append({
            "test_battery": test_b_id,
            "train_batteries": ", ".join(train_b_ids),
            "soh_mae": soh_mae,
            "soh_rmse": soh_rmse,
            "rul_mae": rul_mae,
            "rul_rmse": rul_rmse,
            "y_soh_true": test_df["soh"].values,
            "y_soh_pred": pred_soh,
            "y_rul_true": test_df["rul"].values,
            "y_rul_pred": pred_rul,
            "cycles": test_df["cycle"].values
        })
        
        all_y_soh_true.extend(test_df["soh"].values)
        all_y_soh_pred.extend(pred_soh)
        all_y_rul_true.extend(test_df["rul"].values)
        all_y_rul_pred.extend(pred_rul)

    overall_metrics = {
        "soh_mae": mean_absolute_error(all_y_soh_true, all_y_soh_pred),
        "soh_rmse": root_mean_squared_error(all_y_soh_true, all_y_soh_pred),
        "rul_mae": mean_absolute_error(all_y_rul_true, all_y_rul_pred),
        "rul_rmse": root_mean_squared_error(all_y_rul_true, all_y_rul_pred),
        "folds": fold_results
    }
    return overall_metrics


def run_robustness_sweep(dataset: Dict[str, Dict], factor_name: str, levels: list, model_type: str = "rf") -> pd.DataFrame:
    """
    Sweeps a specific degradation factor across defined levels to compute error progression.
    """
    records = []
    for lvl in levels:
        if factor_name == "sampling_step":
            sim = DataQualitySimulator(sampling_step=lvl)
            label = f"Step {lvl}x"
        elif factor_name == "noise_std_voltage":
            sim = DataQualitySimulator(noise_std_voltage=lvl, noise_std_current=lvl*2, noise_std_temp=lvl*20)
            label = f"{lvl*1000:.0f} mV"
        elif factor_name == "missing_rate":
            sim = DataQualitySimulator(missing_rate=lvl)
            label = f"{lvl*100:.0f}%"
        elif factor_name == "charge_window_pct":
            sim = DataQualitySimulator(charge_window_pct=lvl)
            label = f"{lvl*100:.0f}% window"
        else:
            raise ValueError(f"Unknown factor: {factor_name}")

        eval_res = evaluate_lobo(dataset, model_type=model_type, test_simulator=sim)
        records.append({
            "factor": factor_name,
            "level": lvl,
            "level_label": label,
            "soh_mae": eval_res["soh_mae"],
            "soh_rmse": eval_res["soh_rmse"],
            "rul_mae": eval_res["rul_mae"],
            "rul_rmse": eval_res["rul_rmse"],
        })
    return pd.DataFrame(records)
