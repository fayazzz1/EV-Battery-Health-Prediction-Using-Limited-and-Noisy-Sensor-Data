"""
Feature Extraction Module
Derives health-sensitive features from each charge cycle:
1. Voltage statistics: mean, std, min, max, skew
2. Charging time / CC duration
3. Maximum temperature & Temperature rise (ΔT)
4. Charge-based capacity (Coulomb counting: ∫ I dt / 3600)
5. Voltage transition duration (time to move from 3.8V to 4.1V)
6. Cycle index
"""

import numpy as np
import pandas as pd
from scipy.stats import skew

FEATURE_NAMES = [
    "cycle",
    "charging_time_sec",
    "v_mean",
    "v_std",
    "v_min",
    "v_max",
    "v_skew",
    "t_max",
    "t_rise",
    "coulomb_capacity_ah",
    "v_trans_time_sec"
]

def extract_cycle_features(df_cycle: pd.DataFrame, cycle_num: int) -> dict:
    """
    Extracts numerical health indicators from raw or degraded cycle timeseries.
    Handles irregular time steps, missing data, and noisy readings gracefully.
    """
    if len(df_cycle) < 3:
        # Fallback for degenerate cycles
        return {
            "cycle": cycle_num,
            "charging_time_sec": 0.0,
            "v_mean": 3.7,
            "v_std": 0.1,
            "v_min": 3.6,
            "v_max": 4.2,
            "v_skew": 0.0,
            "t_max": 25.0,
            "t_rise": 0.0,
            "coulomb_capacity_ah": 0.0,
            "v_trans_time_sec": 0.0
        }

    t = df_cycle["time"].values
    v = df_cycle["voltage"].values
    i = df_cycle["current"].values
    temp = df_cycle["temperature"].values

    charging_time = float(t[-1] - t[0]) if len(t) > 1 else 0.0

    # Voltage statistics
    v_mean = float(np.mean(v))
    v_std = float(np.std(v))
    v_min = float(np.min(v))
    v_max = float(np.max(v))
    try:
        v_sk = float(skew(v))
        if np.isnan(v_sk):
            v_sk = 0.0
    except Exception:
        v_sk = 0.0

    # Temperature statistics
    t_max = float(np.max(temp))
    t_min = float(np.min(temp))
    t_rise = float(max(0.0, t_max - t_min))

    # Charge-based capacity via numerical integration (trapezoidal rule)
    if len(t) > 1:
        # Coulomb counting: ∫ I(t) dt in Ampere-hours (Ah)
        # Ensure positive charging current
        coulomb_ah = float(np.trapezoid(np.maximum(i, 0.0), t) / 3600.0)
    else:
        coulomb_ah = 0.0

    # Time to move between voltage thresholds: e.g. 3.8V to 4.1V
    # As battery degrades, internal resistance rises and capacity shrinks,
    # causing this transition time to shorten significantly.
    mask_38 = v >= 3.80
    mask_41 = v >= 4.10
    if np.any(mask_38) and np.any(mask_41):
        t_38 = t[np.argmax(mask_38)]
        t_41 = t[np.argmax(mask_41)]
        v_trans_time = float(max(0.0, t_41 - t_38))
    else:
        v_trans_time = 0.0

    return {
        "cycle": cycle_num,
        "charging_time_sec": charging_time,
        "v_mean": v_mean,
        "v_std": v_std,
        "v_min": v_min,
        "v_max": v_max,
        "v_skew": v_sk,
        "t_max": t_max,
        "t_rise": t_rise,
        "coulomb_capacity_ah": coulomb_ah,
        "v_trans_time_sec": v_trans_time
    }

def extract_features_for_battery(battery_data: dict, simulator=None) -> pd.DataFrame:
    """
    Extracts features for all cycles of a battery, optionally applying a DataQualitySimulator.
    """
    summary_df = battery_data["summary"]
    timeseries_dict = battery_data["timeseries"]
    records = []

    for _, row in summary_df.iterrows():
        k = int(row["cycle"])
        df_raw = timeseries_dict[k]
        if simulator is not None:
            df_proc = simulator.apply(df_raw)
        else:
            df_proc = df_raw

        feats = extract_cycle_features(df_proc, k)
        feats["battery_id"] = row["battery_id"]
        feats["capacity_ah"] = row["capacity_ah"]
        feats["soh"] = row["soh"]
        feats["rul"] = row["rul"]
        records.append(feats)

    return pd.DataFrame(records)
