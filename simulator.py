"""
Data-Quality Simulator Module
Deliberately degrades clean laboratory sensor measurements to imitate real-world vehicle sensor conditions across 4 core factors:
1. Reduced Sampling Rate (Slower BMS logging: step sampling 1, 2, 5, 10, 30 readings)
2. Sensor Noise & Discretization Rounding (Gaussian noise added to V, I, T + rounding to sensor quantization)
3. Missing Data & Packet Dropouts (Random point drops 0-50% + burst packet gap losses)
4. Incomplete / Partial Charging (Truncating to partial charging intervals e.g. 100%, 75%, 50%, 30%)
"""

import numpy as np
import pandas as pd
from typing import Optional

class DataQualitySimulator:
    def __init__(
        self,
        sampling_step: int = 1,
        noise_std_voltage: float = 0.0,       # in Volts, e.g. 0.001 to 0.05V
        noise_std_current: float = 0.0,       # in Amperes, e.g. 0.005 to 0.2A
        noise_std_temp: float = 0.0,          # in °C, e.g. 0.05 to 2.0°C
        voltage_precision: Optional[float] = None, # e.g. 0.01V (10mV rounding)
        current_precision: Optional[float] = None, # e.g. 0.05A (50mA rounding)
        temp_precision: Optional[float] = None,    # e.g. 0.1°C or 0.5°C rounding
        missing_rate: float = 0.0,            # 0.0 to 0.5 (fraction of dropped readings)
        burst_drop_gap: bool = False,         # simulate consecutive packet drop
        charge_window_pct: float = 1.0,       # 1.0 = full curve, 0.5 = only first or upper 50%
        seed: Optional[int] = 42
    ):
        self.sampling_step = max(1, int(sampling_step))
        self.noise_std_voltage = float(noise_std_voltage)
        self.noise_std_current = float(noise_std_current)
        self.noise_std_temp = float(noise_std_temp)
        self.voltage_precision = voltage_precision
        self.current_precision = current_precision
        self.temp_precision = temp_precision
        self.missing_rate = float(np.clip(missing_rate, 0.0, 0.95))
        self.burst_drop_gap = burst_drop_gap
        self.charge_window_pct = float(np.clip(charge_window_pct, 0.1, 1.0))
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def apply(self, df_cycle: pd.DataFrame) -> pd.DataFrame:
        """
        Applies controlled degradation to raw cycle sensor timeseries.
        """
        df = df_cycle.copy()
        n = len(df)
        if n == 0:
            return df

        # 1. Incomplete / Partial Charging Window
        if self.charge_window_pct < 1.0:
            retained_len = max(10, int(n * self.charge_window_pct))
            # Real EV scenario: charging is terminated early or only the upper window is recorded
            df = df.iloc[:retained_len].copy()

        # 2. Reduced Sampling Rate (Downsampling / BMS decimation)
        if self.sampling_step > 1:
            df = df.iloc[::self.sampling_step].copy()

        if len(df) == 0:
            return df

        # 3. Missing Data / Dropped Packets
        if self.missing_rate > 0.0:
            keep_mask = self.rng.random(len(df)) >= self.missing_rate
            # Ensure at least 5 points are preserved
            if keep_mask.sum() < 5:
                keep_mask[:5] = True
            df = df[keep_mask].copy()

        if self.burst_drop_gap and len(df) > 30:
            # Simulate a continuous 10-20% gap loss (e.g. CAN bus congestion / telemetry drop)
            gap_size = int(len(df) * 0.15)
            gap_start = self.rng.integers(int(len(df) * 0.2), int(len(df) * 0.7))
            df = pd.concat([df.iloc[:gap_start], df.iloc[gap_start + gap_size:]]).copy()

        if len(df) == 0:
            return df

        # 4. Sensor Noise & Discretization Rounding
        # Voltage noise and ADC quantization
        if self.noise_std_voltage > 0.0:
            df["voltage"] = df["voltage"] + self.rng.normal(0, self.noise_std_voltage, size=len(df))
        if self.voltage_precision is not None and self.voltage_precision > 0:
            df["voltage"] = np.round(df["voltage"] / self.voltage_precision) * self.voltage_precision

        # Current noise and ADC quantization
        if self.noise_std_current > 0.0:
            df["current"] = df["current"] + self.rng.normal(0, self.noise_std_current, size=len(df))
        if self.current_precision is not None and self.current_precision > 0:
            df["current"] = np.round(df["current"] / self.current_precision) * self.current_precision

        # Temperature noise and ADC quantization
        if self.noise_std_temp > 0.0:
            df["temperature"] = df["temperature"] + self.rng.normal(0, self.noise_std_temp, size=len(df))
        if self.temp_precision is not None and self.temp_precision > 0:
            df["temperature"] = np.round(df["temperature"] / self.temp_precision) * self.temp_precision

        return df.reset_index(drop=True)
