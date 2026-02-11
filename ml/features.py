import numpy as np
import pandas as pd

def build_features(df, window):
    df = df.copy()
    df["pdc1"] = df["vdc1"] * df["idc1"]
    df["vdc_mean"] = df["vdc1"].rolling(window).mean()
    df["vdc_std"] = df["vdc1"].rolling(window).std()
    df["pdc_mean"] = df["pdc1"].rolling(window).mean()
    df["pdc_std"] = df["pdc1"].rolling(window).std()
    df["pdc_delta"] = df["pdc1"].diff()
    df["pdc_slope"] = df["pdc1"].rolling(window).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
    )
    df["efficiency"] = df["pdc1"] / (df["vdc1"] * df["idc1"] + 1e-6)
    df["efficiency_norm"] = df["efficiency"] / df["efficiency"].rolling(window).mean()
    return df