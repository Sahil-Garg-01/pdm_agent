import json
import os
import joblib
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from safetensors.torch import load_file

from ml.features import build_features
from ml.lstm_model import LSTMAutoencoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

class MLEngine:
    def __init__(self):
        with open(os.path.join(ARTIFACTS_DIR, "ml_config.json")) as f:
            self.config = json.load(f)

        self.feature_cols = self.config["feature_cols"]
        self.window = self.config["window"]
        self.seq_len = self.config["seq_len"]
        self.design_life_days = self.config["design_life_days"]

        # Load scaler from JSON
        with open(os.path.join(ARTIFACTS_DIR, "scaler.json"), "r") as f:
            params = json.load(f)
        self.scaler = StandardScaler()
        self.scaler.mean_ = np.array(params["mean"])
        self.scaler.scale_ = np.array(params["scale"])
        self.scaler.var_ = self.scaler.scale_ ** 2
        self.scaler.n_features_in_ = len(self.scaler.mean_)

        # Retrain IsolationForest at startup using saved training data
        self.iso = IsolationForest(
            n_estimators=200,
            contamination=0.05,
            random_state=42
        )
        # Load training data (scaled features from Colab) and fit
        train_data = pd.read_json(os.path.join(ARTIFACTS_DIR, "training_data.json"))
        self.iso.fit(train_data[self.feature_cols])

        # Load XGBoost from JSON
        import xgboost as xgb
        self.ttf_model = xgb.XGBRegressor()
        self.ttf_model.load_model(os.path.join(ARTIFACTS_DIR, "xgb_ttf.json"))
        self.fail_model = xgb.XGBClassifier()
        self.fail_model.load_model(os.path.join(ARTIFACTS_DIR, "xgb_fail.json"))

        # Load LSTM from safetensors
        self.lstm = LSTMAutoencoder(
            input_dim=len(self.feature_cols),
            hidden_dim=32
        )
        state_dict = load_file(os.path.join(ARTIFACTS_DIR, "lstm_autoencoder.safetensors"))
        self.lstm.load_state_dict(state_dict)
        self.lstm.eval()

    def predict_from_raw(self, raw_df: pd.DataFrame):
        # --- Feature engineering ---
        df = build_features(raw_df, self.window)
        df = df[self.feature_cols].dropna()

        if len(df) < self.seq_len:
            raise ValueError("Not enough data for LSTM sequence")

        # --- Scaling ---
        df_scaled = pd.DataFrame(
            self.scaler.transform(df),
            columns=self.feature_cols,
            index=df.index
        )

        # --- Isolation Forest anomaly ---
        df_scaled["anomaly_iforest"] = -self.iso.decision_function(df_scaled)

        # --- LSTM anomaly ---
        X = df_scaled[self.feature_cols].values
        X_seq = np.array([X[-self.seq_len:]])

        with torch.no_grad():
            recon = self.lstm(torch.tensor(X_seq, dtype=torch.float32))

        anomaly_lstm = float(((recon - torch.tensor(X_seq)) ** 2).mean())

        # --- Health (0–1) ---
        # Normalize anomaly_lstm (assuming max error ~1e6 from training)
        anomaly_norm = min(anomaly_lstm / 1e6, 1.0)
        health = max(0.0, 1.0 - anomaly_norm)

        # --- ML predictions ---
        latest_features = df_scaled[self.feature_cols].iloc[[-1]].copy()
        latest_features["anomaly_lstm"] = anomaly_lstm
        latest_features["health_index"] = health

        expected_ttf_days = float(
            self.ttf_model.predict(latest_features, validate_features=False)[0]
        )

        failure_probability = float(
            self.fail_model.predict_proba(latest_features, validate_features=False)[0][1]
        )

        # --- RUL ---
        expected_rul_days = float(health * self.design_life_days)

        # --- Confidence ---
        confidence = round(
            0.5 * abs(failure_probability - 0.5) * 2
            + 0.5 * health,
            2
        )

        return {
            "asset_id": "PV_INVERTER_001",
            "failure_probability": round(failure_probability, 2),
            "expected_ttf_days": round(expected_ttf_days, 1),
            "expected_rul_days": round(expected_rul_days, 1),
            "confidence": confidence
        }
