import json
import os
import joblib
import torch
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from safetensors.torch import load_file

from ml.features import build_features
from ml.lstm_model import LSTMAutoencoder
from src.config import MLConfig

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")


class MLEngine:
    def __init__(self):
        logger.info("Initializing ML Engine...")
        self._load_ml_config()
        self._load_scaler()
        self._load_isolation_forest()
        self._load_xgboost_models()
        self._load_lstm_model()
        logger.info("ML Engine initialized successfully")

    def _load_ml_config(self):
        """Load ML configuration from config."""
        config = MLConfig.load()
        self.feature_cols = config["feature_cols"]
        self.window = config["window"]
        self.seq_len = config["seq_len"]
        self.design_life_days = config["design_life_days"]
        self.lstm_input_dim = config.get("lstm_input_dim", len(self.feature_cols))
        self.lstm_hidden_dim = config.get("lstm_hidden_dim", 32)

    def _load_scaler(self):
        """Load and reconstruct StandardScaler from JSON."""
        with open(os.path.join(ARTIFACTS_DIR, "scaler.json"), "r") as f:
            params = json.load(f)
        self.scaler = StandardScaler()
        self.scaler.mean_ = np.array(params["mean"])
        self.scaler.scale_ = np.array(params["scale"])
        self.scaler.var_ = self.scaler.scale_ ** 2
        self.scaler.n_features_in_ = len(self.scaler.mean_)

    def _load_isolation_forest(self):
        """Load and retrain IsolationForest using saved training data."""
        self.iso = IsolationForest(
            n_estimators=200,
            contamination=0.05,
            random_state=42
        )
        train_data = pd.read_json(os.path.join(ARTIFACTS_DIR, "training_data.json"))
        self.iso.fit(train_data[self.feature_cols])

    def _load_xgboost_models(self):
        """Load XGBoost models from JSON artifacts."""
        import xgboost as xgb
        self.ttf_model = xgb.XGBRegressor()
        self.ttf_model.load_model(os.path.join(ARTIFACTS_DIR, "xgb_ttf.json"))
        self.fail_model = xgb.XGBClassifier()
        self.fail_model.load_model(os.path.join(ARTIFACTS_DIR, "xgb_fail.json"))

    def _load_lstm_model(self):
        """Load LSTM autoencoder from safetensors."""
        self.lstm = LSTMAutoencoder(
            input_dim=self.lstm_input_dim,
            hidden_dim=self.lstm_hidden_dim
        )
        state_dict = load_file(os.path.join(ARTIFACTS_DIR, "lstm_autoencoder.safetensors"))
        self.lstm.load_state_dict(state_dict)
        self.lstm.eval()

    def _compute_anomalies(self, df_scaled: pd.DataFrame) -> tuple:
        """Compute anomaly scores from LSTM and IsolationForest.
        Returns: (anomaly_lstm, health) tuple
        """
        df_scaled["anomaly_iforest"] = -self.iso.decision_function(df_scaled)
        X = df_scaled[self.feature_cols].values
        X_seq = np.array([X[-self.seq_len:]])
        with torch.no_grad():
            recon = self.lstm(torch.tensor(X_seq, dtype=torch.float32))
        anomaly_lstm = float(((recon - torch.tensor(X_seq)) ** 2).mean())
        anomaly_norm = min(anomaly_lstm / 1e6, 1.0)
        health = max(0.0, 1.0 - anomaly_norm)
        return anomaly_lstm, health

    def _make_predictions(self, df_scaled: pd.DataFrame, anomaly_lstm: float, health: float) -> dict:
        """Make TTF and failure probability predictions.
        Returns: Dictionary with ttf, failure_prob, and rul predictions
        """
        latest_features = df_scaled[self.feature_cols].iloc[[-1]].copy()
        latest_features["anomaly_lstm"] = anomaly_lstm
        latest_features["health_index"] = health
        expected_ttf_days = float(
            self.ttf_model.predict(latest_features, validate_features=False)[0]
        )
        failure_probability = float(
            self.fail_model.predict_proba(latest_features, validate_features=False)[0][1]
        )
        expected_rul_days = float(health * self.design_life_days)
        confidence = round(0.5 * abs(failure_probability - 0.5) * 2 + 0.5 * health, 2)
        return {
            "ttf_days": expected_ttf_days,
            "failure_prob": failure_probability,
            "rul_days": expected_rul_days,
            "confidence": confidence
        }

    def predict_from_raw(self, raw_df: pd.DataFrame, asset_id: str = None):
        logger.info("ML analysis start")
        df = build_features(raw_df, self.window)
        df = df[self.feature_cols].dropna()
        if len(df) < self.seq_len:
            raise ValueError("Not enough data for LSTM sequence")
        df_scaled = pd.DataFrame(
            self.scaler.transform(df), columns=self.feature_cols, index=df.index
        )
        anomaly_lstm, health = self._compute_anomalies(df_scaled)
        predictions = self._make_predictions(df_scaled, anomaly_lstm, health)
        
        # Use provided asset_id or generate default
        if asset_id is None:
            import uuid
            asset_id = f"Solar_Panel_{str(uuid.uuid4())[:8]}"
        
        logger.info("ML analysis end")
        return {
            "asset_id": asset_id,
            "failure_probability": round(predictions["failure_prob"], 2),
            "expected_ttf_days": round(predictions["ttf_days"], 1),
            "expected_rul_days": round(predictions["rul_days"], 1),
            "confidence": predictions["confidence"]
        }