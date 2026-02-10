# Solar PV Predictive Maintenance API

AI-powered predictive maintenance for solar PV inverters using ML models and LLM-based diagnosis.

## API Endpoints

### POST /analyze
Accepts voltage and current sensor data, returns ML predictions and agent diagnosis.

**Request:**
```json
{
  "vdc1": [600.0, 601.0, 602.0],
  "idc1": [10.0, 10.1, 10.2]
}
```

**Response:**
```json
{
  "ml_output": {
    "asset_id": "PV_INVERTER_001",
    "failure_probability": 0.12,
    "expected_ttf_days": 450.5,
    "expected_rul_days": 9800.0,
    "confidence": 0.85
  },
  "agent_output": {
    "diagnosis": "...",
    "urgency": "Low",
    "recommended_action": "...",
    "justification": ["..."]
  }
}
```

## ML Pipeline
- **Anomaly Detection**: Isolation Forest + LSTM Autoencoder
- **Failure Forecasting**: XGBoost (Time-to-Failure + Failure Probability)
- **Agent Reasoning**: Gemini 2.5 Flash Lite via LangChain
