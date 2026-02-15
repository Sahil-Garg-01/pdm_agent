# Solar PV Predictive Maintenance API

AI-powered predictive maintenance for solar PV inverters using ML models and LLM-based diagnosis.

## Features

- **Anomaly Detection**: Isolation Forest + LSTM Autoencoder
- **Failure Forecasting**: XGBoost models for time-to-failure and failure probability
- **AI Diagnosis**: Gemini LLM provides maintenance recommendations
- **REST API**: FastAPI-based service with automatic documentation

## Quick Start

### Installation

```bash
git clone https://github.com/Sahil-Garg-01/pdm_agent.git
cd pdm-agent
pip install -r requirements.txt
```

### Environment Setup

Set your Google API key for LLM features:
```bash
export GOOGLE_API_KEY=your_api_key_here
```

### Run

```bash
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

## API Usage

### POST /analyze

Analyzes sensor data and returns ML predictions with optional AI diagnosis.

**Request:**
```json
{
  "vdc1": [600.0, 601.0, 602.0],
  "idc1": [10.0, 10.1, 10.2],
  "pvt": [25.0, 25.1, 25.2],
  "api_key": "your_google_api_key_here",
  "asset_id": "PV_INVERTER_001"
}
```

**Parameters:**
- `vdc1`, `idc1`: Voltage and current sensor readings
- `pvt`: PV temperature readings (required)
- `api_key`: Optional Google API key for AI diagnosis
- `asset_id`: Optional asset identifier (auto-generated if not provided)

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
    "diagnosis": "Minor voltage fluctuations detected...",
    "urgency": "Low",
    "recommended_action": "Continue monitoring...",
    "justification": ["Voltage within range", "Current stable"]
  }
}
```

### Data Processing Pipeline

1. **Input Validation**: Ensures voltage/current arrays match and contain sufficient data points
2. **Data Preparation**: Pads input to 100 data points for consistent processing
3. **Feature Engineering**: Creates 10 statistical features using rolling window analysis:
   - Voltage mean/standard deviation
   - Power mean/standard deviation
   - Power delta and slope
   - Normalized efficiency
   - Temperature mean/standard deviation/delta
4. **ML Inference**: Processes features through anomaly detection and prediction models
5. **Agent Analysis**: LLM analyzes ML results for human-readable diagnosis (if API key provided)

## Configuration

- `GOOGLE_API_KEY`: Required for AI diagnosis features
- Models: Gemini 2.5 Flash Lite, XGBoost, LSTM Autoencoder
- Port: 7860 (configurable)

## Docker

```bash
docker build -t pdm-agent .
docker run -p 7860:7860 -e GOOGLE_API_KEY=your_key pdm-agent
```

## Testing

```bash
# Test the API
curl -X POST "http://localhost:7860/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "vdc1": [600.0, 601.0, 602.0],
    "idc1": [10.0, 10.1, 10.2],
    "pvt": [25.0, 25.1, 25.2],
    "api_key": "your_api_key",
    "asset_id": "PV_INVERTER_001"
  }'
```

## 📊 ML Models

### Anomaly Detection
- **Isolation Forest**: Unsupervised outlier detection
- **LSTM Autoencoder**: Sequence-based anomaly scoring

### Predictive Models
- **XGBoost Classifier**: Failure probability prediction
- **XGBoost Regressor**: Time-to-failure estimation

### LLM Agent
- **Gemini 2.5 Flash Lite**: Diagnostic reasoning and recommendations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## License

MIT License

