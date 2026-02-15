# Solar PV Predictive Maintenance API

AI-powered predictive maintenance for solar PV inverters using ML models and LLM-based diagnosis.

## Features

- **Advanced Anomaly Detection**: Isolation Forest + LSTM Autoencoder with reconstruction error analysis
- **Comprehensive Health Monitoring**: Real-time health scoring, anomaly detection, and trend analysis
- **Fault Classification**: Multi-class fault type prediction (Normal, Short Circuit, Degradation, Open Circuit, Shadowing)
- **Predictive Maintenance**: XGBoost models for time-to-failure, remaining useful life, and failure probability
- **AI Diagnosis**: Gemini LLM provides detailed maintenance recommendations with urgency levels
- **REST API**: FastAPI-based service with automatic documentation and validation

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
    "health_score": 0.892,
    "anomaly_score": 0.0342,
    "health_trend_200step": -0.0123,
    "failure_probability": 0.12,
    "expected_ttf_days": 450.5,
    "expected_rul_days": 9800.0,
    "predicted_fault_type": "Normal",
    "fault_confidence": 0.78,
    "confidence": 0.85
  },
  "agent_output": {
    "system_health": {
      "status": "Good",
      "health_score_explanation": "Health score of 0.892 indicates good system performance at 89% of optimal",
      "anomaly_analysis": "Anomaly score of 0.0342 shows normal operation with minimal deviations",
      "trend_assessment": "Slight negative trend (-0.0123) suggests gradual performance decline to monitor"
    },
    "failure_risk": {
      "probability_assessment": "12% failure probability within 30 days represents low risk",
      "time_to_failure_analysis": "450 days until predicted failure allows time for scheduled maintenance",
      "remaining_life_summary": "9,800 days remaining useful life indicates good system longevity"
    },
    "fault_diagnosis": {
      "primary_fault": "System classified as 'Normal' with 78% confidence, no major faults detected",
      "confidence_interpretation": "85% overall confidence provides reliable analysis for decision making"
    },
    "maintenance_recommendations": {
      "urgency_level": "Low",
      "immediate_actions": [
        "Continue routine performance monitoring",
        "Schedule panel inspection within 3 months"
      ],
      "monitoring_schedule": "Next comprehensive inspection in 90 days",
      "preventive_measures": "Implement quarterly maintenance checks"
    },
    "key_insights": [
      "System health is good but showing early degradation signs",
      "No immediate action required, continue standard monitoring",
      "Schedule maintenance before 450 days to prevent issues"
    ]
  }
}
```

### Data Processing Pipeline

1. **Input Validation**: Ensures voltage/current/temperature arrays match and contain sufficient data points
2. **Data Preparation**: Filters active periods and downsamples for efficient processing
3. **Feature Engineering**: Creates 10 statistical features using rolling window analysis:
   - Voltage mean/standard deviation
   - Power mean/standard deviation  
   - Power delta and slope
   - Normalized efficiency
   - Temperature mean/standard deviation/delta
4. **ML Inference**: 
   - Anomaly detection via Isolation Forest + LSTM reconstruction error
   - Health scoring and trend analysis
   - Fault type classification (5 classes)
   - Time-to-failure and remaining useful life prediction
   - Failure probability calculation with weighted risk factors
5. **Agent Analysis**: LLM analyzes all ML metrics for comprehensive maintenance diagnosis (if API key provided)

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

