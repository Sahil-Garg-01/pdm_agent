# Solar PV Predictive Maintenance API

AI-powered predictive maintenance for solar photovoltaic (PV) inverters using advanced machine learning models and LLM-based diagnosis. The service analyzes sensor data (voltage, current, and temperature readings) to detect anomalies, forecast failure probabilities, estimate time-to-failure, and provide comprehensive health assessments. It supports optional AI-powered diagnosis via Gemini LLM for detailed maintenance recommendations and root cause analysis. The system includes data validation, feature engineering, ML inference with LSTM autoencoders and XGBoost models, fault classification, and agent reasoning, with modular architecture for scalability. Usage is focused on solar PV maintenance and predictive analytics.

## Features

- **Advanced Anomaly Detection**: Isolation Forest + LSTM Autoencoder with reconstruction error analysis and health scoring
- **Comprehensive Health Monitoring**: Real-time health scoring, anomaly detection, trend analysis over 200 data points
- **Fault Classification**: Multi-class fault type prediction (Normal, Short Circuit, Degradation, Open Circuit, Shadowing) with confidence scores
- **Predictive Maintenance**: XGBoost models for time-to-failure, remaining useful life, and failure probability with weighted risk assessment
- **AI Diagnosis**: Gemini LLM provides detailed maintenance recommendations with urgency levels and actionable insights
- **REST API**: FastAPI-based service with automatic OpenAPI documentation and JSON responses
- **Temperature Integration**: Incorporates PV temperature data for enhanced prediction accuracy
- **Modular Architecture**: Clean separation of ML inference, agent reasoning, and API layers

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
  "vdc1": [600.0,599.8,599.6,599.3,599.0,598.8,598.6,598.3,598.0,597.7,597.5,597.2,596.9,596.6,596.3,596.0,595.8,595.5,595.2,595.0,594.7,594.4,594.0,593.7,593.4,593.0,592.7,592.4,592.0,591.7,591.4,591.0,590.7,590.4,590.0,589.7,589.4,589.0,588.7,588.4,588.0,587.7,587.4,587.0,586.7,586.4,586.0,585.7,585.4,585.0,584.7,584.4,584.0,583.7,583.4,583.0,582.7,582.4],
  "idc1": [10.0,9.98,9.96,9.94,9.92,9.90,9.88,9.86,9.84,9.82,9.80,9.78,9.76,9.74,9.72,9.70,9.68,9.66,9.64,9.62,9.60,9.58,9.56,9.54,9.52,9.50,9.48,9.46,9.44,9.42,9.40,9.38,9.36,9.34,9.32,9.30,9.28,9.26,9.24,9.22,9.20,9.18,9.16,9.14,9.12,9.10,9.08,9.06,9.04,9.02,9.00,8.98,8.96,8.94,8.92,8.90,8.88,8.86],
  "pvt": [25.0,25.1,25.2,25.3,25.4,25.5,25.6,25.7,25.8,25.9,26.0,26.1,26.2,26.3,26.4,26.5,26.6,26.7,26.8,26.9,27.0,27.1,27.2,27.3,27.4,27.5,27.6,27.7,27.8,27.9,28.0,28.1,28.2,28.3,28.4,28.5,28.6,28.7,28.8,28.9,29.0,29.1,29.2,29.3,100,29.5,29.6,29.7,29.8,29.9,30.0,30.1,30.2,30.3,30.4,30.5,30.6,30.7],
  "api_key": "your_google_api_key_here",
  "asset_id": "PV_001"
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
    "asset_id": "PV_001",
    "health_score": 1.0,
    "anomaly_score": 50.6225,
    "health_trend_200step": 0.0,
    "failure_probability": 0.14,
    "expected_ttf_days": 3395.5,
    "expected_rul_days": 10957.4,
    "predicted_fault_type": "Normal",
    "fault_confidence": 0.89,
    "confidence": 0.86
  },
  "agent_output": {
    "system_health": {
      "status": "Excellent",
      "health_score_explanation": "A health score of 1.0 indicates the PV system is operating at its peak performance and is in excellent condition.",
      "anomaly_analysis": "An anomaly score of 50.62 indicates a moderate level of deviation from normal operating parameters, suggesting some minor irregularities that warrant attention but are not immediately critical.",
      "trend_assessment": "A health trend of 0.0 over the last 200 steps signifies that the system's health has remained stable and has not degraded recently."
    },
    "failure_risk": {
      "probability_assessment": "A failure probability of 0.14 suggests a low likelihood of failure in the near future.",
      "time_to_failure_analysis": "An expected time to failure of 3395.5 days indicates that, based on current trends, the system is projected to operate without significant failure for over 9 years.",
      "remaining_life_summary": "An expected remaining useful life of 10957.4 days suggests the system has a substantial operational lifespan ahead, estimated at around 30 years."
    },
    "fault_diagnosis": {
      "primary_fault": "The predicted fault type is 'Normal' with a high confidence of 0.89, meaning the system is currently operating as expected with no identified faults.",
      "confidence_interpretation": "An overall confidence of 0.86 indicates a high degree of certainty in the current assessment of the system's health and operational status."
    },
    "maintenance_recommendations": {
      "urgency_level": "Low",
      "immediate_actions": [
        "Continue routine visual inspections of panels and connections.",
        "Review historical anomaly data for any recurring patterns."
      ],
      "monitoring_schedule": "Continue standard monitoring, with a detailed performance review scheduled in 6 months.",
      "preventive_measures": "Ensure regular cleaning of solar panels to maintain optimal energy generation."
    },
    "key_insights": [
      "The PV system is currently in excellent health with a very low risk of immediate failure.",
      "While the anomaly score is moderate, the 'Normal' fault prediction and high confidence suggest these are minor deviations.",
      "The system has a long projected lifespan, indicating good overall asset management and performance."
    ]
  }
}
```

### Data Processing Pipeline

1. **Input Validation**: Ensures voltage/current/temperature arrays match and contain sufficient data points (minimum 30 for LSTM sequence analysis)
2. **Data Preparation**: Filters active power periods (removes low-power data using 5% power threshold) and downsamples data (10:1 ratio) for efficient processing
3. **Feature Engineering**: Creates 10 statistical features using rolling window analysis (50-point windows):
   - Voltage metrics: mean, standard deviation
   - Power metrics: mean, standard deviation, delta, slope (calculated via linear regression)
   - Efficiency: normalized power conversion efficiency (power / (voltage × current))
   - Temperature metrics: mean, standard deviation, delta
4. **ML Inference Pipeline**:
   - **Anomaly Detection**: Isolation Forest for statistical outliers + LSTM Autoencoder reconstruction error analysis
   - **Health Scoring**: Normalized health index (0-1) based on anomaly levels and reconstruction errors
   - **Trend Analysis**: Health change calculation over last 200 data points for degradation monitoring
   - **Fault Classification**: XGBoost multi-class model predicting 5 fault types (Normal, Short Circuit, Degradation, Open Circuit, Shadowing)
   - **Predictive Modeling**: XGBoost regression models for time-to-failure and failure probability estimation
   - **Risk Assessment**: Weighted failure probability combining anomaly risk (35%), health risk (30%), TTF risk (20%), and trend risk (15%)
5. **Agent Analysis**: Gemini LLM analyzes all ML outputs for human-readable maintenance diagnosis with detailed parameter explanations and actionable recommendations

## Configuration

- `GOOGLE_API_KEY`: Required for AI diagnosis features (optional)
- **ML Models**: 
  - LSTM Autoencoder (10 features, 32 hidden dims, 30 seq length)
  - XGBoost models (200 estimators, max_depth 5, learning_rate 0.05)
  - Isolation Forest (200 estimators, contamination 0.05)
- **Data Processing**: 50-point rolling windows, 10:1 downsampling, 30-step sequences
- **Design Life**: 30 years (10,958 days) baseline for RUL calculations
- **Port**: 7860 (configurable)

## Docker

```bash
docker build -t pdm-agent .
docker run -p 7860:7860 -e GOOGLE_API_KEY=your_key pdm-agent
```

## Testing

```bash
# Test the API with comprehensive sensor data
curl -X POST "http://localhost:7860/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "vdc1": [600.0,599.8,599.6,599.3,599.0,598.8,598.6,598.3,598.0,597.7,597.5,597.2,596.9,596.6,596.3,596.0,595.8,595.5,595.2,595.0,594.7,594.4,594.0,593.7,593.4,593.0,592.7,592.4,592.0,591.7,591.4,591.0,590.7,590.4,590.0,589.7,589.4,589.0,588.7,588.4,588.0,587.7,587.4,587.0,586.7,586.4,586.0,585.7,585.4,585.0,584.7,584.4,584.0,583.7,583.4,583.0,582.7,582.4],
    "idc1": [10.0,9.98,9.96,9.94,9.92,9.90,9.88,9.86,9.84,9.82,9.80,9.78,9.76,9.74,9.72,9.70,9.68,9.66,9.64,9.62,9.60,9.58,9.56,9.54,9.52,9.50,9.48,9.46,9.44,9.42,9.40,9.38,9.36,9.34,9.32,9.30,9.28,9.26,9.24,9.22,9.20,9.18,9.16,9.14,9.12,9.10,9.08,9.06,9.04,9.02,9.00,8.98,8.96,8.94,8.92,8.90,8.88,8.86],
    "pvt": [25.0,25.1,25.2,25.3,25.4,25.5,25.6,25.7,25.8,25.9,26.0,26.1,26.2,26.3,26.4,26.5,26.6,26.7,26.8,26.9,27.0,27.1,27.2,27.3,27.4,27.5,27.6,27.7,27.8,27.9,28.0,28.1,28.2,28.3,28.4,28.5,28.6,28.7,28.8,28.9,29.0,29.1,29.2,29.3,100,29.5,29.6,29.7,29.8,29.9,30.0,30.1,30.2,30.3,30.4,30.5,30.6,30.7],
    "api_key": "your_google_api_key_here",
    "asset_id": "PV_001"
  }'
```

## 📊 ML Models

### Anomaly Detection
- **Isolation Forest**: Unsupervised outlier detection with 200 estimators, contamination factor 0.05
- **LSTM Autoencoder**: Sequence-based anomaly scoring with 10 input features, 32 hidden dimensions, reconstruction error analysis

### Predictive Models
- **XGBoost Classifier**: Failure probability prediction (binary classification)
- **XGBoost Regressor**: Time-to-failure estimation (regression)
- **XGBoost Multi-class Classifier**: Fault type classification (5 classes: Normal, Short Circuit, Degradation, Open Circuit, Shadowing)

### Feature Engineering
- **10 Statistical Features**: Rolling window analysis (50-point windows) including voltage, power, efficiency, and temperature metrics
- **Sequence Processing**: 30-step sequences for LSTM analysis with downsampling (10:1 ratio)

### LLM Agent
- **Gemini 2.5 Flash Lite**: Diagnostic reasoning and maintenance recommendations with structured output format

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## License

MIT License

