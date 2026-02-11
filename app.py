from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
import logging
from dotenv import load_dotenv
from ml.inference import MLEngine
from agent.agent import MaintenanceAgent

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Solar PV Predictive Maintenance API", version="1.0.0")

# Load ML models once on startup for production performance
ml_engine = MLEngine()

# ============ Helper Functions ============

def validate_sensor_data(vdc1: list, idc1: list) -> None:
    """Validate sensor data consistency. Raises HTTPException on error."""
    if len(vdc1) != len(idc1):
        raise HTTPException(status_code=400, detail="Voltage and current lists must have the same length")
    if len(vdc1) < 3:
        raise HTTPException(status_code=400, detail="Need at least 3 data points")

def prepare_dataframe(vdc1: list, idc1: list) -> pd.DataFrame:
    """Prepare sensor data for ML inference by padding to 100 points."""
    return pd.DataFrame({
        "vdc1": (vdc1 * (100 // len(vdc1) + 1))[:100],
        "idc1": (idc1 * (100 // len(idc1) + 1))[:100]
    })

def get_agent_output(api_key: str, ml_output: dict) -> dict:
    """Get agent analysis if API key is provided, otherwise return no-key message."""
    if not api_key:
        return {
            "diagnosis": "No API key provided - LLM features disabled",
            "urgency": "Unknown",
            "recommended_action": "Provide Google API key in request for AI diagnosis",
            "justification": ["Google API key required for maintenance reasoning"]
        }
    
    try:
        agent = MaintenanceAgent(
            api_key=api_key,
            model_name="gemini-2.5-flash-lite",
            temperature=0.0
        )
        return agent.run(ml_output)
    except Exception as e:
        logger.warning(f"Agent initialization failed: {e}")
        return {
            "diagnosis": "Agent initialization failed",
            "urgency": "Unknown",
            "recommended_action": "Check your Google API key",
            "justification": [f"Error: {str(e)}"]
        }

class SensorData(BaseModel):
    vdc1: list[float]
    idc1: list[float]
    api_key: str = None  # Optional Google API key for LLM features

class AnalysisResponse(BaseModel):
    ml_output: dict
    agent_output: dict

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_sensor_data(data: SensorData):
    try:
        logger.info(f"Processing request with {len(data.vdc1)} voltage and {len(data.idc1)} current data points")
        
        # Validate input
        validate_sensor_data(data.vdc1, data.idc1)
        
        # Prepare data and run ML inference
        raw_df = prepare_dataframe(data.vdc1, data.idc1)
        ml_output = ml_engine.predict_from_raw(raw_df)
        
        # Get agent analysis
        agent_output = get_agent_output(data.api_key, ml_output)
        
        return AnalysisResponse(ml_output=ml_output, agent_output=agent_output)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Solar PV Predictive Maintenance API", "endpoint": "/analyze (POST)"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)