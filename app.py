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

app = FastAPI(title="Solar PV Predictive Maintenance API", version="1.0.0")

# Load ML models once on startup for production performance
ml_engine = MLEngine()

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
        logging.info(f"Processing request with {len(data.vdc1)} voltage and {len(data.idc1)} current data points")
        
        if len(data.vdc1) != len(data.idc1):
            raise HTTPException(status_code=400, detail="Voltage and current lists must have the same length")
        
        if len(data.vdc1) < 3:
            raise HTTPException(status_code=400, detail="Need at least 3 data points")
        
        # Repeat to make at least 100 points if needed
        raw_df = pd.DataFrame({
            "vdc1": (data.vdc1 * (100 // len(data.vdc1) + 1))[:100],
            "idc1": (data.idc1 * (100 // len(data.idc1) + 1))[:100]
        })
        
        # ML Inference
        phase2_output = ml_engine.predict_from_raw(raw_df)
        
        # Agent Reasoning (if API key provided)
        if data.api_key:
            try:
                request_agent = MaintenanceAgent(
                    api_key=data.api_key,
                    model_name="gemini-2.5-flash-lite",
                    temperature=0.0
                )
                agent_output = request_agent.run(phase2_output)
            except Exception as e:
                logging.warning(f"Agent initialization failed: {e}")
                agent_output = {
                    "diagnosis": "Agent initialization failed",
                    "urgency": "Unknown",
                    "recommended_action": "Check your Google API key",
                    "justification": [f"Error: {str(e)}"]
                }
        else:
            agent_output = {
                "diagnosis": "No API key provided - LLM features disabled",
                "urgency": "Unknown",
                "recommended_action": "Provide Google API key in request for AI diagnosis",
                "justification": ["Google API key required for maintenance reasoning"]
            }
        
        return AnalysisResponse(ml_output=phase2_output, agent_output=agent_output)
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Solar PV Predictive Maintenance API", "endpoint": "/analyze (POST)"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)