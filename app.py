from fastapi import FastAPI, HTTPException
import logging
from src.config import Config
from src.models import SensorData, AnalysisResponse
from src.services import AnalysisService
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = Config()
app = FastAPI(title=config.APP_TITLE, version=config.APP_VERSION)
service = AnalysisService(config)

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_sensor_data(data: SensorData):
    try:
        logger.info(f"Processing request with {len(data.vdc1)} voltage and {len(data.idc1)} current data points")
        ml_output, agent_output = service.analyze(data.vdc1, data.idc1, data.api_key, data.asset_id)
        return AnalysisResponse(ml_output=ml_output, agent_output=agent_output)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Solar PV Predictive Maintenance", "endpoint": "/analyze (POST)"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)