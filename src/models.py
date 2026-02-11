from pydantic import BaseModel

class SensorData(BaseModel):
    vdc1: list[float]
    idc1: list[float]
    api_key: str = None  # Optional Google API key for LLM features
    asset_id: str = None  # Optional asset identifier

class AnalysisResponse(BaseModel):
    ml_output: dict
    agent_output: dict