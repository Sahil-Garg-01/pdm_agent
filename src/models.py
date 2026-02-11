from pydantic import BaseModel

class SensorData(BaseModel):
    vdc1: list[float]
    idc1: list[float]
    api_key: str = None  # Optional Google API key for LLM features

class AnalysisResponse(BaseModel):
    ml_output: dict
    agent_output: dict