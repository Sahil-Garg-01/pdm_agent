import logging
from ml.inference import MLEngine
from agent.agent import MaintenanceAgent
from src.utils import validate_sensor_data, prepare_dataframe

logger = logging.getLogger(__name__)

class AnalysisService:
    """Service class for handling sensor data analysis logic."""
    
    def __init__(self, config):
        self.config = config
        self.ml_engine = MLEngine()

    def analyze(self, vdc1: list, idc1: list, api_key: str, asset_id: str = None) -> tuple:
        """Analyze sensor data and return ML and agent outputs."""
        logger.info(f"Complete analysis start - processing {len(vdc1)} data points")
        validate_sensor_data(vdc1, idc1)
        raw_df = prepare_dataframe(vdc1, idc1)
        ml_output = self.ml_engine.predict_from_raw(raw_df, asset_id)
        agent_output = self.get_agent_output(api_key, ml_output)
        
        logger.info("Complete analysis end")
        return ml_output, agent_output
    
    def get_agent_output(self, api_key: str, ml_output: dict) -> dict:
        """Get agent analysis if API key is provided, otherwise return no-key message."""
        if not api_key:
            logger.info("No API key provided - skipping agent analysis")
            return {
                "diagnosis": "No API key provided - LLM features disabled",
                "urgency": "Unknown",
                "recommended_action": "Provide Google API key in request for AI diagnosis",
                "justification": ["Google API key required for maintenance reasoning"]
            }
        
        try:
            logger.info("Agent analysis start")
            agent = MaintenanceAgent(
                api_key=api_key,
                model_name=self.config.MODEL_NAME,
                temperature=self.config.TEMPERATURE
            )
            result = agent.run(ml_output)
            logger.info("Agent analysis end")
            return result
        except Exception as e:
            logger.warning(f"Agent initialization failed: {e}")
            return {
                "diagnosis": "Agent initialization failed",
                "urgency": "Unknown",
                "recommended_action": "Check your Google API key",
                "justification": [f"Error: {str(e)}"]
            }