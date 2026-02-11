import json
import os
from typing import Dict, Any


class AppConfig:
    """Application and API configuration."""

    # LLM Configuration
    MODEL_NAME: str = "gemini-2.5-flash-lite"
    TEMPERATURE: float = 0.0

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 7860

    # Application Metadata
    APP_TITLE: str = "Solar PV Predictive Maintenance API"
    APP_VERSION: str = "1.0.0"


class MLConfig:
    """ML model configuration from ml/artifacts/ml_config.json."""

    @staticmethod
    def load() -> Dict[str, Any]:
        """Load and return ML configuration."""
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "ml", "artifacts", "ml_config.json"
        )
        with open(config_path) as f:
            return json.load(f)


# For backwards compatibility
Config = AppConfig