import json
import re
import os
import yaml
from langchain_google_genai import GoogleGenerativeAI

class MaintenanceAgent:
    def __init__(self, api_key, model_name="gemini-2.5-flash-lite", temperature=0.0):
        self.llm = GoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key
        )
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> dict:
        """Load prompts from YAML file."""
        # Get the directory where this file is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompts_file = os.path.join(current_dir, 'prompts.yaml')
        
        with open(prompts_file, 'r') as f:
            return yaml.safe_load(f)

    def _build_prompt(self, phase2_output: dict) -> str:
        """Build the maintenance analysis prompt."""
        user_template = self.prompts['maintenance']['user_template']
        return user_template.format(phase2_output=json.dumps(phase2_output, indent=2))

    def _parse_response(self, response: str) -> dict:
        """Parse LLM response, handling various JSON formats."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try extracting JSON from markdown code blocks
            match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            # Try extracting raw JSON object
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise ValueError(f"Could not parse LLM response: {response[:200]}")

    def run(self, phase2_output: dict) -> dict:
        prompt = self._build_prompt(phase2_output)
        response = self.llm.invoke(prompt)
        return self._parse_response(response)
