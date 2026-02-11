import json
import re
from langchain_google_genai import GoogleGenerativeAI

class MaintenanceAgent:
    def __init__(self, api_key, model_name="gemini-2.5-flash-lite", temperature=0.0):
        self.llm = GoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key
        )

    def _build_prompt(self, phase2_output: dict) -> str:
        """Build the maintenance analysis prompt."""
        return f"""
You are a maintenance decision AI.
You must reason ONLY from the provided JSON.
Do NOT invent data.

INPUT:
{json.dumps(phase2_output, indent=2)}

MANDATORY: Return output strictly in JSON format only. Do not include any markdown, code blocks, or extra text.

OUTPUT FORMAT:
{{
  "diagnosis": "...",
  "urgency": "Low | Medium | High",
  "recommended_action": "...",
  "justification": ["...", "..."]
}}
"""

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
