import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from ml.inference import MLEngine
from agent.agent import MaintenanceAgent

load_dotenv()

raw_df = pd.DataFrame({
    "vdc1": np.random.normal(600, 3, 200),
    "idc1": np.random.normal(10.0, 0.2, 200)
})

engine = MLEngine()
phase2_output = engine.predict_from_raw(raw_df)

print("\n=== ML OUTPUT ===")
print(phase2_output)

agent = MaintenanceAgent(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model_name="gemini-2.5-flash-lite",
    temperature=0.0
)

agent_output = agent.run(phase2_output)

print("\n=== AGENT OUTPUT ===")
print(agent_output)
