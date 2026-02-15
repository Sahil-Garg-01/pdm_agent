from fastapi import HTTPException
import pandas as pd

def validate_sensor_data(vdc1: list, idc1: list, pvt: list) -> None:
    """Validate sensor data consistency. Raises HTTPException on error."""
    if len(vdc1) != len(idc1):
        raise HTTPException(status_code=400, detail="Voltage and current lists must have the same length")
    if len(vdc1) != len(pvt):
        raise HTTPException(status_code=400, detail="Voltage, current, and temperature lists must have the same length")
    if len(vdc1) < 3:
        raise HTTPException(status_code=400, detail="Need at least 3 data points")

def prepare_dataframe(vdc1: list, idc1: list, pvt: list) -> pd.DataFrame:
    """Prepare sensor data for ML inference by padding to 100 points."""
    df = pd.DataFrame({
        "vdc1": (vdc1 * (100 // len(vdc1) + 1))[:100],
        "idc1": (idc1 * (100 // len(idc1) + 1))[:100],
        "pvt": (pvt * (100 // len(pvt) + 1))[:100]
    })
    return df