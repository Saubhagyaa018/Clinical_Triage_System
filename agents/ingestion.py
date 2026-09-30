from datetime import datetime, timezone
import pandas as pd
from pydantic import BaseModel, Field

# Define structured Pydantic schemas for data validation
class VitalSigns(BaseModel):
    heart_rate: int = Field(..., description="Heart rate in bpm")
    bp_systolic: int = Field(..., description="Systolic blood pressure in mmHg")
    bp_diastolic: int = Field(..., description="Diastolic blood pressure in mmHg")
    respiratory_rate: int = Field(..., description="Respiratory rate in breaths/min")
    spo2: int = Field(..., description="Oxygen saturation percentage")
    temperature_c: float = Field(..., description="Body temperature in Celsius")

class IngestedPatientPayload(BaseModel):
    patient_id: str
    name: str
    age: int
    gender: str
    chief_complaint: str
    past_medical_history: str
    active_medications: str
    vitals: VitalSigns
    timestamp_utc: str

class DataIngestionAgent:
    """Agent 1: Handles data acquisition, validation, and temporal synchronization."""
    
    def __init__(self, csv_file_path: str):
        self.csv_file_path = csv_file_path

    def process_patient_record(self, patient_id: str) -> IngestedPatientPayload:
        """Loads raw record, validates fields, and appends UTC timestamp."""
        df = pd.read_csv(self.csv_file_path)
        record = df[df['patient_id'] == patient_id]

        if record.empty:
            raise ValueError(f"Patient with ID '{patient_id}' not found in dataset.")

        row = record.iloc[0]

        # Extract vitals into validated sub-model
        vitals = VitalSigns(
            heart_rate=int(row['heart_rate']),
            bp_systolic=int(row['bp_systolic']),
            bp_diastolic=int(row['bp_diastolic']),
            respiratory_rate=int(row['respiratory_rate']),
            spo2=int(row['spo2']),
            temperature_c=float(row['temperature_c'])
        )

        # Build synchronized payload with UTC timestamp
        payload = IngestedPatientPayload(
            patient_id=str(row['patient_id']),
            name=str(row['name']),
            age=int(row['age']),
            gender=str(row['gender']),
            chief_complaint=str(row['chief_complaint']),
            past_medical_history=str(row['past_medical_history']),
            active_medications=str(row['active_medications']),
            vitals=vitals,
            timestamp_utc=datetime.now(timezone.utc).isoformat()
        )

        return payload


# Simple local test runner to verify Agent 1 standalone execution
if __name__ == "__main__":
    agent = DataIngestionAgent(csv_file_path="../data/raw_patients.csv")
    test_output = agent.process_patient_record("P001")
    print("\n--- Agent 1 Output Verification ---")
    print(test_output.model_dump_json(indent=2))