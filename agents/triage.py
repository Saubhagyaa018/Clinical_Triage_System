import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

class TriageAgent:
    """Agent 4: Combines deterministic NEWS2 mathematical scoring with LLM-based ESI Level classification."""

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables. Check your .env file.")
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-3.8-flash"

    def calculate_news2_score(self, vitals: dict) -> tuple[int, str]:
        """Calculates standard NHS NEWS2 score deterministically from vitals."""
        score = 0

        # 1. Respiration Rate (rpm)
        rr = vitals.get('respiratory_rate', 0)
        if rr <= 8 or rr >= 25:
            score += 3
        elif 21 <= rr <= 24:
            score += 2
        elif 9 <= rr <= 11:
            score += 1

        # 2. SpO2 (%) - Scale 1
        spo2 = vitals.get('spo2', 100)
        if spo2 <= 91:
            score += 3
        elif 92 <= spo2 <= 93:
            score += 2
        elif 94 <= spo2 <= 95:
            score += 1

        # 3. Systolic Blood Pressure (mmHg)
        sys_bp = vitals.get('bp_systolic', 120)
        if sys_bp <= 90 or sys_bp >= 220:
            score += 3
        elif 91 <= sys_bp <= 100:
            score += 2
        elif 101 <= sys_bp <= 110:
            score += 1

        # 4. Heart Rate (bpm)
        hr = vitals.get('heart_rate', 80)
        if hr <= 40 or hr >= 131:
            score += 3
        elif 111 <= hr <= 130:
            score += 2
        elif 41 <= hr <= 50 or 91 <= hr <= 110:
            score += 1

        # 5. Temperature (°C)
        temp = vitals.get('temperature_c', 37.0)
        if temp <= 35.0:
            score += 3
        elif temp >= 39.1:
            score += 2
        elif (35.1 <= temp <= 36.0) or (38.1 <= temp <= 39.0):
            score += 1

        # Determine NEWS2 Clinical Risk Level
        if score >= 7:
            risk_level = "HIGH (Emergency - Immediate Clinical Assessment Required)"
        elif 5 <= score <= 6:
            risk_level = "MEDIUM (Urgent - Urgent Medical Review)"
        else:
            risk_level = "LOW (Routine Observation)"

        return score, risk_level

    def evaluate_triage_level(self, patient_payload: dict, clinical_summary: str) -> dict:
        """Computes NEWS2 score and prompts Gemini for ESI triage level classification."""
        
        # Calculate NEWS2
        news2_score, news2_risk = self.calculate_news2_score(patient_payload['vitals'])

        prompt = f"""You are an Emergency Triage Physician assigning an Emergency Severity Index (ESI) Level (1-5).

PATIENT OVERVIEW:
- Chief Complaint: {patient_payload['chief_complaint']}
- Age/Gender: {patient_payload['age']} / {patient_payload['gender']}
- Calculated NEWS2 Score: {news2_score} ({news2_risk})

CLINICAL SUMMARY:
{clinical_summary}

INSTRUCTIONS:
Assign an ESI Triage Level (1 to 5) based on standard triage criteria:
- ESI Level 1: Immediate life-saving intervention required.
- ESI Level 2: High risk, confused/lethargic/disoriented, or severe pain/distress.
- ESI Level 3: Stable, but requires multiple resources/tests.
- ESI Level 4: Stable, requires one resource.
- ESI Level 5: Non-urgent, requires no resources.

Structure your response clearly with:
- **Assigned ESI Level:** ESI Level X (Name)
- **Primary Clinical Justification:** Concise explanation.
- **Recommended Immediate Triage Actions:** 2-3 bullet points.
"""

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=500
            )
        )

        return {
            "news2_score": news2_score,
            "news2_risk": news2_risk,
            "triage_assessment": response.text
        }


# Local verification execution
if __name__ == "__main__":
    from agents.ingestion import DataIngestionAgent
    from agents.retrieval import RetrievalAgent
    from agents.summarization import SummarizationAgent

    # 1. Agent 1: Data Ingestion
    ingestion_agent = DataIngestionAgent(csv_file_path="data/raw_patients.csv")
    patient_data = ingestion_agent.process_patient_record("P001")

    # 2. Agent 2: Semantic Retrieval
    retrieval_agent = RetrievalAgent(alpha=0.7, beta=0.3)
    filtered_evidence = retrieval_agent.retrieve_relevant_history(
        chief_complaint=patient_data.chief_complaint,
        medical_history_raw=patient_data.past_medical_history,
        top_k=2
    )

    # 3. Agent 3: Grounded Summarization
    summarizer = SummarizationAgent()
    summary = summarizer.generate_clinical_summary(
        patient_payload=patient_data.model_dump(),
        filtered_evidence=filtered_evidence
    )

    # 4. Agent 4: Triage & Risk Engine
    triage_agent = TriageAgent()
    triage_result = triage_agent.evaluate_triage_level(
        patient_payload=patient_data.model_dump(),
        clinical_summary=summary
    )

    print("\n================ Agent 4 (Triage & Risk Score) ================")
    print(f"NEWS2 Score: {triage_result['news2_score']}")
    print(f"NEWS2 Risk Level: {triage_result['news2_risk']}")
    print("\nLLM ESI Classification:")
    print(triage_result['triage_assessment'])
    print("=================================================================")