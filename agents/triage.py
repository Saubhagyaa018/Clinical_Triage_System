# import os
# from dotenv import load_dotenv
from google import genai
from google.genai import types
from config import settings
from pydantic import BaseModel
from config import settings



class TriageResult(BaseModel):
    news2_score: int
    news2_risk: str
    esi_level: int
    triage_assessment: str

# load_dotenv()

class TriageAgent:
    """Agent 4: Combines deterministic NEWS2 mathematical scoring with LLM-based ESI Level classification."""

    def __init__(self):
        self.model = settings.GEMINI_MODEL
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found")
        
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.model_name = settings.GEMINI_MODEL

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
        print("DEBUG: Patient vitals:", repr(patient_payload.get("vitals")))

        news2_score, news2_risk = self.calculate_news2_score(
        patient_payload["vitals"])

        print("DEBUG: NEWS2 score:", news2_score)
        print("DEBUG: NEWS2 risk:", news2_risk)
        prompt = f"""You are an Emergency Triage Physician assigning an Emergency Severity Index (ESI) Level (1-5).

PATIENT OVERVIEW:
- Chief Complaint: {patient_payload['chief_complaint']}
- Age/Gender: {patient_payload['age']} / {patient_payload['gender']}
- Calculated NEWS2 Score: {news2_score} ({news2_risk})

CLINICAL SUMMARY:
{clinical_summary}
INSTRUCTIONS:

Assign an Emergency Severity Index (ESI) level from 1 to 5 using standard ESI decision criteria and only the information supplied in the current encounter.

ASSESSMENT RULES:
- Assess whether an immediate life-saving intervention is required before considering lower acuity levels.
- Assess high-risk presentations, altered mental status, and severe pain or distress using documented evidence.
- Do not use a NEWS2 score of 0 or normal vital signs alone to exclude ESI 1 or ESI 2.
- Do not interpret information that is missing from the encounter as a confirmed normal finding. For example, if mental status is not documented, do not state that mental status is normal.
- Do not treat historical diagnoses or symptoms as current symptoms.
- Do not infer that a fall involved syncope, caused an injury, or requires imaging unless the supplied evidence supports that conclusion.
- Assign ESI 3, 4, or 5 according to the expected number of distinct ED resources only after assessing the higher-acuity criteria. Do not invent anticipated tests, imaging, consultations, or procedures merely to justify a level.
- If important information needed for classification is missing, explicitly identify the uncertainty in the justification. Do not fabricate findings or change the classification simply to match a reference label.

OUTPUT FORMAT:
Assigned ESI Level: ESI Level X

Primary Clinical Justification:
Briefly explain the classification using documented current-encounter evidence. Clearly identify important missing information.

Recommended Immediate Triage Actions:
Provide 2–3 concise actions supported by the available evidence. Do not prescribe tests or monitoring solely on the basis of undocumented assumptions.
"""

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=2000
            )
        )
        print("DEBUG: Gemini response text:", repr(response.text))
        print("DEBUG: Finish reason:", response.candidates[0].finish_reason)

        
        import re

        match = re.search(
            r"Assigned\s+ESI\s+Level\s*:\s*\**\s*ESI\s+Level\s*([1-5])\b",
            response.text,
            re.IGNORECASE,
        )

        if not match:
            raise ValueError(
                "Could not extract a valid ESI level (1-5) "
                "from the model response. Do not use this result "
                "for evaluation.\n"
                f"Model response: {response.text!r}"
            )

        esi_level = int(match.group(1))

        return TriageResult(
            news2_score=news2_score,
            news2_risk=news2_risk,
            esi_level=esi_level,
            triage_assessment=response.text,
        )
        
        
        


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