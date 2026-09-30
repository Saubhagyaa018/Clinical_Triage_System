import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv()

class SummarizationAgent:
    """Agent 3: Synthesizes filtered patient history and real-time vitals into a grounded clinical summary."""

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables. Please check your .env file.")
        
        # Initialize Google GenAI client
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-3.6-flash"

    def generate_clinical_summary(self, patient_payload: dict, filtered_evidence: list[dict]) -> str:
        """Constructs a strict grounding prompt and invokes Gemini LLM."""
        
        # Format Top-K evidence into clean text for prompt grounding
        evidence_text = "\n".join([
            f"- {item['clinical_fact']} (Relevance Score: {item['combined_relevance_score']})"
            for item in filtered_evidence
        ])

        prompt = f"""You are a Board-Certified Emergency Medicine Physician Assistant performing rapid clinical synthesis.

PATIENT DEMOGRAPHICS & TELEMETRY:
- Name: {patient_payload['name']} (Age: {patient_payload['age']}, Gender: {patient_payload['gender']})
- Chief Complaint: {patient_payload['chief_complaint']}
- Vital Signs: HR: {patient_payload['vitals']['heart_rate']} bpm, BP: {patient_payload['vitals']['bp_systolic']}/{patient_payload['vitals']['bp_diastolic']} mmHg, RR: {patient_payload['vitals']['respiratory_rate']} rpm, SpO2: {patient_payload['vitals']['spo2']}%, Temp: {patient_payload['vitals']['temperature_c']}°C

GROUNDED HISTORICAL EVIDENCE (Top Semantic Matches):
{evidence_text}

INSTRUCTIONS:
Generate a concise, high-density clinical summary strictly using the provided telemetry and grounded historical evidence above.
Do NOT invent facts, diagnoses, or treatments not implied by the provided data.

Structure your response into 3 concise bulleted sections:
1. Presenting Clinical Situation: Synthesize Chief Complaint and demographics.
2. Relevant Grounded History: Explain how the retrieved historical facts increase risk for the current complaint.
3. Physiological Telemetry Flags: Explicitly state any abnormal vitals requiring immediate triage attention.
"""

        interaction = self.client.interactions.create(
            model=self.model_name,
            input=prompt,
            generation_config={
            "temperature": 0.2,
            "max_output_tokens": 1000
            }
        )

        return interaction.output_text


# Local verification execution
if __name__ == "__main__":
    from agents.ingestion import DataIngestionAgent
    from agents.retrieval import RetrievalAgent

    # 1. Data Ingestion
    ingestion_agent = DataIngestionAgent(csv_file_path="data/raw_patients.csv")
    patient_data = ingestion_agent.process_patient_record("P001")

    # 2. Semantic Retrieval
    retrieval_agent = RetrievalAgent(alpha=0.7, beta=0.3)
    filtered_evidence = retrieval_agent.retrieve_relevant_history(
        chief_complaint=patient_data.chief_complaint,
        medical_history_raw=patient_data.past_medical_history,
        top_k=2
    )

    # 3. LLM Grounded Summarization
    summarizer = SummarizationAgent()
    summary = summarizer.generate_clinical_summary(
        patient_payload=patient_data.model_dump(),
        filtered_evidence=filtered_evidence
    )

    print("\n================ Agent 3 (Grounded Gemini Summary) ================")
    print(summary)
    print("===================================================================")