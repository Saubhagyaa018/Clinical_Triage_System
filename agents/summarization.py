# import os
# from dotenv import load_dotenv
from google import genai
from google.genai import types
from config import settings

# Load environment variables from .env file
# load_dotenv()

class SummarizationAgent:
    """Agent 3: Synthesizes filtered patient history and real-time vitals into a grounded clinical summary."""

    def __init__(self):
        # api_key = os.getenv("GEMINI_API_KEY")
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found")
    
        # Initialize Google GenAI client
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.model_name =  settings.GEMINI_MODEL

    def generate_clinical_summary(self, patient_payload: dict, filtered_evidence: list[dict]) -> str:
        """Constructs a strict grounding prompt and invokes Gemini LLM."""
        
        # Format Top-K evidence into clean text for prompt grounding
        evidence_text = "\n".join([
            f"- {item.clinical_fact} (Relevance Score: {item.combined_relevance_score})"
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

Generate a concise clinical summary using only the current patient information and retrieved historical evidence provided above.

1. Presenting Clinical Situation:
   - Report the current chief complaint and supplied demographics.
   - Do not add symptoms that are not explicitly reported in the current encounter.

2. Relevant Grounded History:
   - Clearly identify retrieved diagnoses as historical records.
   - Do not describe historical symptoms, such as chest pain or weakness, as current symptoms unless the current encounter explicitly confirms them.
   - Do not claim that historical diagnoses caused, precipitated, or explain the current presentation.
   - If the relevance of a historical fact is uncertain, state that its relevance to the current presentation is not established.

3. Physiological Telemetry Flags:
   - Report abnormalities supported by the supplied vital signs.
   - Do not claim that normal vital signs rule out a serious condition.
   - Do not invent examination findings, diagnoses, symptoms, or test results.

GENERAL GROUNDING RULE:
Every clinical statement must be traceable to the supplied current encounter data or retrieved historical evidence. Distinguish recorded facts from clinical uncertainty. Do not make unsupported causal or risk claims.
"""

        interaction = self.client.interactions.create(
            model=self.model_name,
            input=prompt,
            generation_config={
            "temperature": 0.2,
            "max_output_tokens": 2000
            }
        )
        print("DEBUG: Summary output:", repr(interaction.output_text))
        print("DEBUG: Interaction details:", interaction)
        print("DEBUG: Summary status:", interaction.status)

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