from agents.ingestion import DataIngestionAgent
from agents.retrieval import RetrievalAgent
from agents.summarization import SummarizationAgent
from agents.triage import TriageAgent

class ClinicalTriagePipeline:
    """Orchestrates the sequential execution of the 4 specialized clinical agents."""

    def __init__(self, csv_file_path: str = "data/raw_patients.csv"):
        self.ingestion_agent = DataIngestionAgent(csv_file_path=csv_file_path)
        self.retrieval_agent = RetrievalAgent(alpha=0.7, beta=0.3)
        self.summarization_agent = SummarizationAgent()
        self.triage_agent = TriageAgent()

    def run_pipeline(self, patient_id: str) -> dict:
        """Runs patient data through Ingestion -> Retrieval -> Summarization -> Risk Scoring."""
        
        # 1. Agent 1: Data Ingestion
        patient_data = self.ingestion_agent.process_patient_record(patient_id)
        patient_payload = patient_data.model_dump()

        # 2. Agent 2: Qdrant Semantic Retrieval
        filtered_evidence = self.retrieval_agent.retrieve_relevant_history(
            chief_complaint=patient_data.chief_complaint,
            medical_history_raw=patient_data.past_medical_history,
            top_k=2
        )

        # 3. Agent 3: Grounded Gemini Summary
        summary = self.summarization_agent.generate_clinical_summary(
            patient_payload=patient_payload,
            filtered_evidence=filtered_evidence
        )

        # 4. Agent 4: Triage & Risk Scoring
        triage_results = self.triage_agent.evaluate_triage_level(
            patient_payload=patient_payload,
            clinical_summary=summary
        )

        return {
            "patient_payload": patient_payload,
            "filtered_evidence": filtered_evidence,
            "clinical_summary": summary,
            "triage_results": triage_results
        }


if __name__ == "__main__":
    pipeline = ClinicalTriagePipeline()
    result = pipeline.run_pipeline("P001")
    print("\n--- Pipeline Execution Test Complete ---")
    print(f"Patient Name: {result['patient_payload']['name']}")
    print(f"NEWS2 Score: {result['triage_results']['news2_score']}")