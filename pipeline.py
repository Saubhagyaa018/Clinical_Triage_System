from pydantic import BaseModel

from agents.ingestion import (
    DataIngestionAgent,
    IngestedPatientPayload,
)
from agents.retrieval import RetrievalAgent, RetrievedEvidence
from agents.summarization import SummarizationAgent
from agents.triage import TriageAgent, TriageResult
from config import settings


class PipelineResult(BaseModel):
    """Combined output from the four-agent clinical triage pipeline."""

    patient_payload: IngestedPatientPayload
    filtered_evidence: list[RetrievedEvidence]
    clinical_summary: str
    triage_results: TriageResult


class ClinicalTriagePipeline:
    """Orchestrate ingestion, retrieval, summarization, and triage."""

    def __init__(self) -> None:
        self.ingestion_agent = DataIngestionAgent()

        self.retrieval_agent = RetrievalAgent(
            alpha=settings.RETRIEVAL_ALPHA,
            beta=settings.RETRIEVAL_BETA,
            decay_lambda=settings.RETRIEVAL_DECAY_LAMBDA,
        )

        self.summarization_agent = SummarizationAgent()
        self.triage_agent = TriageAgent()

    def run_pipeline(self, stay_id: int) -> PipelineResult:
        """Run the pipeline for one PostgreSQL encounter."""

        # Agent 1: Retrieve and validate current encounter data
        # and structured historical evidence.
        patient_data = self.ingestion_agent.process_patient_record(
            stay_id
        )

        # Keep current prediction inputs separate from historical evidence,
        # identifiers, evaluation labels, and encounter outcomes.
        prediction_payload = patient_data.to_prediction_payload()

        # Agent 2: Retrieve relevant evidence from earlier ED encounters.
        medical_history = patient_data.past_medical_history
        arrival_time = patient_data.arrival_time

        if medical_history and arrival_time is not None:
            filtered_evidence = (
                self.retrieval_agent.retrieve_relevant_history(
                    chief_complaint=patient_data.chief_complaint or "",
                    medical_history_raw=medical_history,
                    top_k=2,
                    reference_time=arrival_time,
                )
            )
        else:
            filtered_evidence = []

        # Agent 3: Generate a clinical summary from current inputs
        # and the retrieved historical evidence.
        summary = self.summarization_agent.generate_clinical_summary(
            patient_payload=prediction_payload,
            filtered_evidence=filtered_evidence,
        )

        # Agent 4: Calculate risk and produce the triage assessment.
        triage_results = self.triage_agent.evaluate_triage_level(
            patient_payload=prediction_payload,
            clinical_summary=summary,
        )

        # Retain the reference label for evaluation only.
        # It is not passed to either prediction agent.
        return PipelineResult(
            patient_payload=patient_data,
            filtered_evidence=filtered_evidence,
            clinical_summary=summary,
            triage_results=triage_results,
        )


if __name__ == "__main__":
    # Use a real stay_id from the PostgreSQL encounters table.
    test_stay_id = 30094124

    pipeline = ClinicalTriagePipeline()

    try:
        result = pipeline.run_pipeline(test_stay_id)

        print("\n--- Pipeline Execution Complete ---")
        print(f"Encounter ID: {result.patient_payload.stay_id}")
        print(
            "Reference acuity (evaluation only):",
            result.patient_payload.reference_acuity,
        )
        print(
            "NEWS2 score:",
            result.triage_results.news2_score,
        )
        print(
            "Retrieved evidence:",
            len(result.filtered_evidence),
        )
        print(
            "Summary length:",
            len(result.clinical_summary),
        )

        print("\n--- Retrieved Evidence ---")
        for item in result.filtered_evidence:
            print(item.model_dump())

        print("\n--- Clinical Summary ---")
        print(result.clinical_summary)

        print("\n--- Triage Assessment ---")
        print(result.triage_results.triage_assessment)

    except (ValueError, RuntimeError) as exc:
        print(f"Pipeline execution failed: {exc}")
