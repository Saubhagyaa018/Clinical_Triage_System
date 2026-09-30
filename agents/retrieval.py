import math
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

class RetrievalAgent:
    """Agent 2: Handles vector indexing, semantic search, temporal decay scoring, and filtering via Qdrant & FastEmbed."""

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, decay_lambda: float = 0.05):
        self.alpha = alpha
        self.beta = beta
        self.decay_lambda = decay_lambda

        # Initialize local Qdrant Client in pure in-memory mode
        self.client = QdrantClient(":memory:")
        
        # Initialize lightweight FastEmbed model
        self.embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.vector_dim = 384  # bge-small-en-v1.5 vector dimension

    def _calculate_temporal_decay(self, record_days_ago: float) -> float:
        """Math Section 3.2: f_T(delta_t) = e^(-lambda * delta_t)"""
        return math.exp(-self.decay_lambda * record_days_ago)

    def retrieve_relevant_history(self, chief_complaint: str, medical_history_raw: str, top_k: int = 2) -> list[dict]:
        """Indexes historical items in Qdrant using FastEmbed and returns Top-K scored items."""

        entries = [item.strip() for item in medical_history_raw.split(',') if item.strip()]
        if not entries:
            return []

        collection_name = "patient_history"

        # Re-create collection
        if self.client.collection_exists(collection_name):
            self.client.delete_collection(collection_name)

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE)
        )

        # 1. Generate embeddings using FastEmbed
        entry_embeddings = list(self.embedding_model.embed(entries))

        # 2. Build and upload Qdrant PointStructs
        points = []
        for idx, (entry, vector) in enumerate(zip(entries, entry_embeddings)):
            days_ago = idx * 180.0  # Simulating temporal spread (180 days per item)
            points.append(
                PointStruct(
                    id=idx + 1,
                    vector=vector.tolist(),
                    payload={"clinical_fact": entry, "days_ago": days_ago}
                )
            )

        self.client.upsert(collection_name=collection_name, points=points)

        # 3. Embed Chief Complaint query
        query_vector = list(self.embedding_model.embed([chief_complaint]))[0].tolist()

        # 4. Search Qdrant using query_points
        search_results = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=len(entries)
        ).points

        scored_results = []

        for hit in search_results:
            doc = hit.payload["clinical_fact"]
            days_ago = hit.payload["days_ago"]
            
            # Semantic Similarity Score S_sem
            s_sem = float(hit.score)

            # Temporal Decay Score S_temp
            s_temp = self._calculate_temporal_decay(days_ago)

            # Section 3.3 Formula: Combined Relevance Score
            s_combined = (self.alpha * s_sem) + (self.beta * s_temp)

            scored_results.append({
                "clinical_fact": doc,
                "semantic_similarity": round(s_sem, 4),
                "temporal_relevance": round(s_temp, 4),
                "combined_relevance_score": round(s_combined, 4)
            })

        # Sort by Combined Relevance Score and pick Top-K
        scored_results.sort(key=lambda x: x['combined_relevance_score'], reverse=True)
        return scored_results[:top_k]


# Local verification execution
if __name__ == "__main__":
    from agents.ingestion import DataIngestionAgent

    # 1. Ingest Patient P001
    ingestion_agent = DataIngestionAgent(csv_file_path="data/raw_patients.csv")
    patient_data = ingestion_agent.process_patient_record("P001")

    # 2. Run Qdrant Retrieval Agent
    retrieval_agent = RetrievalAgent(alpha=0.7, beta=0.3)
    filtered_evidence = retrieval_agent.retrieve_relevant_history(
        chief_complaint=patient_data.chief_complaint,
        medical_history_raw=patient_data.past_medical_history,
        top_k=2
    )

    print("\n--- Agent 2 (Qdrant + FastEmbed) Verification ---")
    print(f"Chief Complaint: {patient_data.chief_complaint}")
    print("Filtered Evidence Set (Top-K):")
    for fact in filtered_evidence:
        print(fact)