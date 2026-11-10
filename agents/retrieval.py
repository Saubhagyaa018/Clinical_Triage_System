import math
from datetime import datetime
from typing import Optional

from fastembed import TextEmbedding
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from config import settings


class ClinicalHistoryFact(BaseModel):
    """One genuine clinical-history item with optional source metadata."""

    clinical_fact: str
    source: str = "unknown"
    event_time: Optional[datetime] = None


class RetrievedEvidence(BaseModel):
    clinical_fact: str
    semantic_similarity: float
    temporal_relevance: Optional[float] = None
    combined_relevance_score: float
    source: str = "unknown"
    event_time: Optional[datetime] = None


class RetrievalAgent:
    """
    Agent 2: Retrieves clinically relevant historical facts.

    Uses FastEmbed for embeddings, Qdrant for semantic search,
    and actual timestamps for temporal decay when available.
    """

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        decay_lambda: float = 0.05,
    ) -> None:
        if alpha < 0 or beta < 0 or alpha + beta <= 0:
            raise ValueError(
                "alpha and beta must be non-negative, "
                "and their sum must be greater than zero."
            )

        if decay_lambda < 0:
            raise ValueError("decay_lambda cannot be negative.")

        # Normalize weights so their sum is 1.
        weight_sum = alpha + beta
        self.alpha = alpha / weight_sum
        self.beta = beta / weight_sum
        self.decay_lambda = decay_lambda

        self.client = QdrantClient(":memory:")
        self.embedding_model = TextEmbedding(
            settings.EMBEDDING_MODEL
        )

    def _calculate_temporal_decay(
        self,
        record_time: datetime,
        reference_time: datetime,
    ) -> float:
        """Calculate exp(-lambda * elapsed_days) using real timestamps."""

        if record_time.tzinfo != reference_time.tzinfo:
            raise ValueError(
                "Record and reference timestamps must use "
                "compatible timezone information."
            )

        elapsed_days = (
            reference_time - record_time
        ).total_seconds() / 86400.0

        if elapsed_days < 0:
            raise ValueError(
                "Future records cannot be used as historical evidence."
            )

        return math.exp(-self.decay_lambda * elapsed_days)

    def retrieve_relevant_history(
        self,
        chief_complaint: str,
        medical_history_raw: (
            str | list[ClinicalHistoryFact] | list[dict] | None
        ),
        top_k: int = 2,
        reference_time: Optional[datetime] = None,
    ) -> list[RetrievedEvidence]:
        """
        Retrieve the top-k relevant history items.

        Structured facts retain their own timestamps and sources.
        Legacy strings are treated as one text item; timestamps are
        never fabricated for them.
        """

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if not chief_complaint or not chief_complaint.strip():
            return []

        if not medical_history_raw:
            return []

        # Support the old string interface without comma-splitting
        # clinical facts or inventing their timestamps.
        if isinstance(medical_history_raw, str):
            facts = [
                ClinicalHistoryFact(
                    clinical_fact=medical_history_raw.strip()
                )
            ]
        else:
            facts = [
                item if isinstance(item, ClinicalHistoryFact)
                else ClinicalHistoryFact.model_validate(item)
                for item in medical_history_raw
            ]

        facts = [
            fact for fact in facts
            if fact.clinical_fact.strip()
        ]

        # Historical evidence must not come from after the reference time.
        if reference_time is not None:
            facts = [
                fact for fact in facts
                if fact.event_time is None
                or fact.event_time <= reference_time
            ]

        if not facts:
            return []

        collection_name = "patient_history"

        # Recreate the temporary collection for this retrieval request.
        if self.client.collection_exists(collection_name):
            self.client.delete_collection(collection_name)

        texts = [fact.clinical_fact for fact in facts]

        embeddings = [
            vector.tolist()
            for vector in self.embedding_model.embed(texts)
        ]

        if not embeddings:
            return []

        # Derive vector size from the actual embedding model output.
        vector_dim = len(embeddings[0])

        if any(len(vector) != vector_dim for vector in embeddings):
            raise RuntimeError(
                "The embedding model returned inconsistent vector sizes."
            )

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_dim,
                distance=Distance.COSINE,
            ),
        )

        points = [
            PointStruct(
                id=index + 1,
                vector=embedding,
                payload={
                    "clinical_fact": fact.clinical_fact,
                    "source": fact.source,
                    "event_time": (
                        fact.event_time.isoformat()
                        if fact.event_time is not None
                        else None
                    ),
                },
            )
            for index, (fact, embedding) in enumerate(
                zip(facts, embeddings)
            )
        ]

        self.client.upsert(
            collection_name=collection_name,
            points=points,
        )

        query_vector = next(
            self.embedding_model.embed([chief_complaint])
        ).tolist()

        search_results = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=len(facts),
            with_payload=True,
        ).points

        # Map point IDs back to the original structured facts.
        facts_by_id = {
            index + 1: fact
            for index, fact in enumerate(facts)
        }

        scored_results: list[RetrievedEvidence] = []

        for hit in search_results:
            fact = facts_by_id[hit.id]
            semantic_score = float(hit.score)

            temporal_score = None

            if (
                fact.event_time is not None
                and reference_time is not None
            ):
                temporal_score = self._calculate_temporal_decay(
                    record_time=fact.event_time,
                    reference_time=reference_time,
                )

                combined_score = (
                    self.alpha * semantic_score
                    + self.beta * temporal_score
                )
            else:
                # Do not penalize or reward a fact using an invented age.
                # Without timestamps, rank it using semantic relevance.
                combined_score = semantic_score

            scored_results.append(
                RetrievedEvidence(
                    clinical_fact=fact.clinical_fact,
                    semantic_similarity=round(
                        semantic_score, 4
                    ),
                    temporal_relevance=(
                        round(temporal_score, 4)
                        if temporal_score is not None
                        else None
                    ),
                    combined_relevance_score=round(
                        combined_score, 4
                    ),
                    source=fact.source,
                    event_time=fact.event_time,
                )
            )

        scored_results.sort(
            key=lambda item: item.combined_relevance_score,
            reverse=True,
        )

        return scored_results[:top_k]
