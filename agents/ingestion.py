import math
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from utils.database import engine


def _number_or_none(value) -> Optional[float]:
    """Convert valid numeric values; normalize NULL and NaN to None."""
    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if not math.isfinite(number):
        return None

    return number


def _text_or_none(value) -> Optional[str]:
    """Normalize SQL NULL and common textual missing-value markers."""
    if value is None:
        return None

    cleaned = str(value).strip()

    if not cleaned or cleaned.lower() in {
        "nan", "none", "null", "nat"
    }:
        return None

    return cleaned


def _normalize_temperature(value):
    """
    Return (raw_value, temperature_c, unit_status).

    Dataset preprocessing assumptions:
    - 25 to 45: likely Celsius.
    - 77 to 113: likely Fahrenheit.
    - Other values: retain raw value and flag uncertainty.

    These ranges do not prove the original unit.
    """
    raw = _number_or_none(value)

    if raw is None:
        return None, None, "missing_or_invalid"

    if 25.0 <= raw <= 45.0:
        return raw, raw, "assumed_celsius"

    if 77.0 <= raw <= 113.0:
        return raw, (raw - 32.0) * 5.0 / 9.0, "assumed_fahrenheit"

    return raw, None, "unit_uncertain_or_out_of_range"


class VitalSigns(BaseModel):
    """Triage vital signs with normalized temperature."""

    heart_rate: Optional[float] = None
    bp_systolic: Optional[float] = None
    bp_diastolic: Optional[float] = None
    respiratory_rate: Optional[float] = None
    spo2: Optional[float] = None

    temperature_raw: Optional[float] = None
    temperature_c: Optional[float] = None
    temperature_unit_status: str = "missing_or_invalid"

    pain: Optional[str] = None


class VitalObservation(BaseModel):
    """One timestamped observation from the vitals table."""

    vital_id: int
    charttime: datetime

    heart_rate: Optional[float] = None
    bp_systolic: Optional[float] = None
    bp_diastolic: Optional[float] = None
    respiratory_rate: Optional[float] = None
    spo2: Optional[float] = None

    temperature_raw: Optional[float] = None
    temperature_c: Optional[float] = None
    temperature_unit_status: str = "missing_or_invalid"

    rhythm: Optional[str] = None
    pain: Optional[str] = None


class IngestedPatientPayload(BaseModel):
    """Validated current encounter data and separately stored history."""

    stay_id: int
    subject_id: int

    gender: Optional[str] = None
    race: Optional[str] = None
    chief_complaint: Optional[str] = None

    arrival_time: Optional[datetime] = None
    encounter_end_time: Optional[datetime] = None
    arrival_transport: Optional[str] = None
    disposition: Optional[str] = None

    # Evaluation ground truth; never pass this to prediction models.
    reference_acuity: Optional[int] = None

    # These fields are not currently populated from the database.
    name: Optional[str] = None
    age: Optional[int] = None

    # Structured historical evidence for Agent 2.
    past_medical_history: list[dict] = Field(default_factory=list)

    # Do not infer active medication status from historical records.
    active_medications: Optional[str] = None

    vitals: VitalSigns
    vitals_observations: list[VitalObservation] = Field(
        default_factory=list
    )

    def to_prediction_payload(self) -> dict:
        """
        Return current clinical inputs only.

        Historical evidence, identifiers, evaluation labels, encounter
        outcomes, and timestamped observations are excluded.
        """
        return self.model_dump(
            exclude={
                "stay_id",
                "subject_id",
                "race",
                "reference_acuity",
                "encounter_end_time",
                "disposition",
                "arrival_time",
                "arrival_transport",
                "vitals_observations",
                "past_medical_history",
                "active_medications",
            }
        )


class DataIngestionAgent:
    """Agent 1: Fetch and validate encounter data from PostgreSQL."""

    def process_patient_record(
        self, stay_id: int
    ) -> IngestedPatientPayload:

        # Fetch the current encounter and its triage record.
        encounter_query = text("""
            SELECT
                e.stay_id,
                e.subject_id,
                e.intime AS arrival_time,
                e.outtime AS encounter_end_time,
                e.arrival_transport,
                e.disposition,
                p.gender,
                p.race,
                t.temperature,
                t.heartrate,
                t.resprate,
                t.o2sat,
                t.sbp,
                t.dbp,
                t.pain,
                t.chiefcomplaint,
                t.acuity AS reference_acuity
            FROM encounters AS e
            JOIN patients AS p
                ON p.subject_id = e.subject_id
            JOIN triage AS t
                ON t.stay_id = e.stay_id
               AND t.subject_id = e.subject_id
            WHERE e.stay_id = :stay_id
        """)

        # Fetch timestamped observations from the current encounter.
        observations_query = text("""
            SELECT
                vital_id,
                charttime,
                temperature,
                heartrate,
                resprate,
                o2sat,
                sbp,
                dbp,
                rhythm,
                pain
            FROM vitals
            WHERE stay_id = :stay_id
            ORDER BY charttime ASC, vital_id ASC
        """)

        # Fetch diagnoses associated with earlier completed encounters.
        # The earlier encounter's end time is used as the evidence time.
        diagnoses_query = text("""
            SELECT
                d.stay_id,
                d.icd_title,
                old_e.outtime AS event_time
            FROM public.diagnosis_history AS d
            JOIN public.encounters AS old_e
                ON old_e.stay_id = d.stay_id
               AND old_e.subject_id = d.subject_id
            WHERE d.subject_id = :subject_id
              AND d.stay_id <> :current_stay_id
              AND old_e.outtime IS NOT NULL
              AND old_e.outtime <= :arrival_time
              AND d.icd_title IS NOT NULL
              AND TRIM(d.icd_title) <> ''
            ORDER BY old_e.outtime DESC
        """)

        # Fetch medication-reconciliation records from earlier encounters.
        # These records are evidence of recorded medications, not proof
        # that a medication is currently active.
        medications_query = text("""
            SELECT
                m.stay_id,
                m.medication_name,
                m.etcdescription,
                m.charttime AS event_time
            FROM public.medication_history AS m
            JOIN public.encounters AS old_e
                ON old_e.stay_id = m.stay_id
               AND old_e.subject_id = m.subject_id
            WHERE m.subject_id = :subject_id
              AND m.stay_id <> :current_stay_id
              AND old_e.outtime IS NOT NULL
              AND old_e.outtime <= :arrival_time
              AND m.charttime IS NOT NULL
              AND m.charttime < :arrival_time
              AND m.medication_name IS NOT NULL
              AND TRIM(m.medication_name) <> ''
            ORDER BY m.charttime DESC
        """)

        try:
            with engine.connect() as connection:
                result = connection.execute(
                    encounter_query,
                    {"stay_id": stay_id},
                ).mappings().one_or_none()

                if result is None:
                    raise ValueError(
                        "No matching encounter and triage record "
                        f"found for stay_id={stay_id}."
                    )

                observation_rows = connection.execute(
                    observations_query,
                    {"stay_id": stay_id},
                ).mappings().all()

                # If arrival time is unavailable, history cannot be
                # safely filtered against the current encounter.
                if result["arrival_time"] is not None:
                    history_params = {
                        "subject_id": result["subject_id"],
                        "current_stay_id": result["stay_id"],
                        "arrival_time": result["arrival_time"],
                    }

                    diagnosis_rows = connection.execute(
                        diagnoses_query,
                        history_params,
                    ).mappings().all()

                    medication_rows = connection.execute(
                        medications_query,
                        history_params,
                    ).mappings().all()
                else:
                    diagnosis_rows = []
                    medication_rows = []

        except SQLAlchemyError as exc:
            raise RuntimeError(
                "Unable to fetch encounter or history data from "
                "PostgreSQL. Check the database connection, schema, "
                "and history tables."
            ) from exc

        # Normalize the current triage temperature.
        (
            triage_temp_raw,
            triage_temp_c,
            triage_temp_status,
        ) = _normalize_temperature(result["temperature"])

        # Normalize current encounter vital observations.
        observations = []

        for row in observation_rows:
            raw_temp, temp_c, temp_status = _normalize_temperature(
                row["temperature"]
            )

            # A malformed timestamp cannot satisfy VitalObservation.
            if row["charttime"] is None:
                continue

            observations.append(
                VitalObservation(
                    vital_id=row["vital_id"],
                    charttime=row["charttime"],
                    heart_rate=_number_or_none(row["heartrate"]),
                    bp_systolic=_number_or_none(row["sbp"]),
                    bp_diastolic=_number_or_none(row["dbp"]),
                    respiratory_rate=_number_or_none(row["resprate"]),
                    spo2=_number_or_none(row["o2sat"]),
                    temperature_raw=raw_temp,
                    temperature_c=temp_c,
                    temperature_unit_status=temp_status,
                    rhythm=_text_or_none(row["rhythm"]),
                    pain=_text_or_none(row["pain"]),
                )
            )

        # Build structured historical evidence for Agent 2.
        historical_evidence = []

        for row in diagnosis_rows:
            diagnosis = _text_or_none(row["icd_title"])
            event_time = row["event_time"]

            if diagnosis is None or event_time is None:
                continue

            historical_evidence.append({
                "clinical_fact": diagnosis,
                "source": "diagnosis_history",
                "event_time": event_time,
            })

        for row in medication_rows:
            medication_name = _text_or_none(row["medication_name"])
            event_time = row["event_time"]

            if medication_name is None or event_time is None:
                continue

            description = _text_or_none(row["etcdescription"])
            fact = f"Medication recorded: {medication_name}"

            if description:
                fact += f" ({description})"

            historical_evidence.append({
                "clinical_fact": fact,
                "source": "medication_history",
                "event_time": event_time,
            })

        # Return current encounter data and historical evidence separately.
        return IngestedPatientPayload(
            stay_id=result["stay_id"],
            subject_id=result["subject_id"],
            gender=_text_or_none(result["gender"]),
            race=_text_or_none(result["race"]),
            chief_complaint=_text_or_none(result["chiefcomplaint"]),
            arrival_time=result["arrival_time"],
            encounter_end_time=result["encounter_end_time"],
            arrival_transport=_text_or_none(
                result["arrival_transport"]
            ),
            disposition=_text_or_none(result["disposition"]),
            reference_acuity=result["reference_acuity"],
            past_medical_history=historical_evidence,
            active_medications=None,
            vitals=VitalSigns(
                heart_rate=_number_or_none(result["heartrate"]),
                bp_systolic=_number_or_none(result["sbp"]),
                bp_diastolic=_number_or_none(result["dbp"]),
                respiratory_rate=_number_or_none(result["resprate"]),
                spo2=_number_or_none(result["o2sat"]),
                temperature_raw=triage_temp_raw,
                temperature_c=triage_temp_c,
                temperature_unit_status=triage_temp_status,
                pain=_text_or_none(result["pain"]),
            ),
            vitals_observations=observations,
        )


if __name__ == "__main__":
    test_stay_id = 30094124
    agent = DataIngestionAgent()

    try:
        payload = agent.process_patient_record(test_stay_id)

        print("\n--- PostgreSQL Ingestion Test ---")
        print(f"Stay ID: {payload.stay_id}")
        print(f"Subject ID: {payload.subject_id}")
        print(f"Chief complaint: {payload.chief_complaint}")
        print(f"Vital observations: {len(payload.vitals_observations)}")

        print("\n--- Historical Evidence ---")
        print(f"Total records: {len(payload.past_medical_history)}")

        source_counts = {}
        for item in payload.past_medical_history:
            source = item["source"]
            source_counts[source] = source_counts.get(source, 0) + 1

        print(f"Records by source: {source_counts}")

        if payload.past_medical_history:
            print("Example record:")
            print(payload.past_medical_history[0])

        print("\n--- Prediction Payload ---")
        print(payload.to_prediction_payload())

        print("\n--- Reference Acuity (evaluation only) ---")
        print(payload.reference_acuity)

    except (ValueError, RuntimeError) as exc:
        print(f"Ingestion test failed: {exc}")
