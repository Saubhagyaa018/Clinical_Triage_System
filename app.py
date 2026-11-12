import streamlit as st
from sqlalchemy import text

from pipeline import ClinicalTriagePipeline
from utils.database import engine

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Clinical Triage Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# Database: load available patient encounters
# --------------------------------------------------

@st.cache_data(ttl=60)
def load_encounters():
    """Load real encounter IDs and patient identifiers from PostgreSQL."""

    query = text("""
        SELECT
            e.stay_id,
            e.subject_id,
            t.chiefcomplaint
        FROM encounters AS e
        LEFT JOIN triage AS t
            ON e.stay_id = t.stay_id
        ORDER BY e.stay_id
    """)

    with engine.connect() as connection:
        rows = connection.execute(query).mappings().all()

    return [dict(row) for row in rows]


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def get_field(obj, field, default=None):
    """Read a field from a Pydantic model or dictionary."""
    if isinstance(obj, dict):
        return obj.get(field, default)

    return getattr(obj, field, default)


def display_value(value):
    """Format values for readable display."""
    if value is None or value == "":
        return "Not available"

    if isinstance(value, dict):
        return ", ".join(
            f"{key.replace('_', ' ').title()}: {val}"
            for key, val in value.items()
        )

    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value) or "Not available"

    return str(value)


def render_details(title, value):
    """Display a detail only when a value exists."""
    if value is not None and value != "":
        st.markdown(f"**{title}**")
        st.write(display_value(value))


def evidence_to_dict(item):
    """Convert retrieved evidence into a displayable dictionary."""
    if hasattr(item, "model_dump"):
        return item.model_dump()

    if isinstance(item, dict):
        return item

    return {"content": str(item)}

# --------------------------------------------------
# Styling — preserve the original Streamlit theme
# --------------------------------------------------

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.25);
            padding: 16px;
            border-radius: 12px;
        }

        div[data-testid="stExpander"] {
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 10px;
        }

        h1, h2, h3 {
            letter-spacing: -0.3px;
        }

        div.stButton > button {
            border-radius: 8px;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Header
# --------------------------------------------------

header_left, header_right = st.columns([5, 1])

with header_left:
    st.title("🏥 Clinical Triage Dashboard")
    st.markdown(
        '<p class="dashboard-subtitle">'
        'Agentic clinical assessment · Patient encounter insights'
        '</p>',
        unsafe_allow_html=True,
    )

with header_right:
    st.markdown(
        """
        <div style="
            text-align: right;
            padding-top: 20px;
            color: #1769AA;
            font-weight: 600;
        ">
            ● System Dashboard
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None

if "last_stay_id" not in st.session_state:
    st.session_state.last_stay_id = None


# --------------------------------------------------
# Sidebar: patient selection
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 🩺 Patient Selection")
    st.caption("Choose an available patient encounter.")

    try:
        encounters = load_encounters()
        database_error = None
    except Exception as exc:
        encounters = []
        database_error = str(exc)

    if database_error:
        st.error("Could not load encounters from the database.")
        st.caption(database_error)
        selected_stay_id = None

    elif not encounters:
        st.warning("No encounters were found in the database.")
        selected_stay_id = None

    else:
        # Build a display label for every real encounter.
        encounter_lookup = {
            int(item["stay_id"]): item
            for item in encounters
        }

        encounter_ids = list(encounter_lookup.keys())

        def encounter_label(stay_id):
            item = encounter_lookup[stay_id]

            patient_id = item.get("subject_id")
            complaint = item.get("chiefcomplaint")

            label = (
                f"Patient {patient_id} · "
                f"Encounter {stay_id}"
            )

            if complaint:
                label += f" · {complaint}"

            return label

        # Prefer the existing demo encounter when it exists.
        default_id = (
            30094124 if 30094124 in encounter_ids
            else encounter_ids[0]
        )

        default_index = encounter_ids.index(default_id)

        selected_stay_id = st.selectbox(
            "Patient / Encounter",
            options=encounter_ids,
            index=default_index,
            format_func=encounter_label,
            help="Search by patient identifier, encounter ID, or complaint.",
        )

        selected_patient = encounter_lookup[selected_stay_id]

        st.markdown(
            f"""
            <div style="
                background: #EEF5FC;
                border: 1px solid #D6E6F6;
                border-radius: 10px;
                padding: 12px;
                margin: 8px 0 16px 0;
            ">
                <div style="color:#64748B;font-size:0.8rem;">
                    SELECTED PATIENT
                </div>
                <div style="
                    color:#163B65;
                    font-size:1.05rem;
                    font-weight:700;
                ">
                    {selected_patient.get("subject_id", "N/A")}
                </div>
                <div style="color:#475569;font-size:0.85rem;">
                    Encounter ID: {selected_stay_id}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Pipeline Workflow")

    st.markdown(
        """
        <div style="line-height:2;">
            <span style="color:#1769AA;">01</span>
            &nbsp; Patient data ingestion<br>
            <span style="color:#1769AA;">02</span>
            &nbsp; Historical evidence retrieval<br>
            <span style="color:#1769AA;">03</span>
            &nbsp; Clinical summarization<br>
            <span style="color:#1769AA;">04</span>
            &nbsp; NEWS2 and ESI assessment
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    run_clicked = st.button(
        "▶  Run Clinical Triage",
        type="primary",
        use_container_width=True,
        disabled=selected_stay_id is None,
    )

    if st.button("Clear Results", use_container_width=True):
        st.session_state.pipeline_result = None
        st.session_state.last_stay_id = None
        st.rerun()


# --------------------------------------------------
# Execute the existing four-agent pipeline
# --------------------------------------------------

if run_clicked and selected_stay_id is not None:
    st.session_state.pipeline_result = None
    st.session_state.last_stay_id = int(selected_stay_id)

    try:
        with st.spinner(
            "Processing encounter through the clinical triage pipeline..."
        ):
            pipeline = ClinicalTriagePipeline()
            result = pipeline.run_pipeline(int(selected_stay_id))

        st.session_state.pipeline_result = result
        st.success("Clinical triage pipeline completed successfully.")

    except (ValueError, RuntimeError) as exc:
        st.error(f"Pipeline execution failed: {exc}")

    except Exception:
        st.error(
            "An unexpected error occurred. Check the application logs "
            "for the detailed error."
        )


# --------------------------------------------------
# Display dashboard
# --------------------------------------------------

result = st.session_state.pipeline_result

if result is None:
    st.markdown("## Welcome")
    st.write(
        "Select a patient from the sidebar and run the pipeline "
        "to view the encounter overview, clinical summary, "
        "retrieved evidence, and triage assessment."
    )

    st.markdown("### Available Workflow")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("### 📥 Ingestion")
        st.caption("Collect encounter data and vital signs.")

    with col2:
        st.markdown("### 🔎 Retrieval")
        st.caption("Retrieve relevant historical evidence.")

    with col3:
        st.markdown("### 📝 Summary")
        st.caption("Generate a grounded clinical summary.")

    with col4:
        st.markdown("### 🩺 Assessment")
        st.caption("Calculate NEWS2 and produce an ESI assessment.")

else:
    patient = result.patient_payload
    triage = result.triage_results
    evidence = result.filtered_evidence

    # ----------------------------------------------
    # Encounter overview
    # ----------------------------------------------

    st.markdown("## Encounter Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Encounter ID",
            display_value(get_field(patient, "stay_id")),
        )

    with col2:
        st.metric(
            "NEWS2 Score",
            display_value(get_field(triage, "news2_score")),
        )

    with col3:
        st.metric(
            "Historical Records Retrieved",
            len(evidence),
        )

    st.write("")

    # ----------------------------------------------
    # Patient and encounter information
    # ----------------------------------------------

    st.markdown("## Patient Information")

    with st.container(border=True):
        col1, col2 = st.columns(2)

        with col1:
            render_details("Age", get_field(patient, "age"))
            render_details("Sex", get_field(patient, "gender"))
            render_details(
                "Chief Complaint",
                get_field(patient, "chief_complaint"),
            )

        with col2:
            render_details(
                "Arrival Time",
                get_field(patient, "arrival_time"),
            )
            render_details(
                "Vital Signs",
                get_field(patient, "vitals"),
            )
            render_details(
                "Current Encounter",
                get_field(patient, "current_encounter"),
            )

    # ----------------------------------------------
    # Clinical summary
    # ----------------------------------------------

    st.markdown("## Clinical Summary")

    with st.container(border=True):
        summary = result.clinical_summary

        if summary:
            st.markdown(summary)
        else:
            st.info("No clinical summary was returned.")

    # ----------------------------------------------
    # Retrieved historical evidence
    # ----------------------------------------------

    st.markdown("## Retrieved Historical Evidence")
    st.caption(f"{len(evidence)} record(s) retrieved")

    if evidence:
        for index, item in enumerate(evidence, start=1):
            item_data = evidence_to_dict(item)

            with st.expander(
                f"Evidence {index}",
                expanded=(index == 1),
            ):
                for key, value in item_data.items():
                    label = key.replace("_", " ").title()
                    render_details(label, value)
    else:
        st.info(
            "No historical evidence was retrieved for this encounter."
        )

    # ----------------------------------------------
    # Triage assessment
    # ----------------------------------------------

    st.markdown("## Triage Assessment")

    assessment = get_field(triage, "triage_assessment")

    with st.container(border=True):
        if assessment:
            st.markdown(assessment)
        else:
            st.info("No triage assessment was returned.")

    # ----------------------------------------------
    # Footer
    # ----------------------------------------------

    st.divider()

    st.caption(
        "Clinical Triage Dashboard  ·  "
        "Agentic workflow: ingestion → retrieval → summarization → assessment"
    )