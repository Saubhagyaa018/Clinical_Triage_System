import streamlit as st
from pipeline import ClinicalTriagePipeline

st.set_page_config(
    page_title="Agentic Clinical Triage System",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Agentic Clinical Triage & Risk Assessment System")
st.caption("A 4-Agent Sequential Pipeline for Emergency Room Decision Support")

@st.cache_resource
def load_pipeline():
    return ClinicalTriagePipeline(csv_file_path="data/raw_patients.csv")

pipeline = load_pipeline()

# Sidebar - Patient Selector
st.sidebar.header("Patient Intake")
patient_id = st.sidebar.selectbox("Select Patient ID", ["P001", "P002", "P003"])

if st.sidebar.button("Run Clinical Triage Pipeline", type="primary"):
    with st.spinner("Executing Multi-Agent Pipeline..."):
        output = pipeline.run_pipeline(patient_id)

    payload = output['patient_payload']
    evidence = output['filtered_evidence']
    summary = output['clinical_summary']
    triage = output['triage_results']

    # Header Metrics
    st.subheader(f"Patient: {payload['name']} (ID: {payload['patient_id']})")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Age / Gender", f"{payload['age']} / {payload['gender']}")
    col2.metric("Heart Rate", f"{payload['vitals']['heart_rate']} bpm")
    col3.metric("Blood Pressure", f"{payload['vitals']['bp_systolic']}/{payload['vitals']['bp_diastolic']}")
    col4.metric("SpO2", f"{payload['vitals']['spo2']}%")
    col5.metric("NEWS2 Score", f"{triage['news2_score']}")

    st.markdown("---")

    # Main Dashboard Columns
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.header("1. Grounded Semantic Evidence (Agent 2 - Qdrant)")
        st.write(f"**Chief Complaint:** {payload['chief_complaint']}")
        
        for item in evidence:
            with st.expander(f"Fact: {item['clinical_fact']} (Combined Score: {item['combined_relevance_score']})"):
                st.write(f"- Semantic Similarity: `{item['semantic_similarity']}`")
                st.write(f"- Temporal Relevance: `{item['temporal_relevance']}`")

        st.header("2. Grounded Clinical Summary (Agent 3 - Gemini)")
        st.markdown(summary)

    with right_col:
        st.header("3. Risk Assessment & Triage (Agent 4)")
        
        # Risk Badge Color
        if triage['news2_score'] >= 7:
            st.error(f"**NEWS2 Level:** {triage['news2_risk']}")
        elif triage['news2_score'] >= 5:
            st.warning(f"**NEWS2 Level:** {triage['news2_risk']}")
        else:
            st.success(f"**NEWS2 Level:** {triage['news2_risk']}")

        st.markdown(triage['triage_assessment'])