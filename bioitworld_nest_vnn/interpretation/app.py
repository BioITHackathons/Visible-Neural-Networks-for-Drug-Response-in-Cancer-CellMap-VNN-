"""Streamlit dashboard for NeST-VNN LLM interpretations.

Run:  streamlit run interpretation/app.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from interpretation import db

st.set_page_config(page_title="NeST-VNN Interpretations", layout="wide")

DB_PATH = db.DB_PATH


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def load_patients(study_id, label, pred_class):
    return db.list_patients(
        study_id=study_id or None,
        label=label or None,
        pred_class=pred_class or None,
        db_path=DB_PATH,
    )


def badge(text: str, color: str) -> str:
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:0.85em">{text}</span>'


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("NeST-VNN")
    st.caption("LLM Interpretation Dashboard")

    page = st.radio("Page", ["Patient Explorer", "Population Overview"])

    st.divider()

    # Get available study/label combos from db
    try:
        with db.get_conn(DB_PATH) as conn:
            combos = conn.execute(
                "SELECT DISTINCT study_id, label FROM patients ORDER BY study_id, label"
            ).fetchall()
        study_options = sorted({r["study_id"] for r in combos})
        label_options = sorted({r["label"] for r in combos})
    except Exception:
        study_options, label_options = [], []

    sel_study = st.selectbox("Study", ["(all)"] + study_options)
    sel_label = st.selectbox("Label", ["(all)"] + label_options)
    sel_class = st.selectbox("Predicted Class", ["(all)", "high_risk", "low_risk"])

    study_filter = None if sel_study == "(all)" else sel_study
    label_filter = None if sel_label == "(all)" else sel_label
    class_filter = None if sel_class == "(all)" else sel_class


# ── Page: Patient Explorer ────────────────────────────────────────────────────

if page == "Patient Explorer":
    st.header("Patient Explorer")

    patients = load_patients(study_filter, label_filter, class_filter)

    if not patients:
        st.info("No patients found. Run the pipeline first:\n\n"
                "`python interpretation/run_pipeline.py <study_id> <label> --limit 10`")
        st.stop()

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total patients", len(patients))
    c2.metric("High risk", sum(1 for p in patients if p["predicted_class"] == "high_risk"))
    c3.metric("Low risk", sum(1 for p in patients if p["predicted_class"] == "low_risk"))
    avg_fga = sum(p["fraction_genome_altered"] or 0 for p in patients) / len(patients)
    c4.metric("Avg FGA", f"{avg_fga:.3f}")

    # Patient list table
    df = pd.DataFrame([{
        "Patient ID": p["patient_id"],
        "Class": p["predicted_class"],
        "Probability": round(p["predicted_score"] or 0, 4),
        "Sample Type": p["sample_type"],
        "Mutations": p["mutation_count"],
        "FGA": p["fraction_genome_altered"],
        "Study": p["study_id"],
        "Label": p["label"],
    } for p in patients])

    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Patient Detail")

    pid_options = [p["patient_id"] for p in patients]
    sel_pid = st.selectbox("Select patient", pid_options)
    sel_study_for_pid = next(p["study_id"] for p in patients if p["patient_id"] == sel_pid)
    sel_label_for_pid = next(p["label"] for p in patients if p["patient_id"] == sel_pid)

    patient = db.get_patient(sel_pid, sel_study_for_pid, sel_label_for_pid, DB_PATH)
    if not patient:
        st.warning("Patient record not found.")
        st.stop()

    # Prediction banner
    prob = patient["predicted_score"] or 0
    pred_cls = patient["predicted_class"] or ""
    color = "#e74c3c" if pred_cls == "high_risk" else "#27ae60"
    st.markdown(
        f"### {patient['patient_id']}&nbsp;&nbsp;"
        + badge(pred_cls.replace("_", " ").upper(), color)
        + f"&nbsp; probability: <b>{prob:.4f}</b>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Sample Type", patient["sample_type"] or "—")
    col2.metric("Mutation Count", f"{patient['mutation_count']:.1f}" if patient["mutation_count"] is not None else "—")
    col3.metric("FGA", f"{patient['fraction_genome_altered']:.3f}" if patient["fraction_genome_altered"] is not None else "—")

    # NeST cards
    nests = db.get_nests(sel_pid, sel_study_for_pid, sel_label_for_pid, DB_PATH)
    if nests:
        st.subheader("Top Biological Systems (NeSTs)")
        for nest in nests:
            dir_color = "#e74c3c" if nest["outcome_direction"] == "worsened" else "#27ae60"
            with st.expander(
                f"**{nest['nest_id']}** — {nest['pathway_name'] or '(pathway TBD)'} "
                f"| importance: {nest['importance_score']:.4f}"
                f"| {nest['outcome_direction'] or ''}",
                expanded=(nest["rank"] == 1),
            ):
                c1, c2, c3 = st.columns(3)
                c1.metric("Importance Score", f"{nest['importance_score']:.4f}")
                c2.metric("Patient RLIPP", f"{nest['rlipp_score']:.4f}" if nest["rlipp_score"] is not None else "—")
                c3.metric("Population RLIPP", f"{nest['population_rlipp']:.4f}" if nest["population_rlipp"] is not None else "—")

                if nest.get("reactome_link") and nest["reactome_link"] not in ("N/A", ""):
                    st.markdown(f"[Reactome Pathway]({nest['reactome_link']})")

                if nest.get("biological_explanation"):
                    st.markdown(f"**Biology:** {nest['biological_explanation']}")
                if nest.get("clinical_reasoning"):
                    st.markdown(f"**Clinical:** {nest['clinical_reasoning']}")

                # Gene table for this NEST
                genes = db.get_genes(sel_pid, sel_study_for_pid, sel_label_for_pid, nest["nest_id"], DB_PATH)
                if genes:
                    gene_rows = []
                    for g in genes:
                        try:
                            drug_list = json.loads(g["drugs"]) if g["drugs"] else []
                        except Exception:
                            drug_list = [g["drugs"]] if g["drugs"] else []
                        gene_rows.append({
                            "Gene": g["gene_name"],
                            "Alteration": g["alteration_type"],
                            "Direction": g["outcome_direction"],
                            "Biological Role": g["biological_role"],
                            "Targeted Therapies": ", ".join(drug_list) if drug_list else "—",
                        })
                    st.dataframe(pd.DataFrame(gene_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No NEST interpretations stored for this patient yet.")

    # Full LLM markdown
    if patient.get("llm_raw_markdown"):
        with st.expander("Full LLM Reasoning (raw markdown)"):
            st.markdown(patient["llm_raw_markdown"])


# ── Page: Population Overview ─────────────────────────────────────────────────

elif page == "Population Overview":
    st.header("Population Overview")

    patients = load_patients(study_filter, label_filter, class_filter)
    if not patients:
        st.info("No processed patients found.")
        st.stop()

    st.metric("Patients with LLM interpretations", len(patients))

    # NEST frequency across patients
    nest_counts: dict[str, int] = {}
    nest_rlipp: dict[str, float] = {}

    with db.get_conn(DB_PATH) as conn:
        rows = conn.execute("""
            SELECT nest_id, COUNT(*) as cnt, AVG(population_rlipp) as avg_rlipp
            FROM patient_nests
            WHERE 1=1
            {study_filter_sql}
            {label_filter_sql}
            GROUP BY nest_id
            ORDER BY cnt DESC
        """.replace(
            "{study_filter_sql}", f"AND study_id = '{study_filter}'" if study_filter else ""
        ).replace(
            "{label_filter_sql}", f"AND label = '{label_filter}'" if label_filter else ""
        )).fetchall()

    if rows:
        nest_df = pd.DataFrame([dict(r) for r in rows])
        st.subheader("NEST Frequency (how often each system is a top-3 driver)")
        st.bar_chart(nest_df.set_index("nest_id")["cnt"])

        st.subheader("NEST Details")
        st.dataframe(nest_df, use_container_width=True, hide_index=True)

    # Top genes across patients
    with db.get_conn(DB_PATH) as conn:
        gene_rows = conn.execute("""
            SELECT gene_name, alteration_type, COUNT(*) as cnt
            FROM patient_genes
            WHERE 1=1
            {study_filter_sql}
            {label_filter_sql}
            GROUP BY gene_name, alteration_type
            ORDER BY cnt DESC
            LIMIT 30
        """.replace(
            "{study_filter_sql}", f"AND study_id = '{study_filter}'" if study_filter else ""
        ).replace(
            "{label_filter_sql}", f"AND label = '{label_filter}'" if label_filter else ""
        )).fetchall()

    if gene_rows:
        st.subheader("Top Recurrently Important Genes")
        gdf = pd.DataFrame([dict(r) for r in gene_rows])
        st.bar_chart(gdf.set_index("gene_name")["cnt"])
        st.dataframe(gdf, use_container_width=True, hide_index=True)

    # Outcome class distribution
    st.subheader("Outcome Distribution")
    class_counts = pd.Series([p["predicted_class"] for p in patients]).value_counts()
    st.bar_chart(class_counts)
