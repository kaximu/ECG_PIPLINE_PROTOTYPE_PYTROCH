import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from supabase import create_client

# ----------------------------
# 1. Connect to Supabase
# ----------------------------
url = "https://pbumynpwuptllvjihpia.supabase.co"   # 🔑 replace with your project URL
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBidW15bnB3dXB0bGx2amlocGlhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjYzMDQwNzcsImV4cCI6MjA0MTg4MDA3N30.Ra0j4r_4AtH6U4eZ6JTfascVBmTedusthre-ROg5Lcs"              # 🔑 replace with your API key (service_role for backend)
              
supabase = create_client(url, key)

# ----------------------------
# 2. Load Data from Supabase
# ----------------------------
def load_data():
    patients = supabase.table("patients").select("*").execute().data
    ecg_records = supabase.table("ecg_records").select("*").execute().data
    return pd.DataFrame(patients), pd.DataFrame(ecg_records)

df_patients, df_ecg = load_data()

# ----------------------------
# 3. Streamlit Dashboard
# ----------------------------
st.set_page_config(page_title="3-Lead ECG Dashboard", layout="wide")
st.title("🫀 Patient-Linked 3-Lead ECG Dashboard")

if df_patients.empty:
    st.warning("⚠️ No patients found. Run the simulator to insert some data.")
else:
    # Sidebar: choose patient
    patient_list = df_patients["name"].tolist()
    selected_patient = st.sidebar.selectbox("Select a patient:", patient_list)

    # Sidebar: Delete patient and ECG
    st.sidebar.markdown("---")
    st.sidebar.header("⚠️ Danger Zone")
    if st.sidebar.button("Delete patient and all ECG data", type="primary"):
        # Get patient id
        patient_id = df_patients[df_patients["name"] == selected_patient].iloc[0]["id"]
        # Delete ECG records for this patient
        supabase.table("ecg_records").delete().eq("patient_id", patient_id).execute()
        # Delete patient
        supabase.table("patients").delete().eq("id", patient_id).execute()
        st.sidebar.success(f"Deleted {selected_patient} and all their ECG data.")
        st.rerun()

    # Patient info
    patient_info = df_patients[df_patients["name"] == selected_patient].iloc[0]
    st.subheader("👤 Patient Information")
    st.write(f"- **Name:** {patient_info['name']}")
    st.write(f"- **Age:** {patient_info['age']}")
    st.write(f"- **Gender:** {patient_info['gender']}")

    # Filter ECG records for this patient
    patient_ecg = df_ecg[df_ecg["patient_id"] == patient_info["id"]]

    if patient_ecg.empty:
        st.warning("⚠️ No ECG data available for this patient.")
    else:
        st.subheader("📊 ECG Records Overview")
        st.dataframe(patient_ecg[["created_at", "sampling_rate", "prediction"]])

        # Create tabs for each ECG record
        record_tabs = st.tabs([
            f"ECG {i+1} | {row['prediction']} ({row['created_at'][:19]})"
            for i, (_, row) in enumerate(patient_ecg.iterrows())
        ])

        # Plot each ECG in its own tab
        for i, (tab, (_, record)) in enumerate(zip(record_tabs, patient_ecg.iterrows())):
            with tab:
                lead_i = np.array(record["lead_i"])
                lead_ii = np.array(record["lead_ii"])
                lead_v2 = np.array(record["lead_v2"])
                fs = record.get("sampling_rate", 500)

                t = np.arange(len(lead_i)) / fs

                fig = make_subplots(
                    rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                    subplot_titles=(f"Lead I ({record['prediction']})", "Lead II", "Lead V2")
                )

                fig.add_trace(go.Scatter(x=t, y=lead_i, mode="lines", name="Lead I"), row=1, col=1)
                fig.add_trace(go.Scatter(x=t, y=lead_ii, mode="lines", name="Lead II"), row=2, col=1)
                fig.add_trace(go.Scatter(x=t, y=lead_v2, mode="lines", name="Lead V2"), row=3, col=1)

                fig.update_layout(
                    height=700, showlegend=False,
                    margin=dict(l=40, r=20, t=40, b=40)
                )
                fig.update_yaxes(title="mV")
                fig.update_xaxes(title="Time (s)")

                # ✅ Unique key to avoid duplicate chart error
                st.plotly_chart(fig, use_container_width=True, key=f"plot_{record['id']}_{i}")
