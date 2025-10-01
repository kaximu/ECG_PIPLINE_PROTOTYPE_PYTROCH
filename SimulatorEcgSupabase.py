import numpy as np
import pandas as pd
import streamlit as st
from supabase import create_client
from signalz.generators import ecgsyn
import plotly.graph_objects as go
from plotly.subplots import make_subplots   # ✅ Fix import

# ----------------------------
# 1. Connect to Supabase
# ----------------------------
url = "https://pbumynpwuptllvjihpia.supabase.co"   # 🔑 replace with your project URL
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBidW15bnB3dXB0bGx2amlocGlhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjYzMDQwNzcsImV4cCI6MjA0MTg4MDA3N30.Ra0j4r_4AtH6U4eZ6JTfascVBmTedusthre-ROg5Lcs"              # 🔑 replace with your API key (service_role for backend)
              # 🔑 replace with your service_role key
supabase = create_client(url, key)

# ----------------------------
# 2. ECG Simulation Functions
# ----------------------------
def simulate_ecgsyn(hr_mean=75, hr_std=1.0, duration_s=5, fs=500, noise_level=0.0):
    hr_std = max(hr_std, 1.0)

    # overshoot number of beats to ensure enough samples
    n_beats = int(duration_s * hr_mean / 60) * 2 + 10

    x, _ = ecgsyn(
        n=n_beats,
        hrmean=hr_mean,
        hrstd=hr_std,
        sfecg=fs,
        sfint=fs * 2,
    )

    # Clip safely to target length
    target_len = int(duration_s * fs)
    x = x[: min(target_len, len(x))]

    if noise_level > 0.0:
        x = x + noise_level * np.random.randn(len(x))

    return x

def generate_three_leads(base_signal):
    return {
        "Lead I": base_signal * 0.8,
        "Lead II": base_signal,
        "Lead V2": base_signal * 0.6 + np.roll(base_signal, 10)
    }

def simulate_condition(condition, duration_s, fs, noise):
    if condition == "normal":
        return simulate_ecgsyn(75, 1.0, duration_s, fs, noise)
    elif condition == "bradycardia":
        return simulate_ecgsyn(40, 1.0, duration_s, fs, noise)
    elif condition == "tachycardia":
        return simulate_ecgsyn(150, 1.0, duration_s, fs, noise)
    elif condition == "afib":
        return simulate_ecgsyn(75, 20.0, duration_s, fs, noise)
    elif condition == "pvc":
        base = simulate_ecgsyn(75, 2.0, duration_s, fs, noise)
        return base * 0.5  # simple PVC effect for demo
    else:
        raise ValueError("Unknown condition")

# ----------------------------
# 3. Save to Supabase
# ----------------------------
def save_to_supabase(patient_name, age, gender, fs, condition, leads_df):
    # Create or fetch patient
    existing = supabase.table("patients").select("*").eq("name", patient_name).execute()
    if existing.data:
        patient_id = existing.data[0]["id"]
    else:
        new_patient = supabase.table("patients").insert({
            "name": patient_name,
            "age": age,
            "gender": gender
        }).execute()
        patient_id = new_patient.data[0]["id"]

    # Insert ECG record linked to patient
    record = supabase.table("ecg_records").insert({
        "patient_id": patient_id,
        "sampling_rate": fs,
        "lead_i": leads_df["Lead I"].tolist(),
        "lead_ii": leads_df["Lead II"].tolist(),
        "lead_v2": leads_df["Lead V2"].tolist(),
        "prediction": condition
    }).execute()

    return record.data[0]["id"]

# ----------------------------
# 4. Streamlit UI
# ----------------------------
st.set_page_config(page_title="ECG Simulator", layout="wide")
st.title("🩺 ECG Simulation and Supabase Storage")

# Patient info
st.sidebar.header("👤 Patient Information")
patient_name = st.sidebar.text_input("Patient Name", "Simulated Patient")
patient_age = st.sidebar.number_input("Age", 30)
patient_gender = st.sidebar.selectbox("Gender", ["male", "female"])

# Simulation settings
st.sidebar.header("⚙️ Simulation Settings")
condition = st.sidebar.selectbox("Condition", ["normal", "bradycardia", "tachycardia", "afib", "pvc"])
fs = st.sidebar.selectbox("Sampling rate (Hz)", [250, 500, 1000], index=1)
duration = st.sidebar.slider("Duration (s)", 2, 20, 5)
noise = st.sidebar.slider("Noise level", 0.0, 0.5, 0.0, step=0.01)

if st.sidebar.button("▶️ Simulate & Save"):
    # Generate signal
    ecg = simulate_condition(condition, duration, fs, noise)
    leads = generate_three_leads(ecg)
    df = pd.DataFrame(leads)

    # Save to Supabase
    record_id = save_to_supabase(patient_name, patient_age, patient_gender, fs, condition, df)
    st.success(f"✅ ECG saved for {patient_name} (Record ID: {record_id})")

    # Plot the simulated leads
    t = np.arange(len(ecg)) / fs
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=("Lead I", "Lead II", "Lead V2"))
    fig.add_trace(go.Scatter(x=t, y=leads["Lead I"], mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=leads["Lead II"], mode="lines"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=leads["Lead V2"], mode="lines"), row=3, col=1)
    fig.update_layout(height=700, showlegend=False, margin=dict(l=40, r=20, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)
