import os
import io
import json
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from supabase import create_client
from typing import Optional, Dict, List

# -----------------------------
# Config
# -----------------------------
RECONSTRUCTOR_PT = "ecgnet_reconstructor.pt"
CLASSIFIER_PT = "ecgnet_with_preprocessing.pt"
CLASS_PKL = "class_names.pkl"
CLASS_JSON = "class_names.json"
TARGET_LEN = 5000  # fixed length input

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://pbumynpwuptllvjihpia.supabase.co")
SUPABASE_KEY = os.getenv(
    "SUPABASE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9....",  # ⚠️ replace in production
)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load models/classes
# -----------------------------
@st.cache_resource
def load_reconstructor():
    model = torch.load(RECONSTRUCTOR_PT, map_location=device)
    model.eval()
    return model

@st.cache_resource
def load_classifier():
    model = torch.load(CLASSIFIER_PT, map_location=device)
    model.eval()
    return model

@st.cache_resource
def load_classes() -> List[str]:
    return joblib.load(CLASS_PKL)

@st.cache_resource
def load_class_fullnames() -> Dict[str, str]:
    try:
        with open(CLASS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

reconstructor = load_reconstructor()
classifier = load_classifier()
class_names = load_classes()
class_fullnames = load_class_fullnames()

# -----------------------------
# Utils
# -----------------------------
def preprocess_3lead(lead_i, lead_ii, lead_v2, target_len=TARGET_LEN):
    """Prepare 3-lead ECG as model input (1, time, 3)."""
    try:
        li = np.asarray(lead_i, dtype=np.float32)
        lii = np.asarray(lead_ii, dtype=np.float32)
        lv2 = np.asarray(lead_v2, dtype=np.float32)
    except Exception:
        return None

    n = min(len(li), len(lii), len(lv2))
    if n == 0:
        return None

    x = np.stack([li[:n], lii[:n], lv2[:n]], axis=-1)
    if n > target_len:
        x = x[:target_len]
    else:
        pad = np.zeros((target_len, 3), dtype=np.float32)
        pad[:n] = x
        x = pad

    x = np.expand_dims(x, axis=0)  # (1, time, 3)
    return torch.tensor(x, dtype=torch.float32).to(device)

def plot_ecg_12(ecg: np.ndarray):
    """Plot reconstructed 12-lead ECG stacked in 12 rows."""
    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF",
                  "V1", "V2", "V3", "V4", "V5", "V6"]
    fig, axes = plt.subplots(12, 1, figsize=(12, 18), sharex=True)
    for i in range(12):
        axes[i].plot(ecg[0, :, i], linewidth=0.8)
        axes[i].set_ylabel(lead_names[i], rotation=0, labelpad=30, fontsize=8)
        axes[i].grid(True, linestyle="--", alpha=0.5)
    axes[-1].set_xlabel("Time (samples)")
    plt.tight_layout()
    return fig

# -----------------------------
# UI
# -----------------------------
st.title("🫀 ECG Diagnosis System (PyTorch + Supabase Data)")

st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider("Detection Threshold (%)", 0, 100, 50, 1) / 100.0
top_k = st.sidebar.slider("Top-K predictions", 1, 12, 5, 1)

# Load patients + records
patients = supabase.table("patients").select("*").execute().data
records = supabase.table("ecg_records").select("*").execute().data
df_patients = pd.DataFrame(patients)
df_records = pd.DataFrame(records)

if df_patients.empty or df_records.empty:
    st.warning("⚠️ No patients or ECG data found in Supabase.")
    st.stop()

# Select patient
selected_patient = st.sidebar.selectbox("Select patient", df_patients["name"].tolist())
patient_info = df_patients[df_patients["name"] == selected_patient].iloc[0]

st.subheader("👤 Patient Information")
st.write(f"**Name:** {patient_info['name']}")
st.write(f"**Age:** {patient_info['age']}")
st.write(f"**Gender:** {patient_info['gender']}")

# Filter ECGs
patient_ecg = df_records[df_records["patient_id"] == patient_info["id"]]
if patient_ecg.empty:
    st.info("ℹ️ No ECGs linked to this patient.")
    st.stop()

# Select ECG record
record_options = [f"{i+1} | {row['created_at'][:19]} | {row['id']}"
                  for i, (_, row) in enumerate(patient_ecg.iterrows())]
selected = st.selectbox("Select ECG record", record_options)
sel_idx = record_options.index(selected)
record = list(patient_ecg.iterrows())[sel_idx][1]

if not all(k in record for k in ["lead_i", "lead_ii", "lead_v2"]):
    st.error("❌ This record does not have 3-lead data.")
    st.stop()

# -----------------------------
# Pipeline
# -----------------------------
X3 = preprocess_3lead(record["lead_i"], record["lead_ii"], record["lead_v2"])
if X3 is None:
    st.error("❌ Could not preprocess ECG leads.")
    st.stop()

try:
    with torch.no_grad():
        ecg_reconstructed = reconstructor(X3).cpu().numpy()
except Exception as e:
    st.error(f"❌ Reconstruction failed: {e}")
    st.stop()

try:
    with torch.no_grad():
        preds = classifier(torch.tensor(ecg_reconstructed, dtype=torch.float32).to(device))
        preds = preds.cpu().numpy()
        probs = 1 / (1 + np.exp(-preds))  # sigmoid
        probs = probs[0]
except Exception as e:
    st.error(f"❌ Classification failed: {e}")
    st.stop()

if len(probs) == 0:
    st.error("❌ No predictions computed.")
    st.stop()

bin_preds = (probs >= threshold).astype(int)

# Predictions
st.subheader("🩺 Predictions")
positive = [class_names[i] for i, v in enumerate(bin_preds) if v == 1]
if not positive:
    st.info(f"⚠️ No condition detected above threshold {threshold:.2f} → defaulting to **Normal**")
    pred_text = "Normal"
    st.success("✅ Normal – Normal ECG (0.500)")
else:
    pred_text = ", ".join(positive)
    for i in np.where(bin_preds == 1)[0]:
        abbr = class_names[i]
        fullname = class_fullnames.get(abbr, abbr)
        st.success(f"✅ {abbr} – {fullname} ({probs[i]:.3f})")

# Top-K
st.write(f"### Top-{top_k} predictions")
top_idx = np.argsort(probs)[::-1][:top_k]
for i in top_idx:
    abbr = class_names[i]
    fullname = class_fullnames.get(abbr, abbr)
    st.write(f"{abbr} – {fullname}: {probs[i]:.3f}")

# -----------------------------
# ECG Plot
# -----------------------------
st.write("### Reconstructed ECG (12 Standard Leads)")
fig = plot_ecg_12(ecg_reconstructed)
st.pyplot(fig)
