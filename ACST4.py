# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 15:36:36 2025

@author: brysonj
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from fpdf import FPDF

# --- Title ---
st.title("Concussion Recovery Tracker")

# --- Patient Information ---
patient_name = st.text_input("Enter Patient Name:")
if patient_name:
    st.write(f"Tracking symptoms for: **{patient_name}**")

# --- Daily Symptom Input ---
days = list(range(1, 11))  # Days 1 to 10

headache_scores = []
dizziness_scores = []
memory_scores = []
mood_scores = []
fatigue_scores = []

st.header("Daily Symptom Input")

severity_levels = ["None", "Mild", "Moderate", "Severe"]
score_mapping = {"None": 0, "Mild": 1, "Moderate": 2, "Severe": 3}

for day in days:
    st.subheader(f"Day {day}")
    headache = st.selectbox(f"Headache Severity (Day {day}):", severity_levels, key=f"headache_{day}")
    dizziness = st.selectbox(f"Dizziness/Vertigo Severity (Day {day}):", severity_levels, key=f"dizziness_{day}")
    memory = st.selectbox(f"Memory Problems Severity (Day {day}):", severity_levels, key=f"memory_{day}")
    mood = st.selectbox(f"Mood Changes Severity (Day {day}):", severity_levels, key=f"mood_{day}")
    fatigue = st.selectbox(f"Fatigue Severity (Day {day}):", severity_levels, key=f"fatigue_{day}")

    headache_scores.append(score_mapping[headache])
    dizziness_scores.append(score_mapping[dizziness])
    memory_scores.append(score_mapping[memory])
    mood_scores.append(score_mapping[mood])
    fatigue_scores.append(score_mapping[fatigue])

# --- Create Symptom DataFrame ---
symptom_data = pd.DataFrame({
    "Day": days,
    "Headache": headache_scores,
    "Dizziness/Vertigo": dizziness_scores,
    "Memory Problems": memory_scores,
    "Mood Changes": mood_scores,
    "Fatigue": fatigue_scores
})

# --- Download Symptom Report ---
st.header("Download Symptom Report")

csv = symptom_data.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Symptom Report as CSV",
    data=csv,
    file_name=f"{patient_name}_symptom_report.csv",
    mime='text/csv',
)

# --- Plot Symptoms Over Time ---
st.header("Symptom Trends Over 10 Days")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(days, headache_scores, label="Headache", marker="o", color="red")
ax.plot(days, dizziness_scores, label="Dizziness/Vertigo", marker="x", color="blue")
ax.plot(days, memory_scores, label="Memory Problems", marker="^", color="green")
ax.plot(days, mood_scores, label="Mood Changes", marker="s", color="purple")
ax.plot(days, fatigue_scores, label="Fatigue", marker="*", color="orange")
ax.set_xlabel("Day")
ax.set_ylabel("Severity (0=None to 3=Severe)")
ax.set_title("Symptom Severity Over Time")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# --- CBF and ICP Recovery Simulation ---
st.header("Modeled Brain Recovery (CBF and ICP)")

hours = np.linspace(0, 240, 2400)  # 10 days, 0.1 hr resolution
ICP_baseline = 10  # mmHg
CBF_baseline = 50  # mL/100g/min

delta_ICP = 20
delta_CBF = 0.3 * CBF_baseline

tau_ICP = 80  # hours
tau_CBF = 100

event_time = 24  # concussion at 24 hours

ICP = np.full_like(hours, ICP_baseline)
CBF = np.full_like(hours, CBF_baseline)

# Apply concussion effects after event_time
event_start_index = np.searchsorted(hours, event_time)
ICP[event_start_index:] = ICP_baseline + delta_ICP * np.exp(-(hours[event_start_index:] - event_time) / tau_ICP)
CBF[event_start_index:] = CBF_baseline - delta_CBF * np.exp(-(hours[event_start_index:] - event_time) / tau_CBF)

# --- Calculate brain stress hour-by-hour ---
brain_stress_hourly = []

for idx, hour in enumerate(hours):
    if hour < event_time:
        brain_stress_hourly.append(0)
    else:
        icp_stress = (ICP[idx] - ICP_baseline) / delta_ICP
        cbf_stress = (CBF_baseline - CBF[idx]) / delta_CBF
        stress = (icp_stress + cbf_stress) / 2
        brain_stress_hourly.append(stress)

# --- Plot Brain Stress Hour-by-Hour ---
st.header("Modeled Brain Stress Hour-by-Hour")

fig4, ax4 = plt.subplots(figsize=(12, 6))
ax4.plot(hours / 24, brain_stress_hourly, color='black', label='Modeled Brain Stress (Hourly)')
ax4.axvspan(1, 3, color='red', alpha=0.2, label='Acute Phase (Days 1-3)')
ax4.axvspan(3, 10, color='orange', alpha=0.2, label='Recovery Phase (Days 3-10)')
ax4.axhline(0.2, color='green', linestyle='--', label='Target Safe Stress Level')
ax4.set_xlabel('Time (Days)')
ax4.set_ylabel('Brain Stress (0-1 scale)')
ax4.set_title('Modeled Brain Stress Hour-by-Hour After Concussion')
ax4.grid(True)
ax4.legend()
st.pyplot(fig4)

# --- Recovery Recommendation ---
st.header("Recovery Recommendation")

average_symptoms = np.array(headache_scores) + np.array(dizziness_scores) + np.array(memory_scores) + np.array(mood_scores) + np.array(fatigue_scores)
average_symptoms = average_symptoms / 5
final_symptom_score = np.mean(average_symptoms[-3:])

if final_symptom_score <= 0.5:
    st.success("Strong recovery! Gradual return to activities recommended.")
elif final_symptom_score <= 1.5:
    st.warning("Moderate symptoms still present. Caution is advised.")
else:
    st.error("Significant symptoms. You should not return to normal activities yet. See a healthcare provider.")

# --- Generate Full PDF Report ---
st.header("Download Full PDF Report")

if st.button("Generate PDF Report"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt=f"Concussion Recovery Report", ln=True, align="C")
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Patient Name: {patient_name}", ln=True)
    pdf.cell(200, 10, txt=f"---------------------------------", ln=True)
    pdf.ln(5)
    pdf.cell(200, 10, txt="Symptom Scores (0=None, 3=Severe):", ln=True)
    pdf.ln(5)

    for index, row in symptom_data.iterrows():
        day = row['Day']
        scores = f"Headache: {row['Headache']}, Dizziness: {row['Dizziness/Vertigo']}, Memory: {row['Memory Problems']}, Mood: {row['Mood Changes']}, Fatigue: {row['Fatigue']}"
        pdf.cell(200, 10, txt=f"Day {day}: {scores}", ln=True)

    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Recovery Recommendation:", ln=True)
    if final_symptom_score <= 0.5:
        pdf.cell(200, 10, txt="Strong recovery. Gradual return to activities recommended.", ln=True)
    elif final_symptom_score <= 1.5:
        pdf.cell(200, 10, txt="Moderate symptoms still present. Caution advised.", ln=True)
    else:
        pdf.cell(200, 10, txt="Significant symptoms. See a healthcare provider.", ln=True)

    pdf_buffer = BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)

    st.download_button(
        label="Download Full PDF Report",
        data=pdf_buffer,
        file_name=f"{patient_name}_Concussion_Report.pdf",
        mime="application/pdf"
    )
