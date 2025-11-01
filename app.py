import streamlit as st
st.set_page_config(page_title="MediAssist", page_icon="🩺", layout="wide")

import fitz  # PyMuPDF
from transformers import pipeline
import re
import torch

# -----------------------------
# Load AI Summarization Model
# -----------------------------
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# -----------------------------
# Extract Vitals
# -----------------------------
def extract_vitals(text):
    vitals = {}

    bp = re.search(r"(Blood Pressure|BP)[:\- ]*\s*(\d{2,3}/\d{2,3})", text, re.IGNORECASE)
    if bp:
        vitals["Blood Pressure"] = bp.group(2)

    hr = re.search(r"(Heart Rate|Pulse)[:\- ]*\s*(\d{2,3})", text, re.IGNORECASE)
    if hr:
        vitals["Heart Rate"] = hr.group(2) + " bpm"

    glucose = re.search(r"(Glucose|Blood Glucose|Sugar|Fasting Glucose)[:\-\s]*\(?[A-Za-z]*\)?[:\-\s]*(\d{2,3})", text, re.IGNORECASE)
    if glucose:
        vitals["Blood Glucose"] = glucose.group(2) + " mg/dL"

    temp = re.search(r"(Temp|Temperature)[:\- ]*\s*(\d{2,3}\.?\d*)", text, re.IGNORECASE)
    if temp:
        vitals["Temperature"] = temp.group(2) + " °F"

    spo2 = re.search(r"(SpO2|Oxygen)[:\- ]*\s*(\d{2,3})", text, re.IGNORECASE)
    if spo2:
        vitals["Oxygen Saturation"] = spo2.group(2) + " %"

    hb = re.search(r"(Hb|Hemoglobin)[:\- ]*\s*(\d{1,2}\.?\d*)", text, re.IGNORECASE)
    if hb:
        vitals["Hemoglobin"] = hb.group(2) + " g/dL"

    return vitals

# -----------------------------
# Classify Severity
# -----------------------------
def classify_severity(vitals):
    alerts = []

    bp = vitals.get("Blood Pressure")
    if bp:
        try:
            systolic, diastolic = map(int, bp.split("/"))
            if systolic > 140 or diastolic > 90:
                alerts.append("⚠️ High Blood Pressure detected.")
            elif systolic < 90 or diastolic < 60:
                alerts.append("⚠️ Low Blood Pressure detected.")
        except:
            pass

    glucose = vitals.get("Blood Glucose")
    if glucose:
        val = int(re.findall(r"\d+", glucose)[0])
        if val > 140:
            alerts.append("⚠️ High Blood Sugar detected.")
        elif val < 70:
            alerts.append("⚠️ Low Blood Sugar detected.")

    hr = vitals.get("Heart Rate")
    if hr:
        val = int(re.findall(r"\d+", hr)[0])
        if val > 100:
            alerts.append("⚠️ High Heart Rate detected.")
        elif val < 60:
            alerts.append("⚠️ Low Heart Rate detected.")

    temp = vitals.get("Temperature")
    if temp:
        val = float(re.findall(r"\d+\.?\d*", temp)[0])
        if val > 100.4:
            alerts.append("⚠️ Fever detected.")
        elif val < 95:
            alerts.append("⚠️ Low Body Temperature detected.")

    return alerts

# -----------------------------
# PDF Text Extraction
# -----------------------------
def extract_text_from_pdf(uploaded_file):
    pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in pdf:
        text += page.get_text()
    return text

# -----------------------------
# Streamlit App Layout
# -----------------------------
# -----------------------------
# Streamlit App Layout
# -----------------------------
st.title("🩺 MediAssist – AI-Powered Medical Report Summarizer")
st.caption("Built for NavHacks 2025 | Quick • Smart • Actionable")

uploaded_file = st.file_uploader("📄 Upload your medical report (PDF)", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting report content..."):
        text = extract_text_from_pdf(uploaded_file)

    st.subheader("📝 Extracted Report Text")
    st.text_area("Report Content", text, height=200)

    # Extract vitals
    vitals = extract_vitals(text)
    st.subheader("📋 Key Vitals Extracted")
    if vitals:
        for k, v in vitals.items():
            st.write(f"**{k}:** {v}")
    else:
        st.info("No key vitals detected in this report.")

    # Classify severity
    alerts = classify_severity(vitals)
    st.subheader("🚨 Severity Analysis")
    if alerts:
        for a in alerts:
            st.error(a)
    else:
        st.success("✅ All vitals appear within normal range.")

    # Generate AI summary
    if st.button("🧠 Generate AI Summary"):
        with st.spinner("Summarizing report..."):
            clean_text = text.replace("\n", " ")
            summary = summarizer(clean_text[:1024], max_length=180, min_length=60, do_sample=False)
            st.subheader("💡 Summary")
            st.write(summary[0]['summary_text'])

    st.caption("⚙️ Powered by Streamlit + Transformers")

