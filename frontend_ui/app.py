import streamlit as st
import requests
import pandas as pd

# Using localhost so you can test everything on your own laptop first
BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Cyber Threat Monitor", page_icon="🛡️", layout="wide")

st.title("🛡️ Real-Time Cyber Threat Detection")
st.markdown("Monitoring live network traffic anomalies using the trained Machine Learning model.")

# Sidebar Status
with st.sidebar:
    st.header("System Status")
    try:
        res = requests.get(f"{BACKEND_URL}/", timeout=2)
        if res.status_code == 200:
            st.success("✅ Connected to Backend API")
        else:
            st.warning("⚠️ Backend returned an error")
    except Exception:
        st.error("❌ Backend Offline. Start Uvicorn first.")

    if st.button("🔄 Refresh Live Feed"):
        st.rerun()

# Fetch live packet logs from Backend
total_scanned = 0
threats_detected = 0
logs_data = []

try:
    log_res = requests.get(f"{BACKEND_URL}/logs", timeout=2)
    if log_res.status_code == 200:
        data = log_res.json()
        total_scanned = data.get("total_scanned", 0)
        threats_detected = data.get("threats_detected", 0)
        logs_data = data.get("logs", [])
except Exception:
    pass

# Top-level Metrics
col1, col2, col3 = st.columns(3)
col1.metric(label="Packets Scanned", value=total_scanned)
col2.metric(label="Threats Detected", value=threats_detected)
col3.metric(label="System State", value="Alert" if threats_detected > 0 else "Secure")

st.subheader("Live Network Traffic Log")

if logs_data:
    df = pd.DataFrame(logs_data)
    st.dataframe(df, use_container_width=True)
else:
    st.info("Waiting for incoming packets from the Network Sniffer...")
    empty_df = pd.DataFrame(columns=[
        "Timestamp", "Source IP", "Destination IP", "Protocol", "Size (Bytes)", "AI Prediction"
    ])
    st.dataframe(empty_df, use_container_width=True)