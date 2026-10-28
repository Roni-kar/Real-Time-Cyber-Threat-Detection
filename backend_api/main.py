from fastapi import FastAPI
import joblib
from datetime import datetime

app = FastAPI(title="Cyber Threat Detection API")

# In-memory storage for live network packets
packet_logs = []

print("Initializing Server...")
try:
    model = joblib.load("ai_model/cyber_threat_model.pkl")
    print("✅ AI Model successfully loaded!")
except Exception as e:
    model = None
    print(f"❌ Error loading model: {e}")

@app.get("/")
def health_check():
    return {"status": "online", "model_loaded": model is not None}

@app.post("/analyze")
def analyze_traffic(packet_data: dict):
    protocol_map = {"TCP": 1, "UDP": 2, "Other": 0}
    proto_val = protocol_map.get(packet_data.get("protocol", "Other"), 0)
    size_val = packet_data.get("size", 0)

    # Run prediction using Sounak's model (with fallback if feature count differs)
    prediction_label = "Normal"
    try:
        if model is not None:
            # Adjust features list if Sounak's model expects more inputs
            pred = model.predict([[proto_val, size_val]])[0]
            prediction_label = "Threat Detected" if pred == 1 else "Normal"
    except Exception:
        # Fallback heuristic if feature dimensions mismatch during testing
        prediction_label = "Threat Detected" if size_val > 1400 else "Normal"

    log_entry = {
        "Timestamp": datetime.now().strftime("%H:%M:%S"),
        "Source IP": packet_data.get("source_ip", "Unknown"),
        "Destination IP": packet_data.get("destination_ip", "Unknown"),
        "Protocol": packet_data.get("protocol", "Other"),
        "Size (Bytes)": size_val,
        "AI Prediction": prediction_label
    }
    
    packet_logs.insert(0, log_entry)  # Keep newest packets at the top
    del packet_logs[100:]             # Keep only the latest 100 packets
    
    return {"status": "processed", "prediction": prediction_label}

@app.get("/logs")
def get_logs():
    threats = sum(1 for p in packet_logs if p["AI Prediction"] == "Threat Detected")
    return {
        "total_scanned": len(packet_logs),
        "threats_detected": threats,
        "logs": packet_logs
    }