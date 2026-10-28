"""Streamlit observability dashboard for the cyber-threat detection service.

The dashboard intentionally contains no detection logic. It is a read-only
client for the FastAPI service, which keeps the monitoring interface safe to
run independently from the packet-capture process.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:  # Keeps the dashboard readable if optional UI tooling is absent.
    st_autorefresh = None


load_dotenv(Path(__file__).resolve().parents[1] / ".env")
DEFAULT_API_URL = "http://127.0.0.1:8000"
API_URL = os.getenv("THREAT_API_URL", os.getenv("BACKEND_URL", DEFAULT_API_URL)).rstrip("/")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("DASHBOARD_REQUEST_TIMEOUT", "3"))


st.set_page_config(
    page_title="SentinelFlow | Threat Monitoring",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.0rem; padding-bottom: 2rem; }
      [data-testid="stMetric"] {
          border: 1px solid rgba(49, 130, 206, .22);
          background: linear-gradient(135deg, rgba(13, 28, 48, .96), rgba(20, 45, 70, .92));
          border-radius: 12px; padding: 1rem;
      }
      .hero { padding: 1.25rem 1.5rem; border-radius: 14px; margin-bottom: 1.2rem;
          background: linear-gradient(100deg, #071a2f, #0b3b5d); border: 1px solid #1c628c; }
      .hero h1 { margin: 0; color: #ecf8ff; font-size: 2rem; }
      .hero p { margin: .4rem 0 0; color: #b8d8e9; }
      .caption { color: #7aa9c4; font-size: .86rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Live update")
    auto_refresh = st.toggle("Auto-refresh telemetry", value=False)
    refresh_seconds = st.slider("Refresh interval (seconds)", 3, 60, 8, disabled=not auto_refresh)
    if auto_refresh and st_autorefresh is None:
        st.warning("Install the dashboard dependencies to enable automatic refresh.")

if auto_refresh and st_autorefresh is not None:
    st_autorefresh(interval=refresh_seconds * 1000, key="threat_telemetry_refresh")


def _request(path: str) -> tuple[dict[str, Any] | list[Any] | None, str | None]:
    """Request an API resource without letting a temporary outage break the UI."""
    try:
        response = requests.get(f"{API_URL}{path}", timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as exc:
        return None, str(exc)
    except ValueError:
        return None, "The API returned a non-JSON response."


def _as_records(payload: Any, keys: Iterable[str]) -> list[dict[str, Any]]:
    """Support the current API contract and gracefully display older API builds."""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _value(payload: Any, *names: str, default: int | float | str = 0) -> Any:
    if not isinstance(payload, dict):
        return default
    for name in names:
        value = payload.get(name)
        if value is not None:
            return value
    return default


def _normalise_log_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame

    aliases = {
        "timestamp": "Timestamp",
        "created_at": "Timestamp",
        "source_ip": "Source IP",
        "destination_ip": "Destination IP",
        "protocol": "Protocol",
        "packet_size": "Packet Size",
        "size": "Packet Size",
        "prediction": "Prediction",
        "prediction_label": "Prediction",
        "classification": "Prediction",
        "is_threat": "Threat",
        "severity": "Severity",
        "confidence": "Confidence",
        "id": "ID",
    }
    frame = frame.rename(columns={key: value for key, value in aliases.items() if key in frame.columns})
    preferred = [
        "ID", "Timestamp", "Source IP", "Destination IP", "Protocol", "Packet Size",
        "Prediction", "Threat", "Severity", "Confidence",
    ]
    return frame[[column for column in preferred if column in frame.columns] +
                 [column for column in frame.columns if column not in preferred]]


def _render_connection_status(health: Any, error: str | None) -> None:
    with st.sidebar:
        st.header("Control panel")
        st.caption(f"API endpoint: `{API_URL}`")
        if error:
            st.error("Backend unavailable")
            st.caption(error)
        else:
            state = str(_value(health, "status", default="unknown")).title()
            model_loaded = bool(_value(health, "model_loaded", "model_available", default=False))
            st.success(f"Service: {state}")
            (st.success if model_loaded else st.warning)(
                "Model ready" if model_loaded else "Model unavailable"
            )
        st.divider()
        if st.button("Refresh telemetry", use_container_width=True, type="primary"):
            st.rerun()
        st.caption("Live capture runs separately; this page refreshes API-backed telemetry.")


health, health_error = _request("/health")
if health_error:
    # Avoid five serial timeouts when a laptop demo starts the UI before the API.
    stats, stats_error = None, health_error
    logs_payload, logs_error = None, health_error
    alerts_payload, alerts_error = None, health_error
    model_status = None
else:
    stats, stats_error = _request("/statistics")
    logs_payload, logs_error = _request("/logs?limit=100")
    alerts_payload, alerts_error = _request("/alerts?limit=100")
    model_status, _ = _request("/model/status")

_render_connection_status(health, health_error)

st.markdown(
    """
    <div class="hero">
      <h1>🛡️ SentinelFlow: Real-Time Cyber Threat Detection</h1>
      <p>Network traffic analysis, NSL-KDD–trained machine learning, and auditable security alerts.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if health_error:
    st.info(
        "Start the FastAPI service to populate this dashboard. The interface remains available so it can "
        "be demonstrated independently of packet capture."
    )
elif stats_error:
    st.warning("The API is reachable, but statistics are temporarily unavailable.")

total_packets = _value(stats, "total_packets", "total_scanned", "packets_scanned")
threat_count = _value(stats, "threat_packets", "threats_detected", "threat_count", "total_threats")
normal_count = _value(stats, "normal_packets", "normal_count")
alert_count = _value(stats, "open_alerts", "active_alerts", "alert_count", default=0)

metric_a, metric_b, metric_c, metric_d = st.columns(4)
metric_a.metric("Packets analysed", total_packets)
metric_b.metric("Threats detected", threat_count, delta=None if not threat_count else "Requires review")
metric_c.metric("Normal traffic", normal_count)
metric_d.metric("Alerts retained", alert_count)

logs = _as_records(logs_payload, ("items", "logs", "data", "records"))
alerts = _as_records(alerts_payload, ("items", "alerts", "data", "records"))
log_frame = _normalise_log_frame(logs)
alert_frame = _normalise_log_frame(alerts)

monitoring_tab, analytics_tab, alerts_tab, model_tab = st.tabs(
    ["Live monitoring", "Historical analytics", "Alert queue", "Model registry"]
)

with monitoring_tab:
    left, right = st.columns((1.7, 1))
    with left:
        st.subheader("Recent packet decisions")
        if log_frame.empty:
            st.info("No packet decisions have been recorded yet.")
        else:
            st.dataframe(log_frame, use_container_width=True, hide_index=True, height=360)
    with right:
        st.subheader("Protocol mix")
        if not log_frame.empty and "Protocol" in log_frame:
            protocol_counts = log_frame["Protocol"].fillna("Unknown").value_counts()
            st.bar_chart(protocol_counts, horizontal=True)
        else:
            st.caption("Protocol trend data will appear after traffic is received.")

        st.subheader("Severity distribution")
        severity_source = alert_frame if not alert_frame.empty else log_frame
        if not severity_source.empty and "Severity" in severity_source:
            st.bar_chart(severity_source["Severity"].fillna("informational").value_counts(), horizontal=True)
        else:
            st.caption("No severity-labelled events yet.")

with analytics_tab:
    st.subheader("Historical behaviour")
    if log_frame.empty:
        st.info("Analytics will become available as the service persists packet decisions.")
    else:
        graph_left, graph_right = st.columns(2)
        with graph_left:
            st.caption("Top source addresses in the retained observation window")
            if "Source IP" in log_frame:
                st.bar_chart(log_frame["Source IP"].fillna("Unknown").value_counts().head(10), horizontal=True)
        with graph_right:
            st.caption("Packet-size distribution")
            if "Packet Size" in log_frame:
                sizes = pd.to_numeric(log_frame["Packet Size"], errors="coerce").dropna()
                if not sizes.empty:
                    st.bar_chart(sizes.value_counts(bins=12, sort=False))

        st.subheader("Suspicious IPs")
        suspicious = log_frame.copy()
        if "Threat" in suspicious:
            suspicious = suspicious[suspicious["Threat"].astype(bool)]
        elif "Prediction" in suspicious:
            suspicious = suspicious[
                suspicious["Prediction"].astype(str).str.lower().str.contains("threat|attack|anomal", regex=True)
            ]
        if not suspicious.empty and "Source IP" in suspicious:
            ip_table = suspicious["Source IP"].fillna("Unknown").value_counts().rename_axis("Source IP").reset_index(name="Events")
            st.dataframe(ip_table, use_container_width=True, hide_index=True)
        else:
            st.caption("No suspicious source addresses in the current retention window.")

with alerts_tab:
    st.subheader("Actionable security alerts")
    if alert_frame.empty:
        st.success("No alerts are currently retained.")
    else:
        st.dataframe(alert_frame, use_container_width=True, hide_index=True, height=430)

with model_tab:
    st.subheader("Deployed model")
    if isinstance(model_status, dict):
        visible = {
            key.replace("_", " ").title(): value
            for key, value in model_status.items()
            if key not in {"feature_names", "metrics", "details"}
        }
        if visible:
            st.dataframe(pd.DataFrame([visible]), use_container_width=True, hide_index=True)
        metrics = model_status.get("metrics")
        if isinstance(metrics, dict) and metrics:
            st.caption("Evaluation metrics stored with the model artifact")
            st.json(metrics)
        details = model_status.get("details")
        if details:
            st.caption(str(details))
    else:
        st.info("Model metadata will be shown once a trained model artifact is available.")

errors = [error for error in (logs_error, alerts_error) if error]
if errors and not health_error:
    st.caption("Some telemetry resources could not be loaded: " + " | ".join(errors))

st.markdown(
    "<p class='caption'>SentinelFlow records operational telemetry for demonstration and research. "
    "Treat ML alerts as decision support, validate them before an incident response action.</p>",
    unsafe_allow_html=True,
)
