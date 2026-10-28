"""FastAPI service for validated ML-assisted network traffic analysis.

The service persists every accepted analysis decision in SQLite. It never uses
a packet-size heuristic as a substitute for a missing or incompatible model:
when no versioned NSL-KDD artifact is available, the returned prediction is
explicitly marked ``model_unavailable`` and no threat alert is created.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from ai_model.inference import IntrusionDetectionModel, ModelLoadError

from .config import Settings
from .schemas import (
    AlertsResponse,
    AnalysisResponse,
    HealthResponse,
    ModelStatusResponse,
    PacketAnalysisRequest,
    PaginatedLogsResponse,
    PredictionResponse,
    SeverityLevel,
    StatisticsResponse,
)
from .storage import TrafficRepository


LOGGER = logging.getLogger(__name__)
SETTINGS = Settings.from_environment()
REPOSITORY = TrafficRepository(SETTINGS.database_path, max_records=SETTINGS.max_log_records)


class ModelRuntime:
    """Own model lifecycle and make a loading failure visible to API consumers."""

    def __init__(self) -> None:
        self.model: IntrusionDetectionModel | None = None
        self.error: str | None = None

    def load(self) -> None:
        try:
            self.model = IntrusionDetectionModel.from_artifact(SETTINGS.model_path)
            self.error = None
            LOGGER.info(
                "Loaded %s version %s from %s",
                self.model.model_name,
                self.model.model_version,
                SETTINGS.model_path,
            )
        except ModelLoadError as exc:
            self.model = None
            self.error = str(exc)
            LOGGER.warning("ML model is unavailable: %s", exc)


RUNTIME = ModelRuntime()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    logging.basicConfig(
        level=getattr(logging, SETTINGS.log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    REPOSITORY.initialize()
    RUNTIME.load()
    yield


app = FastAPI(
    title=SETTINGS.app_name,
    description=(
        "Validated, auditable packet/flow analysis backed by a versioned "
        "NSL-KDD machine-learning pipeline."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if SETTINGS.enable_docs else None,
    redoc_url="/redoc" if SETTINGS.enable_docs else None,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(SETTINGS.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _packet_mapping(packet: PacketAnalysisRequest) -> dict[str, Any]:
    """Build JSON-like telemetry for the ML feature mapper across Pydantic versions."""
    return {
        "timestamp": packet.timestamp.isoformat(),
        "source_ip": str(packet.source_ip),
        "destination_ip": str(packet.destination_ip),
        "protocol": packet.protocol,
        "source_port": packet.source_port,
        "destination_port": packet.destination_port,
        "packet_size": packet.packet_size,
        "ttl": packet.ttl,
        "tcp_flags": packet.tcp_flags,
        "flow_packet_count": packet.flow_packet_count,
        "flow_byte_count": packet.flow_byte_count,
        "flow_duration_seconds": packet.flow_duration_seconds,
        "packets_per_second": packet.packets_per_second,
        "bytes_per_second": packet.bytes_per_second,
        "is_private_source": packet.is_private_source,
        "is_private_destination": packet.is_private_destination,
        "anomaly_indicators": packet.anomaly_indicators,
        "capture_interface": packet.capture_interface,
    }


def _severity(value: str) -> SeverityLevel:
    """Map the model's non-alert `none` state to an informational log level."""
    try:
        return SeverityLevel(value)
    except ValueError:
        return SeverityLevel.INFO


def _unavailable_outcome(reason: str) -> dict[str, Any]:
    return {
        "label": "unavailable",
        "is_threat": False,
        "confidence": None,
        "source": "model_unavailable",
        "model_version": None,
        "explanation": [reason],
        "severity": SeverityLevel.INFO,
    }


def _analyse(packet: PacketAnalysisRequest) -> tuple[dict[str, Any], SeverityLevel]:
    """Run a versioned model once, retaining an honest non-ML failure result."""
    if RUNTIME.model is None:
        outcome = _unavailable_outcome(RUNTIME.error or "No trained model artifact is loaded.")
        return outcome, SeverityLevel.INFO
    try:
        result = RUNTIME.model.predict_live_packet(_packet_mapping(packet))
    except Exception as exc:  # Keep the service auditable if a model fails at runtime.
        LOGGER.exception("Inference failed for packet from %s", packet.source_ip)
        outcome = _unavailable_outcome(f"Model inference failed: {exc}")
        return outcome, SeverityLevel.INFO

    explanation = [
        f"Model={result.model_name}",
        f"Live feature quality={result.feature_quality}",
    ]
    if packet.anomaly_indicators:
        explanation.append("Observed indicators=" + ", ".join(packet.anomaly_indicators[:5]))
    outcome = {
        "label": result.label,
        "is_threat": result.is_threat,
        "confidence": result.confidence,
        "source": "ml_pipeline",
        "model_version": result.model_version,
        "explanation": explanation,
        "severity": _severity(result.severity),
    }
    return outcome, outcome["severity"]


def _health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc),
        environment=SETTINGS.environment,
        database="ready",
        model_loaded=RUNTIME.model is not None,
    )


@app.get("/", response_model=HealthResponse, include_in_schema=False)
@app.get("/health", response_model=HealthResponse, tags=["service"])
@app.get("/api/v1/health", response_model=HealthResponse, tags=["service"])
def health() -> HealthResponse:
    """Report whether the API, database, and versioned model are ready."""
    return _health()


@app.get("/model/status", response_model=ModelStatusResponse, tags=["model"])
@app.get("/api/v1/model/status", response_model=ModelStatusResponse, tags=["model"])
def model_status() -> ModelStatusResponse:
    """Return deployment provenance without exposing the model internals."""
    if RUNTIME.model is None:
        return ModelStatusResponse(
            loaded=False,
            usable_for_live_packets=False,
            model_path=str(SETTINGS.model_path),
            message=RUNTIME.error or "No trained model has been loaded.",
        )
    status = RUNTIME.model.status()
    metadata = RUNTIME.model.metadata
    return ModelStatusResponse(
        loaded=True,
        usable_for_live_packets=True,
        model_path=status["artifact_path"],
        model_type=status["model_name"],
        model_version=status["model_version"],
        trained_at=status.get("trained_at"),
        expected_feature_count=len(status.get("feature_schema", [])),
        classes=[str(item) for item in metadata.get("classes", [])],
        evaluation=dict(metadata.get("test_metrics", {})),
        message="Versioned NSL-KDD pipeline is ready for live packet/flow mapping.",
    )


@app.post("/analyze", response_model=AnalysisResponse, status_code=201, tags=["analysis"])
@app.post("/api/v1/packets/analyze", response_model=AnalysisResponse, status_code=201, tags=["analysis"])
def analyze_packet(packet: PacketAnalysisRequest) -> AnalysisResponse:
    """Validate, classify, persist, and optionally alert on one packet/flow event."""
    outcome, severity = _analyse(packet)
    try:
        packet_id, alert = REPOSITORY.save_analysis(packet, outcome, severity)
    except Exception as exc:
        LOGGER.exception("Could not persist analysis result")
        raise HTTPException(status_code=500, detail="Unable to persist the analysis result.") from exc

    return AnalysisResponse(
        packet_id=packet_id,
        analyzed_at=datetime.now(timezone.utc),
        prediction=PredictionResponse(
            label=outcome["label"],
            is_threat=bool(outcome["is_threat"]),
            confidence=outcome["confidence"],
            source=outcome["source"],
            model_version=outcome.get("model_version"),
            explanation=list(outcome["explanation"]),
        ),
        severity=severity,
        alert=alert,
    )


@app.get("/logs", response_model=PaginatedLogsResponse, tags=["telemetry"])
@app.get("/api/v1/logs", response_model=PaginatedLogsResponse, tags=["telemetry"])
def logs(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    threats_only: bool = Query(default=False),
) -> PaginatedLogsResponse:
    """Read persisted packet analysis logs, newest first."""
    total, entries = REPOSITORY.list_logs(limit, offset, threats_only=threats_only)
    return PaginatedLogsResponse(total=total, limit=limit, offset=offset, logs=entries)


@app.get("/alerts", response_model=AlertsResponse, tags=["telemetry"])
@app.get("/api/v1/alerts", response_model=AlertsResponse, tags=["telemetry"])
def alerts(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    severity: SeverityLevel | None = None,
    status: str | None = Query(default=None, max_length=32),
) -> AlertsResponse:
    """Read severity-aware alerts retained by the local telemetry repository."""
    total, entries = REPOSITORY.list_alerts(limit, offset, severity=severity, status=status)
    return AlertsResponse(total=total, limit=limit, offset=offset, alerts=entries)


@app.get("/statistics", response_model=StatisticsResponse, tags=["telemetry"])
@app.get("/api/v1/statistics", response_model=StatisticsResponse, tags=["telemetry"])
def statistics() -> StatisticsResponse:
    """Return dashboard-ready aggregate counts, protocol trends, and suspicious IPs."""
    return StatisticsResponse(generated_at=datetime.now(timezone.utc), **REPOSITORY.statistics())
