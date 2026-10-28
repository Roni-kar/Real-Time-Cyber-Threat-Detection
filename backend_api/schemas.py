"""Validated request and response contracts exposed by the API."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, IPvAnyAddress, validator


class SeverityLevel(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PacketAnalysisRequest(BaseModel):
    """Normalized packet/flow data accepted from a trusted sensor.

    A single packet does not contain all flow statistics.  The sniffer maintains
    those counters and supplies zero/default values until enough packets have
    been seen.  This makes the API usable with both live capture and replayed
    experiment data.
    """

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_ip: IPvAnyAddress
    destination_ip: IPvAnyAddress
    protocol: str = Field(default="OTHER", min_length=1, max_length=16)
    source_port: Optional[int] = Field(default=None, ge=0, le=65535)
    destination_port: Optional[int] = Field(default=None, ge=0, le=65535)
    packet_size: int = Field(ge=0, le=262144)
    ttl: Optional[int] = Field(default=None, ge=0, le=255)
    tcp_flags: Optional[str] = Field(default=None, max_length=32)
    flow_packet_count: int = Field(default=1, ge=0)
    flow_byte_count: int = Field(default=0, ge=0)
    flow_duration_seconds: float = Field(default=0.0, ge=0.0)
    packets_per_second: float = Field(default=0.0, ge=0.0)
    bytes_per_second: float = Field(default=0.0, ge=0.0)
    is_private_source: Optional[bool] = None
    is_private_destination: Optional[bool] = None
    anomaly_indicators: List[str] = Field(default_factory=list, max_items=30)
    capture_interface: Optional[str] = Field(default=None, max_length=128)

    @validator("protocol")
    def normalize_protocol(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized in {"IP", "IPV4", "IPV6"}:
            return "OTHER"
        return normalized

    @validator("tcp_flags")
    def normalize_flags(cls, value: Optional[str]) -> Optional[str]:
        return value.strip().upper() if value else None

    @validator("anomaly_indicators", each_item=True)
    def validate_indicator(cls, value: str) -> str:
        value = value.strip().lower()
        if not value or len(value) > 100:
            raise ValueError("anomaly indicators must be non-empty labels of at most 100 characters")
        return value

    @validator("timestamp")
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


class AlertResponse(BaseModel):
    id: int
    packet_id: int
    timestamp: datetime
    severity: SeverityLevel
    title: str
    message: str
    source_ip: str
    destination_ip: str
    status: str = "open"


class PredictionResponse(BaseModel):
    label: str
    is_threat: bool
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    source: str
    model_version: Optional[str] = None
    explanation: List[str] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    status: str = "processed"
    packet_id: int
    analyzed_at: datetime
    prediction: PredictionResponse
    severity: SeverityLevel
    alert: Optional[AlertResponse] = None


class PacketLogResponse(BaseModel):
    id: int
    timestamp: datetime
    source_ip: str
    destination_ip: str
    protocol: str
    source_port: Optional[int] = None
    destination_port: Optional[int] = None
    packet_size: int
    flow_packet_count: int
    flow_duration_seconds: float
    anomaly_indicators: List[str] = Field(default_factory=list)
    prediction_label: str
    is_threat: bool
    confidence: Optional[float] = None
    severity: SeverityLevel
    prediction_source: str
    model_version: Optional[str] = None


class PaginatedLogsResponse(BaseModel):
    total: int
    limit: int
    offset: int
    logs: List[PacketLogResponse]


class AlertsResponse(BaseModel):
    total: int
    limit: int
    offset: int
    alerts: List[AlertResponse]


class ProtocolStatistic(BaseModel):
    protocol: str
    packets: int
    threats: int


class SuspiciousIP(BaseModel):
    ip_address: str
    threat_count: int
    highest_severity: SeverityLevel


class StatisticsResponse(BaseModel):
    generated_at: datetime
    total_packets: int
    threat_packets: int
    normal_packets: int
    threat_rate: float
    open_alerts: int
    severity_distribution: Dict[str, int]
    protocol_trends: List[ProtocolStatistic]
    suspicious_ips: List[SuspiciousIP]


class ModelStatusResponse(BaseModel):
    loaded: bool
    usable_for_live_packets: bool
    model_path: str
    model_type: Optional[str] = None
    model_version: Optional[str] = None
    trained_at: Optional[str] = None
    expected_feature_count: Optional[int] = None
    classes: List[str] = Field(default_factory=list)
    evaluation: Dict[str, Any] = Field(default_factory=dict)
    message: str


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    environment: str
    database: str
    model_loaded: bool
