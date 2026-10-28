"""Configuration for the live network collector.

All runtime settings are sourced from environment variables so a capture can
be moved between a lab machine, a local demo, and a monitored host without
editing source code. The defaults are intentionally local and conservative.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, received {value!r}") from exc
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}, received {parsed}")
    return parsed


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric, received {value!r}") from exc
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}, received {parsed}")
    return parsed


@dataclass(frozen=True)
class SnifferConfig:
    """Runtime configuration for :mod:`network_sniffer.sniffer`."""

    backend_url: str
    interface: Optional[str]
    bpf_filter: str
    packet_limit: int
    promiscuous_mode: bool
    forward_to_api: bool
    api_timeout_seconds: float
    verify_tls: bool
    flow_idle_seconds: float
    max_tracked_flows: int
    large_packet_threshold: int
    high_flow_rate_threshold: float
    log_level: str


def load_config() -> SnifferConfig:
    """Load and validate the collector's environment based configuration.

    ``CYBER_PACKET_LIMIT=0`` represents an unbounded capture. For a safe
    classroom demonstration, set it to a small positive number.
    """

    endpoint = os.getenv(
        "CYBER_API_URL", "http://127.0.0.1:8000/api/v1/packets/analyze"
    ).strip()
    if not endpoint.startswith(("http://", "https://")):
        raise ValueError("CYBER_API_URL must start with http:// or https://")

    interface = os.getenv("CYBER_SNIFFER_INTERFACE", "").strip() or None
    bpf_filter = os.getenv("CYBER_SNIFFER_BPF_FILTER", "ip").strip()
    log_level = os.getenv("CYBER_LOG_LEVEL", "INFO").upper()

    return SnifferConfig(
        backend_url=endpoint.rstrip("/"),
        interface=interface,
        bpf_filter=bpf_filter,
        packet_limit=_env_int("CYBER_PACKET_LIMIT", 0),
        promiscuous_mode=_env_bool("CYBER_PROMISCUOUS_MODE", True),
        forward_to_api=_env_bool("CYBER_FORWARD_TO_API", True),
        api_timeout_seconds=_env_float("CYBER_API_TIMEOUT_SECONDS", 2.5, 0.1),
        verify_tls=_env_bool("CYBER_VERIFY_TLS", True),
        flow_idle_seconds=_env_float("CYBER_FLOW_IDLE_SECONDS", 120.0, 1.0),
        max_tracked_flows=_env_int("CYBER_MAX_TRACKED_FLOWS", 10000, 1),
        large_packet_threshold=_env_int("CYBER_LARGE_PACKET_THRESHOLD", 1400, 64),
        high_flow_rate_threshold=_env_float("CYBER_HIGH_FLOW_RATE", 100.0, 1.0),
        log_level=log_level,
    )
