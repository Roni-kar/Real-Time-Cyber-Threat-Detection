"""Packet-to-feature transformation and lightweight flow accounting.

The collector emits a stable, JSON-serialisable packet event. It intentionally
does *not* claim to recreate every NSL-KDD flow feature from a single live
packet. Instead, it records directly observable packet and short-lived flow
measurements that can be mapped by the backend's deployment feature adapter.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from ipaddress import ip_address
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

try:  # Importing this module should remain possible in documentation tooling.
    from scapy.layers.inet import ICMP, IP, TCP, UDP
except ImportError:  # pragma: no cover - used only without the runtime dependency
    ICMP = IP = TCP = UDP = None  # type: ignore[assignment]


Endpoint = Tuple[str, int]
FlowKey = Tuple[str, Endpoint, Endpoint]


@dataclass
class _FlowState:
    first_seen: float
    last_seen: float
    packet_count: int = 0
    byte_count: int = 0


@dataclass(frozen=True)
class FlowSnapshot:
    """Aggregate measurements for the bidirectional transport flow."""

    packet_count: int
    byte_count: int
    duration_seconds: float
    packets_per_second: float
    bytes_per_second: float


class FlowTracker:
    """Bounded in-memory tracker of bidirectional IP/transport flows.

    The OrderedDict permits predictable eviction under high traffic volume,
    preventing an unattended capture process from growing without bound.
    """

    def __init__(self, idle_seconds: float = 120.0, max_flows: int = 10_000) -> None:
        self.idle_seconds = idle_seconds
        self.max_flows = max_flows
        self._flows: "OrderedDict[FlowKey, _FlowState]" = OrderedDict()

    @staticmethod
    def make_key(
        protocol: str,
        source_ip: str,
        source_port: Optional[int],
        destination_ip: str,
        destination_port: Optional[int],
    ) -> FlowKey:
        """Return a direction-independent flow key for a packet's endpoints."""

        first: Endpoint = (source_ip, int(source_port or 0))
        second: Endpoint = (destination_ip, int(destination_port or 0))
        endpoints = tuple(sorted((first, second)))
        return protocol, endpoints[0], endpoints[1]

    def _evict_expired(self, current_time: float) -> None:
        expired = [
            key
            for key, state in self._flows.items()
            if current_time - state.last_seen > self.idle_seconds
        ]
        for key in expired:
            self._flows.pop(key, None)
        while len(self._flows) > self.max_flows:
            self._flows.popitem(last=False)

    def update(self, key: FlowKey, packet_size: int, observed_at: float) -> FlowSnapshot:
        """Record one packet and return the updated aggregate features."""

        self._evict_expired(observed_at)
        state = self._flows.get(key)
        if state is None:
            while len(self._flows) >= self.max_flows:
                self._flows.popitem(last=False)
            state = _FlowState(first_seen=observed_at, last_seen=observed_at)
            self._flows[key] = state
        else:
            self._flows.move_to_end(key)
            # Packet capture timestamps can be slightly out of order.
            state.last_seen = max(state.last_seen, observed_at)

        state.packet_count += 1
        state.byte_count += packet_size
        duration = max(0.0, state.last_seen - state.first_seen)
        rate_window = max(duration, 1.0)
        return FlowSnapshot(
            packet_count=state.packet_count,
            byte_count=state.byte_count,
            duration_seconds=round(duration, 6),
            packets_per_second=round(state.packet_count / rate_window, 6),
            bytes_per_second=round(state.byte_count / rate_window, 6),
        )

    @property
    def tracked_flow_count(self) -> int:
        return len(self._flows)


def _safe_private_address(value: str) -> bool:
    try:
        return ip_address(value).is_private
    except ValueError:
        return False


def _packet_timestamp(packet: Any) -> tuple[datetime, float]:
    """Obtain a UTC timestamp from Scapy, falling back only when unavailable."""

    try:
        epoch_seconds = float(packet.time)
    except (AttributeError, TypeError, ValueError):
        timestamp = datetime.now(timezone.utc)
        return timestamp, timestamp.timestamp()
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc), epoch_seconds


def _tcp_flag_text(packet: Any) -> Optional[str]:
    if TCP is None or not packet.haslayer(TCP):
        return None
    return str(packet[TCP].sprintf("%TCP.flags%"))


class PacketFeatureExtractor:
    """Extract a normalised event from a Scapy IPv4 packet.

    ``extract`` returns ``None`` for non-IPv4 packets; callers can safely use
    it as a Scapy callback without logging routine ARP/IPv6 traffic as errors.
    """

    def __init__(
        self,
        flow_tracker: FlowTracker,
        large_packet_threshold: int = 1400,
        high_flow_rate_threshold: float = 100.0,
        capture_interface: Optional[str] = None,
    ) -> None:
        self.flow_tracker = flow_tracker
        self.large_packet_threshold = large_packet_threshold
        self.high_flow_rate_threshold = high_flow_rate_threshold
        self.capture_interface = capture_interface

    def extract(self, packet: Any) -> Optional[Dict[str, Any]]:
        """Convert one Scapy packet into the API's documented event contract."""

        if IP is None:
            raise RuntimeError("Scapy is required for packet feature extraction")
        if not packet.haslayer(IP):
            return None

        ip_layer = packet[IP]
        source_ip = str(ip_layer.src)
        destination_ip = str(ip_layer.dst)
        protocol = "OTHER"
        source_port: Optional[int] = None
        destination_port: Optional[int] = None

        if TCP is not None and packet.haslayer(TCP):
            protocol = "TCP"
            source_port = int(packet[TCP].sport)
            destination_port = int(packet[TCP].dport)
        elif UDP is not None and packet.haslayer(UDP):
            protocol = "UDP"
            source_port = int(packet[UDP].sport)
            destination_port = int(packet[UDP].dport)
        elif ICMP is not None and packet.haslayer(ICMP):
            protocol = "ICMP"

        packet_size = len(packet)
        # IP.len excludes link-layer framing. It is useful as an ML feature but
        # may be absent on crafted packets, hence the safe fallback.
        ip_total_length = int(getattr(ip_layer, "len", packet_size) or packet_size)
        ip_header_length = int(getattr(ip_layer, "ihl", 5) or 5) * 4
        transport_header_length = 0
        if protocol == "TCP":
            transport_header_length = int(getattr(packet[TCP], "dataofs", 5) or 5) * 4
        elif protocol == "UDP":
            transport_header_length = 8
        payload_size = max(0, ip_total_length - ip_header_length - transport_header_length)

        timestamp, observed_at = _packet_timestamp(packet)
        flow_key = self.flow_tracker.make_key(
            protocol, source_ip, source_port, destination_ip, destination_port
        )
        flow = self.flow_tracker.update(flow_key, packet_size, observed_at)
        tcp_flags = _tcp_flag_text(packet)
        indicators = self._anomaly_indicators(packet, packet_size, tcp_flags, flow)

        return {
            "event_id": str(uuid4()),
            "timestamp": timestamp.isoformat(),
            "source_ip": source_ip,
            "destination_ip": destination_ip,
            "source_port": source_port,
            "destination_port": destination_port,
            "protocol": protocol,
            "protocol_number": int(getattr(ip_layer, "proto", 0) or 0),
            "packet_size": packet_size,
            "payload_size": payload_size,
            "ttl": int(getattr(ip_layer, "ttl", 0) or 0),
            "tcp_flags": tcp_flags,
            "flow_packet_count": flow.packet_count,
            "flow_byte_count": flow.byte_count,
            "flow_duration_seconds": flow.duration_seconds,
            "packets_per_second": flow.packets_per_second,
            "bytes_per_second": flow.bytes_per_second,
            "is_private_source": _safe_private_address(source_ip),
            "is_private_destination": _safe_private_address(destination_ip),
            "anomaly_indicators": indicators,
            "capture_interface": self.capture_interface,
        }

    def _anomaly_indicators(
        self,
        packet: Any,
        packet_size: int,
        tcp_flags: Optional[str],
        flow: FlowSnapshot,
    ) -> list[str]:
        """Return transparent observation flags, not a substitute for ML inference."""

        indicators: list[str] = []
        if packet_size >= self.large_packet_threshold:
            indicators.append("large_packet")
        if flow.packets_per_second >= self.high_flow_rate_threshold:
            indicators.append("high_flow_rate")
        if tcp_flags and "S" in tcp_flags and "A" not in tcp_flags:
            indicators.append("tcp_syn_without_ack")

        if IP is not None:
            ip_layer = packet[IP]
            flags = str(getattr(ip_layer, "flags", ""))
            fragment_offset = int(getattr(ip_layer, "frag", 0) or 0)
            if "MF" in flags or fragment_offset > 0:
                indicators.append("ip_fragment")
        return indicators
