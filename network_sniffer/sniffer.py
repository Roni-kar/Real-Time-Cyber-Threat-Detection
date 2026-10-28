"""Command-line Scapy collector for live network traffic.

Run from the repository root::

    python -m network_sniffer.sniffer --count 25

Packet capture usually requires Administrator/root privileges and an installed
capture driver (Npcap on Windows). The collector only observes IPv4 traffic by
default and never sends or modifies packets.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from typing import Any, Dict, Optional

from .client import BackendClient
from .config import SnifferConfig, load_config
from .features import FlowTracker, PacketFeatureExtractor


LOGGER = logging.getLogger(__name__)


class LivePacketProcessor:
    """Coordinate feature extraction, optional API delivery, and counters."""

    def __init__(self, config: SnifferConfig) -> None:
        self.config = config
        self.extractor = PacketFeatureExtractor(
            flow_tracker=FlowTracker(
                idle_seconds=config.flow_idle_seconds,
                max_flows=config.max_tracked_flows,
            ),
            large_packet_threshold=config.large_packet_threshold,
            high_flow_rate_threshold=config.high_flow_rate_threshold,
            capture_interface=config.interface,
        )
        self.client = (
            BackendClient(
                config.backend_url,
                timeout_seconds=config.api_timeout_seconds,
                verify_tls=config.verify_tls,
            )
            if config.forward_to_api
            else None
        )
        self.total_seen = 0
        self.ipv4_events = 0
        self.delivery_failures = 0

    def handle_packet(self, packet: Any) -> None:
        """Scapy callback that never lets an individual packet stop capture."""

        self.total_seen += 1
        try:
            event = self.extractor.extract(packet)
            if event is None:
                return
            self.ipv4_events += 1
            response: Optional[Dict[str, Any]] = None
            if self.client is not None:
                response = self.client.submit(event)
                if response is None:
                    self.delivery_failures += 1
            self._log_event(event, response)
        except Exception:  # Packet parsing errors must not halt a live sensor.
            LOGGER.exception("Unable to process captured packet")

    @staticmethod
    def _log_event(event: Dict[str, Any], response: Optional[Dict[str, Any]]) -> None:
        prediction = "not forwarded"
        if response is not None:
            prediction = str(
                response.get("prediction")
                or response.get("classification")
                or response.get("status")
                or "received"
            )
        LOGGER.info(
            "%s %s:%s -> %s:%s (%s bytes) | %s",
            event["protocol"],
            event["source_ip"],
            event["source_port"] if event["source_port"] is not None else "-",
            event["destination_ip"],
            event["destination_port"] if event["destination_port"] is not None else "-",
            event["packet_size"],
            prediction,
        )

    def close(self) -> None:
        if self.client is not None:
            self.client.close()


def run_capture(config: SnifferConfig) -> LivePacketProcessor:
    """Start synchronous capture and return processing counters after it stops."""

    try:
        from scapy.all import sniff
    except ImportError as exc:  # pragma: no cover - depends on local install
        raise RuntimeError(
            "Scapy is not installed. Run `pip install -r requirements.txt` first."
        ) from exc

    processor = LivePacketProcessor(config)
    capture_options: Dict[str, Any] = {
        "prn": processor.handle_packet,
        "store": False,
        "promisc": config.promiscuous_mode,
    }
    if config.interface:
        capture_options["iface"] = config.interface
    if config.bpf_filter:
        capture_options["filter"] = config.bpf_filter
    if config.packet_limit > 0:
        capture_options["count"] = config.packet_limit

    LOGGER.info(
        "Starting IPv4 capture%s; forwarding=%s, endpoint=%s",
        f" on {config.interface}" if config.interface else " on the default interface",
        config.forward_to_api,
        config.backend_url if config.forward_to_api else "disabled",
    )
    try:
        sniff(**capture_options)
    except KeyboardInterrupt:
        LOGGER.info("Capture stopped by user")
    finally:
        processor.close()
        LOGGER.info(
            "Capture summary: %d observed, %d IPv4 events, %d delivery failures, %d active flows",
            processor.total_seen,
            processor.ipv4_events,
            processor.delivery_failures,
            processor.extractor.flow_tracker.tracked_flow_count,
        )
    return processor


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture IPv4 traffic for ML-assisted analysis.")
    parser.add_argument("--count", type=int, help="Stop after this many captured packets (0 is unbounded).")
    parser.add_argument("--interface", help="Override CYBER_SNIFFER_INTERFACE for this run.")
    parser.add_argument("--no-forward", action="store_true", help="Extract and log events without contacting the API.")
    parser.add_argument("--print-config", action="store_true", help="Validate and print configuration without capturing.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        config = load_config()
        if args.count is not None:
            if args.count < 0:
                raise ValueError("--count must be zero or a positive integer")
            config = replace(config, packet_limit=args.count)
        if args.interface is not None:
            config = replace(config, interface=args.interface.strip() or None)
        if args.no_forward:
            config = replace(config, forward_to_api=False)
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if args.print_config:
        LOGGER.info("Effective sniffer configuration: %s", config)
        return 0

    try:
        run_capture(config)
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
