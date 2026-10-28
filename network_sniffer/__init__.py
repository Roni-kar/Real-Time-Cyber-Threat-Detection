"""Live packet capture utilities for the cyber threat detection system.

The package deliberately keeps packet acquisition separate from feature
normalisation and delivery. This makes the packet schema testable without
requiring administrator privileges or an active network interface.
"""

from .config import SnifferConfig, load_config
from .features import FlowTracker, PacketFeatureExtractor

__all__ = ["FlowTracker", "PacketFeatureExtractor", "SnifferConfig", "load_config"]
