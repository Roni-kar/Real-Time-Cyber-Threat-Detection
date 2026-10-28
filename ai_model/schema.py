"""NSL-KDD schema and deterministic live-traffic feature mapping.

NSL-KDD describes *connections*, whereas Scapy observes individual packets.
The mapping in this module is intentionally explicit about that distinction:
packet fields are used where they are observable and connection-level fields
are populated only from an optional flow tracker.  Unknown context is set to a
documented neutral default rather than invented.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


NSL_KDD_FEATURES: tuple[str, ...] = (
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
)

CATEGORICAL_FEATURES: tuple[str, ...] = ("protocol_type", "service", "flag")
NUMERICAL_FEATURES: tuple[str, ...] = tuple(
    feature for feature in NSL_KDD_FEATURES if feature not in CATEGORICAL_FEATURES
)
NSL_KDD_COLUMNS: tuple[str, ...] = NSL_KDD_FEATURES + ("label", "difficulty")


# Attack families used in NSL-KDD/KDD'99 research.  The mapping includes the
# attack names occurring in KDDTrain+ and the additional KDDTest+ names so that
# an external test set can be evaluated without changing the pipeline.
ATTACK_CATEGORY_MAP: dict[str, str] = {
    "normal": "normal",
    # denial of service
    "back": "dos",
    "land": "dos",
    "neptune": "dos",
    "pod": "dos",
    "smurf": "dos",
    "teardrop": "dos",
    "mailbomb": "dos",
    "apache2": "dos",
    "processtable": "dos",
    "udpstorm": "dos",
    # probing / reconnaissance
    "satan": "probe",
    "ipsweep": "probe",
    "nmap": "probe",
    "portsweep": "probe",
    "mscan": "probe",
    "saint": "probe",
    # remote-to-local
    "guess_passwd": "r2l",
    "ftp_write": "r2l",
    "imap": "r2l",
    "phf": "r2l",
    "multihop": "r2l",
    "warezmaster": "r2l",
    "warezclient": "r2l",
    "spy": "r2l",
    "xlock": "r2l",
    "xsnoop": "r2l",
    "snmpguess": "r2l",
    "snmpgetattack": "r2l",
    "httptunnel": "r2l",
    "sendmail": "r2l",
    "named": "r2l",
    "worm": "r2l",
    # user-to-root
    "buffer_overflow": "u2r",
    "loadmodule": "u2r",
    "rootkit": "u2r",
    "perl": "u2r",
    "sqlattack": "u2r",
    "xterm": "u2r",
    "ps": "u2r",
}


# Services are retained as named NSL-KDD values when a port is unambiguous.
# Unmapped ports are deliberately represented by ``other`` instead of a
# hard-coded service claim.
PORT_TO_SERVICE: dict[int, str] = {
    20: "ftp_data",
    21: "ftp",
    22: "ssh",
    23: "telnet",
    25: "smtp",
    43: "whois",
    53: "domain_u",
    69: "tftp_u",
    79: "finger",
    80: "http",
    88: "kerberos",
    109: "pop_2",
    110: "pop_3",
    111: "sunrpc",
    113: "auth",
    119: "nntp",
    123: "ntp_u",
    135: "epmap",
    137: "netbios_ns",
    139: "netbios_ssn",
    143: "imap4",
    161: "snmp",
    162: "snmptrap",
    179: "bgp",
    389: "ldap",
    443: "http_443",
    445: "microsoft_ds",
    514: "shell",
    515: "printer",
    993: "imap4",
    995: "pop_3",
}

_RATE_FEATURES = {
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
}

_BOOLEAN_FEATURES = {
    "land",
    "logged_in",
    "root_shell",
    "su_attempted",
    "is_host_login",
    "is_guest_login",
}


@dataclass(frozen=True)
class FeatureMappingDiagnostics:
    """Provenance of values passed from live telemetry to the ML model."""

    provided_fields: list[str]
    derived_fields: list[str]
    defaulted_context_features: list[str]

    @property
    def quality(self) -> str:
        """A concise confidence-in-input indicator, not model confidence."""

        if not self.defaulted_context_features:
            return "flow_enriched"
        if len(self.defaulted_context_features) <= 8:
            return "partially_enriched"
        return "packet_only"

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["quality"] = self.quality
        return result


def canonical_attack_label(value: Any) -> str:
    """Return a normalized NSL-KDD attack label.

    NSL-KDD files sometimes use a trailing period in exported copies, so it is
    removed here before target construction.
    """

    return str(value).strip().lower().rstrip(".")


def attack_category(value: Any) -> str:
    """Map a raw attack name to its standard NSL-KDD attack family."""

    label = canonical_attack_label(value)
    return ATTACK_CATEGORY_MAP.get(label, "unknown_attack")


def binary_label(value: Any) -> str:
    """Return the deployment-oriented ``normal``/``attack`` target label."""

    return "normal" if canonical_attack_label(value) == "normal" else "attack"


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result == result else default  # NaN-safe without numpy.


def _non_negative(value: Any, default: float = 0.0) -> float:
    return max(0.0, _number(value, default))


def _rate(value: Any, default: float = 0.0) -> float:
    return min(1.0, max(0.0, _number(value, default)))


def _integer(value: Any, default: int = 0) -> int:
    return int(round(_non_negative(value, default)))


def _value(record: Mapping[str, Any], *names: str) -> tuple[Any, str | None]:
    """Retrieve the first meaningful value from aliases in a telemetry record."""

    for name in names:
        value = record.get(name)
        if value is not None and value != "":
            return value, name
    return None, None


def canonical_protocol(value: Any) -> str:
    """Convert common telemetry protocol representations to NSL-KDD values."""

    text = str(value or "").strip().lower()
    aliases = {
        "6": "tcp",
        "tcp": "tcp",
        "17": "udp",
        "udp": "udp",
        "1": "icmp",
        "icmp": "icmp",
    }
    return aliases.get(text, "tcp" if not text else text)


def service_from_port(port: Any) -> str:
    """Map a destination port to a conservative NSL-KDD service value."""

    return PORT_TO_SERVICE.get(_integer(port, -1), "other")


def nsl_flag_from_tcp_flags(value: Any, protocol: str) -> str:
    """Approximate the NSL-KDD connection-state flag from observed TCP flags.

    The mapping is necessarily conservative for an individual packet.  A flow
    tracker may pass ``connection_state`` directly for a more reliable value.
    """

    if protocol != "tcp":
        return "SF"
    text = str(value or "").upper()
    if not text:
        return "S0"
    # Scapy can render flags as ``S``, ``SA`` or symbolic names.
    if "R" in text:
        return "REJ"
    if "S" in text and "A" not in text:
        return "S0"
    if "F" in text or "A" in text:
        return "SF"
    return "OTH"


def empty_nsl_kdd_feature_row() -> dict[str, Any]:
    """Create a neutral, schema-complete NSL-KDD connection record."""

    categorical_defaults = {"protocol_type": "tcp", "service": "other", "flag": "S0"}
    return {
        feature: categorical_defaults[feature] if feature in categorical_defaults else 0.0
        for feature in NSL_KDD_FEATURES
    }


def coerce_nsl_kdd_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a schema-complete, type-safe NSL-KDD feature record.

    This accepts direct NSL-KDD fields and is also used after the live-packet
    mapping.  Rates are clipped to [0, 1]; count and byte fields are nonnegative.
    """

    output = empty_nsl_kdd_feature_row()
    for feature in CATEGORICAL_FEATURES:
        value = record.get(feature, output[feature])
        if feature == "protocol_type":
            output[feature] = canonical_protocol(value)
        elif feature == "flag":
            # NSL-KDD connection-state labels are conventionally uppercase
            # (`SF`, `S0`, `REJ`); changing case would create unseen one-hot
            # categories at inference time.
            output[feature] = str(value or output[feature]).strip().upper()
        else:
            output[feature] = str(value or output[feature]).strip().lower()

    for feature in NUMERICAL_FEATURES:
        value = record.get(feature, output[feature])
        if feature in _RATE_FEATURES:
            output[feature] = _rate(value)
        elif feature in _BOOLEAN_FEATURES:
            output[feature] = int(bool(_number(value)))
        elif feature in {"duration", "src_bytes", "dst_bytes"}:
            output[feature] = _non_negative(value)
        else:
            output[feature] = _integer(value)
    return output


def live_packet_to_nsl_kdd(record: Mapping[str, Any]) -> tuple[dict[str, Any], FeatureMappingDiagnostics]:
    """Map standardized packet/flow telemetry to the 41 NSL-KDD features.

    Expected packet keys include ``source_ip``, ``destination_ip``,
    ``source_port``, ``destination_port``, ``protocol``, ``packet_size``,
    ``tcp_flags``, and optional flow statistics such as ``flow_duration_seconds``
    and ``flow_packet_count``.  Direct NSL-KDD feature names take precedence,
    which allows a flow aggregator to supply richer, measured context.
    """

    if not isinstance(record, Mapping):
        raise TypeError("Live packet telemetry must be a mapping/dictionary.")

    row = empty_nsl_kdd_feature_row()
    provided: list[str] = []
    derived: list[str] = []
    defaulted: list[str] = []

    # Direct values supplied by a flow enricher are more faithful than any
    # single-packet estimate, so copy them first.
    for feature in NSL_KDD_FEATURES:
        if feature in record and record[feature] is not None:
            row[feature] = record[feature]
            provided.append(feature)

    protocol_value, protocol_key = _value(record, "protocol", "protocol_name", "ip_protocol")
    if "protocol_type" not in provided and protocol_key:
        row["protocol_type"] = canonical_protocol(protocol_value)
        derived.append("protocol_type")

    destination_port, port_key = _value(record, "destination_port", "dst_port", "dport")
    service_value, service_key = _value(record, "service", "application_protocol")
    if "service" not in provided:
        if service_key:
            row["service"] = str(service_value).strip().lower()
            derived.append("service")
        elif port_key:
            row["service"] = service_from_port(destination_port)
            derived.append("service")

    state_value, state_key = _value(record, "connection_state", "flow_state", "tcp_flags", "flags")
    if "flag" not in provided:
        row["flag"] = nsl_flag_from_tcp_flags(state_value, canonical_protocol(row["protocol_type"]))
        if state_key:
            derived.append("flag")

    aliases: dict[str, tuple[str, ...]] = {
        "duration": ("flow_duration_seconds", "flow_duration", "duration_seconds"),
        "src_bytes": ("bytes_from_source", "source_bytes", "src_bytes_total"),
        "dst_bytes": ("bytes_to_destination", "destination_bytes", "dst_bytes_total"),
        "wrong_fragment": ("fragment_count", "ip_fragment_count"),
        "urgent": ("urgent_count",),
        "num_failed_logins": ("failed_logins",),
        "count": ("flow_packet_count", "packets_in_flow", "packet_count"),
        "srv_count": ("service_packet_count", "packets_to_service"),
        "dst_host_count": ("destination_host_count", "flows_to_destination"),
        "dst_host_srv_count": ("destination_service_count", "flows_to_service"),
    }
    for feature, names in aliases.items():
        if feature in provided:
            continue
        value, source = _value(record, *names)
        if source:
            row[feature] = value
            derived.append(feature)
        elif feature in {"src_bytes", "count"}:
            # A packet size is observable, but it must not be mistaken for a
            # bidirectional flow total.  It is a lower-bound estimate only.
            packet_size, packet_key = _value(record, "packet_size", "length", "wire_length")
            if packet_key:
                row[feature] = packet_size if feature == "src_bytes" else 1
                derived.append(feature)
            else:
                defaulted.append(feature)
        else:
            defaulted.append(feature)

    source_ip, source_ip_key = _value(record, "source_ip", "src_ip")
    destination_ip_value, destination_ip_key = _value(record, "destination_ip", "dst_ip")
    source_port, source_port_key = _value(record, "source_port", "src_port", "sport")
    if "land" not in provided and source_ip_key and destination_ip_key:
        row["land"] = int(
            str(source_ip) == str(destination_ip_value)
            and source_port_key is not None
            and port_key is not None
            and _integer(source_port) == _integer(destination_port)
        )
        derived.append("land")

    # A boolean anomaly indicator can safely enrich the explicit fragment field.
    if "wrong_fragment" not in provided:
        fragmented, fragmented_key = _value(record, "is_fragment", "fragmented")
        if fragmented_key and bool(fragmented):
            row["wrong_fragment"] = max(_integer(row["wrong_fragment"]), 1)
            if "wrong_fragment" not in derived:
                derived.append("wrong_fragment")

    # Explicit, measured rates supplied by a flow tracker are retained.  If the
    # tracker did not compute them, they are intentionally neutral rather than
    # inferred from one packet.
    for feature in _RATE_FEATURES:
        if feature not in provided and feature not in record:
            defaulted.append(feature)

    return coerce_nsl_kdd_record(row), FeatureMappingDiagnostics(
        provided_fields=sorted(set(provided)),
        derived_fields=sorted(set(derived)),
        defaulted_context_features=sorted(set(defaulted)),
    )
