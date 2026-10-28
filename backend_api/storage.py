"""Small SQLite repository for reproducible traffic and alert history."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .schemas import AlertResponse, PacketAnalysisRequest, PacketLogResponse, SeverityLevel


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


class TrafficRepository:
    """Persist analysis events in a local SQLite file.

    Connections are intentionally short-lived. This avoids sharing a SQLite
    connection across FastAPI worker threads and keeps deployment simple.
    """

    def __init__(self, database_path: Path, max_records: int = 500) -> None:
        self.database_path = database_path
        self.max_records = max_records

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.database_path), timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS packets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    source_ip TEXT NOT NULL,
                    destination_ip TEXT NOT NULL,
                    protocol TEXT NOT NULL,
                    source_port INTEGER,
                    destination_port INTEGER,
                    packet_size INTEGER NOT NULL,
                    ttl INTEGER,
                    tcp_flags TEXT,
                    flow_packet_count INTEGER NOT NULL,
                    flow_byte_count INTEGER NOT NULL,
                    flow_duration_seconds REAL NOT NULL,
                    packets_per_second REAL NOT NULL,
                    bytes_per_second REAL NOT NULL,
                    is_private_source INTEGER,
                    is_private_destination INTEGER,
                    anomaly_indicators TEXT NOT NULL,
                    capture_interface TEXT,
                    prediction_label TEXT NOT NULL,
                    is_threat INTEGER NOT NULL,
                    confidence REAL,
                    severity TEXT NOT NULL,
                    prediction_source TEXT NOT NULL,
                    model_version TEXT,
                    explanation TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    packet_id INTEGER NOT NULL UNIQUE,
                    timestamp TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    source_ip TEXT NOT NULL,
                    destination_ip TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',
                    FOREIGN KEY(packet_id) REFERENCES packets(id)
                );
                CREATE INDEX IF NOT EXISTS idx_packets_timestamp ON packets(timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_packets_threat ON packets(is_threat, timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_packets_source ON packets(source_ip, timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status, timestamp DESC);
                """
            )

    def save_analysis(
        self,
        packet: PacketAnalysisRequest,
        outcome: Dict[str, Any],
        severity: SeverityLevel,
    ) -> Tuple[int, Optional[AlertResponse]]:
        """Save an analysis event and create an alert only for threats."""
        values = (
            packet.timestamp.isoformat(),
            str(packet.source_ip),
            str(packet.destination_ip),
            packet.protocol,
            packet.source_port,
            packet.destination_port,
            packet.packet_size,
            packet.ttl,
            packet.tcp_flags,
            packet.flow_packet_count,
            packet.flow_byte_count,
            packet.flow_duration_seconds,
            packet.packets_per_second,
            packet.bytes_per_second,
            None if packet.is_private_source is None else int(packet.is_private_source),
            None if packet.is_private_destination is None else int(packet.is_private_destination),
            json.dumps(packet.anomaly_indicators),
            packet.capture_interface,
            outcome["label"],
            int(outcome["is_threat"]),
            outcome.get("confidence"),
            severity.value,
            outcome["source"],
            outcome.get("model_version"),
            json.dumps(outcome.get("explanation", [])),
            _utc_now(),
        )
        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO packets (
                    timestamp, source_ip, destination_ip, protocol, source_port,
                    destination_port, packet_size, ttl, tcp_flags, flow_packet_count,
                    flow_byte_count, flow_duration_seconds, packets_per_second,
                    bytes_per_second, is_private_source, is_private_destination,
                    anomaly_indicators, capture_interface, prediction_label, is_threat,
                    confidence, severity, prediction_source, model_version, explanation,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            packet_id = int(cursor.lastrowid)
            alert: Optional[AlertResponse] = None
            if outcome["is_threat"]:
                title = f"{severity.value.title()} threat detected"
                message = self._alert_message(packet, outcome, severity)
                alert_cursor = connection.execute(
                    """
                    INSERT INTO alerts (
                        packet_id, timestamp, severity, title, message, source_ip, destination_ip
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        packet_id,
                        packet.timestamp.isoformat(),
                        severity.value,
                        title,
                        message,
                        str(packet.source_ip),
                        str(packet.destination_ip),
                    ),
                )
                alert = AlertResponse(
                    id=int(alert_cursor.lastrowid),
                    packet_id=packet_id,
                    timestamp=packet.timestamp,
                    severity=severity,
                    title=title,
                    message=message,
                    source_ip=str(packet.source_ip),
                    destination_ip=str(packet.destination_ip),
                )
            self._trim_history(connection)
        return packet_id, alert

    @staticmethod
    def _alert_message(
        packet: PacketAnalysisRequest,
        outcome: Dict[str, Any],
        severity: SeverityLevel,
    ) -> str:
        indicators = ", ".join(packet.anomaly_indicators[:3]) or "classifier evidence"
        return (
            f"{outcome['label']} traffic from {packet.source_ip} to {packet.destination_ip} "
            f"over {packet.protocol}; severity={severity.value}; evidence={indicators}."
        )

    def _trim_history(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            DELETE FROM alerts WHERE packet_id IN (
                SELECT id FROM packets ORDER BY id DESC LIMIT -1 OFFSET ?
            )
            """,
            (self.max_records,),
        )
        connection.execute(
            """
            DELETE FROM packets WHERE id IN (
                SELECT id FROM packets ORDER BY id DESC LIMIT -1 OFFSET ?
            )
            """,
            (self.max_records,),
        )

    def list_logs(self, limit: int, offset: int, threats_only: bool = False) -> Tuple[int, List[PacketLogResponse]]:
        where = "WHERE is_threat = 1" if threats_only else ""
        with self._connection() as connection:
            total = int(connection.execute(f"SELECT COUNT(*) FROM packets {where}").fetchone()[0])
            rows = connection.execute(
                f"""
                SELECT id, timestamp, source_ip, destination_ip, protocol, source_port,
                       destination_port, packet_size, flow_packet_count,
                       flow_duration_seconds, anomaly_indicators, prediction_label,
                       is_threat, confidence, severity, prediction_source, model_version
                FROM packets {where} ORDER BY id DESC LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return total, [self._row_to_log(row) for row in rows]

    def list_alerts(
        self,
        limit: int,
        offset: int,
        severity: Optional[SeverityLevel] = None,
        status: Optional[str] = None,
    ) -> Tuple[int, List[AlertResponse]]:
        filters: List[str] = []
        params: List[Any] = []
        if severity:
            filters.append("severity = ?")
            params.append(severity.value)
        if status:
            filters.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(filters)}" if filters else ""
        with self._connection() as connection:
            total = int(connection.execute(f"SELECT COUNT(*) FROM alerts {where}", params).fetchone()[0])
            rows = connection.execute(
                f"""
                SELECT id, packet_id, timestamp, severity, title, message, source_ip,
                       destination_ip, status FROM alerts {where}
                ORDER BY id DESC LIMIT ? OFFSET ?
                """,
                [*params, limit, offset],
            ).fetchall()
        return total, [self._row_to_alert(row) for row in rows]

    def statistics(self) -> Dict[str, Any]:
        with self._connection() as connection:
            totals = connection.execute(
                "SELECT COUNT(*) AS total, COALESCE(SUM(is_threat), 0) AS threats FROM packets"
            ).fetchone()
            severity_rows = connection.execute(
                "SELECT severity, COUNT(*) AS count FROM packets WHERE is_threat = 1 GROUP BY severity"
            ).fetchall()
            protocol_rows = connection.execute(
                """
                SELECT protocol, COUNT(*) AS packets, COALESCE(SUM(is_threat), 0) AS threats
                FROM packets GROUP BY protocol ORDER BY packets DESC LIMIT 10
                """
            ).fetchall()
            suspicious_rows = connection.execute(
                """
                SELECT source_ip, COUNT(*) AS threat_count,
                    CASE
                        WHEN MAX(CASE severity WHEN 'critical' THEN 4 WHEN 'high' THEN 3
                            WHEN 'medium' THEN 2 WHEN 'low' THEN 1 ELSE 0 END) = 4 THEN 'critical'
                        WHEN MAX(CASE severity WHEN 'critical' THEN 4 WHEN 'high' THEN 3
                            WHEN 'medium' THEN 2 WHEN 'low' THEN 1 ELSE 0 END) = 3 THEN 'high'
                        WHEN MAX(CASE severity WHEN 'critical' THEN 4 WHEN 'high' THEN 3
                            WHEN 'medium' THEN 2 WHEN 'low' THEN 1 ELSE 0 END) = 2 THEN 'medium'
                        ELSE 'low'
                    END AS highest_severity
                FROM packets WHERE is_threat = 1
                GROUP BY source_ip ORDER BY threat_count DESC LIMIT 10
                """
            ).fetchall()
            open_alerts = int(connection.execute("SELECT COUNT(*) FROM alerts WHERE status = 'open'").fetchone()[0])

        total = int(totals["total"])
        threats = int(totals["threats"])
        return {
            "total_packets": total,
            "threat_packets": threats,
            "normal_packets": total - threats,
            "threat_rate": round((threats / total) * 100, 2) if total else 0.0,
            "open_alerts": open_alerts,
            "severity_distribution": {row["severity"]: int(row["count"]) for row in severity_rows},
            "protocol_trends": [
                {"protocol": row["protocol"], "packets": int(row["packets"]), "threats": int(row["threats"])}
                for row in protocol_rows
            ],
            "suspicious_ips": [
                {"ip_address": row["source_ip"], "threat_count": int(row["threat_count"]),
                 "highest_severity": row["highest_severity"]}
                for row in suspicious_rows
            ],
        }

    @staticmethod
    def _row_to_log(row: sqlite3.Row) -> PacketLogResponse:
        return PacketLogResponse(
            id=int(row["id"]), timestamp=_as_datetime(row["timestamp"]),
            source_ip=row["source_ip"], destination_ip=row["destination_ip"], protocol=row["protocol"],
            source_port=row["source_port"], destination_port=row["destination_port"],
            packet_size=int(row["packet_size"]), flow_packet_count=int(row["flow_packet_count"]),
            flow_duration_seconds=float(row["flow_duration_seconds"]),
            anomaly_indicators=json.loads(row["anomaly_indicators"]), prediction_label=row["prediction_label"],
            is_threat=bool(row["is_threat"]), confidence=row["confidence"],
            severity=SeverityLevel(row["severity"]), prediction_source=row["prediction_source"],
            model_version=row["model_version"],
        )

    @staticmethod
    def _row_to_alert(row: sqlite3.Row) -> AlertResponse:
        return AlertResponse(
            id=int(row["id"]), packet_id=int(row["packet_id"]), timestamp=_as_datetime(row["timestamp"]),
            severity=SeverityLevel(row["severity"]), title=row["title"], message=row["message"],
            source_ip=row["source_ip"], destination_ip=row["destination_ip"], status=row["status"],
        )
