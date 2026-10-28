import tempfile
import unittest
from pathlib import Path

from backend_api.schemas import PacketAnalysisRequest, SeverityLevel
from backend_api.storage import TrafficRepository


class TrafficRepositoryTests(unittest.TestCase):
    def test_threat_creates_a_persistent_alert(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = TrafficRepository(Path(temporary_directory) / "telemetry.db", max_records=10)
            repository.initialize()
            packet = PacketAnalysisRequest(
                source_ip="192.0.2.1",
                destination_ip="198.51.100.2",
                protocol="TCP",
                packet_size=256,
            )
            packet_id, alert = repository.save_analysis(
                packet,
                {
                    "label": "attack",
                    "is_threat": True,
                    "confidence": 0.91,
                    "source": "ml_pipeline",
                    "model_version": "test-model",
                    "explanation": ["test"],
                },
                SeverityLevel.HIGH,
            )
            self.assertGreater(packet_id, 0)
            self.assertIsNotNone(alert)
            statistics = repository.statistics()
            self.assertEqual(statistics["total_packets"], 1)
            self.assertEqual(statistics["threat_packets"], 1)
            total, alerts = repository.list_alerts(limit=10, offset=0)
            self.assertEqual(total, 1)
            self.assertEqual(alerts[0].severity, SeverityLevel.HIGH)


if __name__ == "__main__":
    unittest.main()
