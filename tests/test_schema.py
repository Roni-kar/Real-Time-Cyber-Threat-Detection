import unittest

from ai_model.schema import (
    NSL_KDD_FEATURES,
    binary_label,
    live_packet_to_nsl_kdd,
    service_from_port,
)


class NslKddSchemaTests(unittest.TestCase):
    def test_binary_labels_normalize_trailing_periods(self) -> None:
        self.assertEqual(binary_label("normal."), "normal")
        self.assertEqual(binary_label("neptune"), "attack")

    def test_port_mapping_is_conservative(self) -> None:
        self.assertEqual(service_from_port(443), "http_443")
        self.assertEqual(service_from_port(54321), "other")

    def test_live_packet_produces_schema_complete_record(self) -> None:
        features, diagnostics = live_packet_to_nsl_kdd(
            {
                "source_ip": "192.0.2.10",
                "destination_ip": "198.51.100.20",
                "source_port": 49152,
                "destination_port": 443,
                "protocol": "TCP",
                "packet_size": 512,
                "tcp_flags": "S",
                "flow_packet_count": 2,
                "flow_duration_seconds": 0.5,
            }
        )
        self.assertEqual(tuple(features), NSL_KDD_FEATURES)
        self.assertEqual(features["protocol_type"], "tcp")
        self.assertEqual(features["service"], "http_443")
        self.assertEqual(features["flag"], "S0")
        self.assertEqual(features["count"], 2)
        self.assertEqual(features["src_bytes"], 512)
        self.assertIn(diagnostics.quality, {"packet_only", "partially_enriched", "flow_enriched"})


if __name__ == "__main__":
    unittest.main()
