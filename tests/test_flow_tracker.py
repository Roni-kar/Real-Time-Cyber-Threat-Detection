import unittest

from network_sniffer.features import FlowTracker


class FlowTrackerTests(unittest.TestCase):
    def test_reverse_direction_reuses_bidirectional_flow(self) -> None:
        tracker = FlowTracker(idle_seconds=60, max_flows=2)
        outbound = tracker.make_key("TCP", "192.0.2.1", 50000, "198.51.100.1", 443)
        inbound = tracker.make_key("TCP", "198.51.100.1", 443, "192.0.2.1", 50000)
        self.assertEqual(outbound, inbound)
        first = tracker.update(outbound, packet_size=100, observed_at=10.0)
        second = tracker.update(inbound, packet_size=50, observed_at=10.5)
        self.assertEqual(first.packet_count, 1)
        self.assertEqual(second.packet_count, 2)
        self.assertEqual(second.byte_count, 150)
        self.assertGreater(second.packets_per_second, 0)

    def test_flow_count_remains_bounded(self) -> None:
        tracker = FlowTracker(idle_seconds=60, max_flows=1)
        first = tracker.make_key("TCP", "192.0.2.1", 1, "198.51.100.1", 2)
        second = tracker.make_key("UDP", "192.0.2.2", 3, "198.51.100.2", 4)
        tracker.update(first, 10, 1.0)
        tracker.update(second, 10, 2.0)
        self.assertEqual(tracker.tracked_flow_count, 1)


if __name__ == "__main__":
    unittest.main()
