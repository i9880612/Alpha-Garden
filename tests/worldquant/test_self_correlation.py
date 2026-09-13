import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from worldquant.self_correlation import maximum_self_correlation_reference


class SelfCorrelationProtocolTests(unittest.TestCase):
    def payload(self, records, maximum=0.9):
        return {
            "is": {
                "checks": [
                    {
                        "name": "SELF_CORRELATION",
                        "result": "FAIL",
                        "value": maximum,
                        "limit": 0.7,
                    }
                ],
                "selfCorrelated": {
                    "max": maximum,
                    "schema": {"properties": [{"name": "correlation"}, {"name": "id"}]},
                    "records": records,
                },
            }
        }

    def test_reported_maximum_and_stable_tie(self):
        self.assertEqual(
            maximum_self_correlation_reference(
                self.payload([[0.8, "other"], [0.9, "z"], [0.9, "a"]])
            ),
            "a",
        )

    def test_contradictory_missing_or_duplicated_check_values_are_not_evidence(self):
        for change in (
            {"value": 0.8},
            {"value": True},
            {"limit": None},
            {"limit": 0.95},
            {"limit": float("nan")},
        ):
            payload = self.payload([[0.9, "a"]])
            payload["is"]["checks"][0].update(change)
            self.assertIsNone(maximum_self_correlation_reference(payload))
        for checks in ([], [{"name": "SELF_CORRELATION"}] * 2):
            payload = self.payload([[0.9, "a"]])
            payload["is"]["checks"] = checks
            self.assertIsNone(maximum_self_correlation_reference(payload))

    def test_unavailable_or_invalid_detail_does_not_invent_peer(self):
        for payload in (
            None,
            {},
            self.payload([]),
            self.payload([[0.8, "a"]]),
            self.payload([[True, "a"]]),
            self.payload([[0.9, ""]]),
            self.payload([[1, "a"]]),
            self.payload([[0.9]]),
            self.payload([[0.9, "a"]], float("nan")),
        ):
            with self.subTest(payload=payload):
                self.assertIsNone(maximum_self_correlation_reference(payload))
