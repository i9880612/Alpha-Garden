import sys
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from worldquant.alphas import parse_user_alpha_page
from worldquant.backtests import WorldQuantProtocolError


class UserAlphaPageTests(unittest.TestCase):
    def test_parses_regular_alpha_identity_and_preserves_settings(self) -> None:
        page = parse_user_alpha_page(
            {
                "count": 1,
                "next": None,
                "previous": None,
                "results": [self._alpha()],
            }
        )

        self.assertEqual(page.total_count, 1)
        self.assertFalse(page.has_next)
        record = page.records[0]
        self.assertEqual(record.platform_alpha_id, "alpha-1")
        self.assertEqual(record.formula, "rank(close)")
        assert record.settings is not None
        self.assertEqual(record.settings["maxTrade"], "OFF")
        self.assertEqual(record.settings["startDate"], "2020-01-01")
        self.assertIsNotNone(record.created_at.utcoffset())

    def test_rejects_missing_or_naive_creation_time(self) -> None:
        for value in (None, "2026-09-04T08:40:42"):
            with self.subTest(value=value):
                alpha = self._alpha()
                alpha["dateCreated"] = value
                with self.assertRaisesRegex(
                    WorldQuantProtocolError,
                    "worldquant_user_alpha_created_at_invalid",
                ):
                    parse_user_alpha_page(
                        {
                            "count": 1,
                            "next": None,
                            "previous": None,
                            "results": [alpha],
                        }
                    )

    def test_rejects_duplicate_alpha_identity(self) -> None:
        with self.assertRaisesRegex(
            WorldQuantProtocolError,
            "worldquant_user_alpha_duplicate",
        ):
            parse_user_alpha_page(
                {
                    "count": 2,
                    "next": None,
                    "previous": None,
                    "results": [self._alpha(), self._alpha()],
                }
            )

    @staticmethod
    def _alpha() -> dict[str, object]:
        return {
            "id": "alpha-1",
            "type": "REGULAR",
            "status": "UNSUBMITTED",
            "hidden": False,
            "dateCreated": "2026-09-04T04:40:42-04:00",
            "regular": {"code": "rank(close)"},
            "settings": {
                "instrumentType": "EQUITY",
                "region": "USA",
                "universe": "TOP3000",
                "delay": 1,
                "decay": 4,
                "neutralization": "SECTOR",
                "truncation": 0.08,
                "pasteurization": "ON",
                "unitHandling": "VERIFY",
                "nanHandling": "OFF",
                "language": "FASTEXPR",
                "visualization": False,
                "maxTrade": "OFF",
                "maxPosition": "OFF",
                "startDate": "2020-01-01",
                "endDate": "2025-01-01",
            },
        }


if __name__ == "__main__":
    unittest.main()
