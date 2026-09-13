from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from persistence.database import open_database
from persistence.submissions import (
    PlatformSubmittedAlphaRecord,
    initialize_submission_schema,
    list_platform_submitted_alphas,
    record_platform_submitted_alphas,
)


class PlatformSubmittedAlphaPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database_path = Path(self.directory.name) / "database.sqlite3"
        with open_database(self.database_path) as connection:
            initialize_submission_schema(connection)

    def test_preserves_complete_platform_payload_and_is_idempotent(self) -> None:
        original = self._record(
            raw_extra={
                "name": "submitted alpha",
                "tags": ["manual", "verified"],
                "test": {"sharpe": 1.76, "checks": [{"name": "LOW_SHARPE"}]},
                "futureField": {"nested": [1, None, True]},
            }
        )
        refreshed = PlatformSubmittedAlphaRecord(
            account_scope=original.account_scope,
            platform_alpha_id=original.platform_alpha_id,
            formula=original.formula,
            status=original.status,
            date_submitted=original.date_submitted,
            hidden=original.hidden,
            raw_payload=original.raw_payload,
            observed_at="2026-09-03T02:00:00+00:00",
        )

        with open_database(self.database_path) as connection:
            record_platform_submitted_alphas(connection, (original,))
            record_platform_submitted_alphas(connection, (refreshed,))

        with open_database(self.database_path) as connection:
            stored = list_platform_submitted_alphas(
                connection,
                account_scope="account-main",
            )

        self.assertEqual(stored, (refreshed,))
        self.assertEqual(stored[0].raw_payload, original.raw_payload)
        self.assertEqual(
            stored[0].normalized_formula,
            "rank((high+low)/2/close)*rank(volume/adv20)",
        )

    def test_rejects_payload_mismatch_without_writing_any_record(self) -> None:
        valid = self._record()
        mismatched = PlatformSubmittedAlphaRecord(
            account_scope=valid.account_scope,
            platform_alpha_id=valid.platform_alpha_id,
            formula="rank(close)",
            status=valid.status,
            date_submitted=valid.date_submitted,
            hidden=valid.hidden,
            raw_payload=valid.raw_payload,
            observed_at=valid.observed_at,
        )

        with self.assertRaisesRegex(
            ValueError,
            "platform_submitted_alpha_raw_payload_mismatch",
        ):
            with open_database(self.database_path) as connection:
                record_platform_submitted_alphas(connection, (valid, mismatched))

        with open_database(self.database_path) as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM platform_submitted_alphas"
            ).fetchone()[0]
        self.assertEqual(count, 0)

    def test_rejects_reusing_platform_id_for_a_different_formula(self) -> None:
        first = self._record()
        changed_payload = dict(first.raw_payload)
        changed_payload["regular"] = {"code": "rank(returns)"}
        changed = PlatformSubmittedAlphaRecord(
            account_scope=first.account_scope,
            platform_alpha_id=first.platform_alpha_id,
            formula="rank(returns)",
            status=first.status,
            date_submitted=first.date_submitted,
            hidden=first.hidden,
            raw_payload=changed_payload,
            observed_at="2026-09-03T02:00:00+00:00",
        )

        with open_database(self.database_path) as connection:
            record_platform_submitted_alphas(connection, (first,))
        with self.assertRaisesRegex(
            ValueError,
            "platform_submitted_alpha_identity_conflict",
        ):
            with open_database(self.database_path) as connection:
                record_platform_submitted_alphas(connection, (changed,))

        with open_database(self.database_path) as connection:
            stored = list_platform_submitted_alphas(
                connection,
                account_scope=first.account_scope,
            )
        self.assertEqual(stored, (first,))

    def test_concurrent_writers_cannot_overwrite_platform_identity(self) -> None:
        first = self._record()
        changed_payload = dict(first.raw_payload)
        changed_payload["regular"] = {"code": "rank(returns)"}
        changed = PlatformSubmittedAlphaRecord(
            account_scope=first.account_scope,
            platform_alpha_id=first.platform_alpha_id,
            formula="rank(returns)",
            status=first.status,
            date_submitted=first.date_submitted,
            hidden=first.hidden,
            raw_payload=changed_payload,
            observed_at=first.observed_at,
        )
        barrier = threading.Barrier(2)

        def write(record: PlatformSubmittedAlphaRecord) -> str:
            barrier.wait()
            try:
                with open_database(self.database_path) as connection:
                    record_platform_submitted_alphas(connection, (record,))
            except ValueError as exc:
                if str(exc) == "platform_submitted_alpha_identity_conflict":
                    return "identity_conflict"
                raise
            return "saved"

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = tuple(executor.map(write, (first, changed)))

        self.assertEqual(sorted(outcomes), ["identity_conflict", "saved"])
        with open_database(self.database_path) as connection:
            stored = list_platform_submitted_alphas(
                connection,
                account_scope=first.account_scope,
            )
        self.assertEqual(len(stored), 1)
        self.assertIn(stored[0].formula, {first.formula, changed.formula})

    @staticmethod
    def _record(
        *,
        raw_extra: dict[str, object] | None = None,
    ) -> PlatformSubmittedAlphaRecord:
        formula = "rank((high + low) / 2 / close) * rank(volume / adv20)"
        raw_payload: dict[str, object] = {
            "id": "alpha-1",
            "status": "ACTIVE",
            "dateSubmitted": "2026-09-01T02:35:02-04:00",
            "hidden": False,
            "regular": {"code": formula},
        }
        raw_payload.update(raw_extra or {})
        return PlatformSubmittedAlphaRecord(
            account_scope="account-main",
            platform_alpha_id="alpha-1",
            formula=formula,
            status="ACTIVE",
            date_submitted="2026-09-01T02:35:02-04:00",
            hidden=False,
            raw_payload=raw_payload,
            observed_at="2026-09-03T01:00:00+00:00",
        )


if __name__ == "__main__":
    unittest.main()
