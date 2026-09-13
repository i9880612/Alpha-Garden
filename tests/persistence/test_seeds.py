from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from persistence.backtests import (
    BacktestCheckRecord,
    BacktestResultRecord,
    BacktestTaskRecord,
    create_backtest_task,
    initialize_backtest_schema,
    replace_backtest_task,
    save_backtest_result,
    save_backtest_yearly_stats,
)
from persistence.database import open_database
from persistence.seeds import (
    SignalSeedRecord,
    create_signal_seed,
    get_signal_seed,
    list_signal_seeds,
)


class SignalSeedPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "seeds.sqlite3"
        with open_database(self.database_path) as connection:
            initialize_backtest_schema(connection)

    def test_completed_task_is_recorded_once_as_seed(self) -> None:
        task = self._task()
        record = SignalSeedRecord(
            root_task_id=task.task_id,
            promoted_at="2026-08-31T00:03:00+00:00",
        )
        with open_database(self.database_path) as connection:
            self._save_completed(connection, task)
            first = create_signal_seed(connection, record)
            replayed = create_signal_seed(connection, record)
            loaded = get_signal_seed(connection, task.task_id)

        self.assertEqual(first, record)
        self.assertEqual(replayed, record)
        self.assertEqual(loaded, record)

    def test_non_completed_task_cannot_become_seed(self) -> None:
        task = self._task()
        with open_database(self.database_path) as connection:
            create_backtest_task(connection, task)
            with self.assertRaisesRegex(
                ValueError,
                "signal_seed_task_not_completed",
            ):
                create_signal_seed(
                    connection,
                    SignalSeedRecord(
                        root_task_id=task.task_id,
                        promoted_at="2026-08-31T00:03:00+00:00",
                    ),
                )

    def test_listing_uses_stable_promotion_order(self) -> None:
        first_task = self._task("rank(close)")
        second_task = self._task("rank(open)")
        with open_database(self.database_path) as connection:
            self._save_completed(connection, first_task)
            self._save_completed(connection, second_task)
            second = create_signal_seed(
                connection,
                SignalSeedRecord(
                    root_task_id=second_task.task_id,
                    promoted_at="2026-08-31T00:04:00+00:00",
                ),
            )
            first = create_signal_seed(
                connection,
                SignalSeedRecord(
                    root_task_id=first_task.task_id,
                    promoted_at="2026-08-31T00:03:00+00:00",
                ),
            )

            listed = list_signal_seeds(connection)

        self.assertEqual(listed, (first, second))

    @staticmethod
    def _task(formula: str = "rank(close)") -> BacktestTaskRecord:
        formula_identity = sha256(formula.encode("utf-8")).hexdigest()
        settings_json = '{"delay":1}'
        return BacktestTaskRecord(
            task_id=f"backtest_{formula_identity}",
            account_scope="group-account",
            formula=formula,
            formula_fingerprint=formula_identity,
            settings_json=settings_json,
            request_fingerprint=sha256(
                f"{formula}|{settings_json}".encode("utf-8")
            ).hexdigest(),
            status="created",
            remote_id=None,
            platform_alpha_id=None,
            created_at="2026-08-31T00:00:00+00:00",
            submission_started_at=None,
            last_observed_at=None,
            retry_not_before=None,
            finished_at=None,
            failure_code=None,
            failure_message=None,
        )

    @staticmethod
    def _save_completed(connection, task: BacktestTaskRecord) -> None:
        create_backtest_task(connection, task)
        pending = replace(
            task,
            status="pending",
            remote_id=f"simulation_{task.formula_fingerprint}",
            platform_alpha_id=f"alpha_{task.formula_fingerprint}",
            submission_started_at="2026-08-31T00:01:00+00:00",
            last_observed_at="2026-08-31T00:01:00+00:00",
        )
        replace_backtest_task(connection, pending, expected_status="created")
        save_backtest_result(
            connection,
            BacktestResultRecord(
                task_id=task.task_id,
                sharpe=1.3,
                fitness=1.1,
                turnover=0.12,
                returns=0.08,
                drawdown=0.04,
                margin=0.001,
                book_size=20_000_000,
                pnl=100_000,
                long_count=None,
                short_count=None,
                check_details_captured=True,
                checks=(
                    BacktestCheckRecord(
                        name="LOW_SHARPE",
                        status="PASS",
                        threshold=1.25,
                        actual=1.3,
                        platform_date=None,
                    ),
                ),
            ),
        )
        save_backtest_yearly_stats(connection, task.task_id, ())
        completed = replace(
            pending,
            status="completed",
            platform_alpha_id=f"alpha_{task.formula_fingerprint}",
            last_observed_at="2026-08-31T00:02:00+00:00",
            finished_at="2026-08-31T00:02:00+00:00",
        )
        replace_backtest_task(connection, completed, expected_status="pending")


if __name__ == "__main__":
    unittest.main()
