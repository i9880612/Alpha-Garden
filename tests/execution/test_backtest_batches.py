from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tests.execution.catalog_fixture import initialize_test_generation_catalog
from execution.backtest_batches import (
    AutomatedCandidateBacktest,
    load_automated_run_backtest_usage,
    prepare_automated_candidate_backtest_batch,
)
from execution.backtests import (
    apply_backtest_detail,
    record_backtest_yearly_stats,
    record_submission_accepted,
    record_submission_unknown,
)
from execution.real_backtests import prepare_real_backtest
from execution.seeds import synchronize_signal_seeds
from execution.runs import AutomatedRunLimits, prepare_automated_run, start_automated_run, fail_automated_run
from generation.candidate import (
    CandidateChange,
    FormulaCandidate,
    exploration_candidate,
    mutation_candidate,
)
from generation.parser import parse_formula
from generation.direction import reverse_direction_candidate
from persistence.backtests import get_backtest_mutation, initialize_backtest_schema
from persistence.database import open_database
from persistence.runs import initialize_run_schema, get_automated_run_backtest_by_task
from persistence.submissions import initialize_submission_schema
from worldquant.backtests import BacktestCheck, BacktestDetail, BacktestSettings


def _checks(statuses: dict[str, str]) -> tuple[BacktestCheck, ...]:
    return tuple(
        BacktestCheck(
            name=name,
            status=status,
            threshold=None,
            actual=None,
            platform_date=None,
        )
        for name, status in sorted(statuses.items())
    )


class AutomatedBacktestBatchPreparationTests(unittest.TestCase):
    def test_optimization_rejects_exploration_without_creating_tasks(self):
        run_id = self._start_automated_run(max_backtests=2, backtest_count=2, optimization_only=True)
        with self.assertRaisesRegex(ValueError, "optimization_run_candidate_not_allowed"):
            self._prepare_formulas(run_id, formulas=("rank(close)",),
                                   created_at="2026-08-30T00:02:00+08:00")
        self.assertEqual(self._task_count(), 0)

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "batch.sqlite3"
        self.settings = BacktestSettings(
            instrument_type="EQUITY",
            region="USA",
            universe="TOP3000",
            delay=1,
            decay=4,
            neutralization="SECTOR",
            truncation=0.08,
            pasteurization="ON",
            unit_handling="VERIFY",
            nan_handling="OFF",
            language="FASTEXPR",
            visualization=False,
            max_trade="OFF",
            max_position="OFF",
        )
        self.settings_path = Path(self.temporary_directory.name) / "backtest.json"
        self.settings_path.write_text(
            json.dumps(self._settings_policy()),
            encoding="utf-8",
        )
        with open_database(self.database_path) as connection:
            initialize_backtest_schema(connection)
            initialize_run_schema(connection)
            initialize_submission_schema(connection)
            initialize_test_generation_catalog(connection)

    def test_creates_only_confirmed_tasks_in_one_transaction(self) -> None:
        run_id = self._start_automated_run(max_backtests=2, backtest_count=2)
        prepared = self._prepare_formulas(
            run_id,
            formulas=("rank(close)", "rank(open)"),
            created_at="2026-08-29T00:00:00+00:00",
        )

        with open_database(self.database_path) as connection:
            tasks = connection.execute(
                "SELECT formula, status FROM backtest_tasks ORDER BY formula"
            ).fetchall()
            result_count = connection.execute(
                "SELECT COUNT(*) FROM backtest_results"
            ).fetchone()[0]
        self.assertEqual(len(prepared), 2)
        self.assertEqual(
            tuple((row["formula"], row["status"]) for row in tasks),
            (("rank(close)", "created"), ("rank(open)", "created")),
        )
        self.assertEqual(result_count, 0)

    def test_duplicate_formula_rolls_back_the_entire_new_batch(self) -> None:
        run_id = self._start_automated_run(max_backtests=2, backtest_count=2)
        with self.assertRaisesRegex(ValueError, "backtest_batch_formula_duplicate"):
            self._prepare_formulas(
                run_id,
                formulas=("rank(close)", "rank((close))"),
                created_at="2026-08-29T00:00:00+00:00",
            )

        self.assertEqual(self._task_count(), 0)

    def test_existing_identity_is_reused_while_new_task_is_created(self) -> None:
        prepare_real_backtest(
            self.database_path,
            account_scope="group-account",
            formula="rank(open)",
            settings=self.settings,
            created_at="2026-08-29T00:00:00+00:00",
        )
        run_id = self._start_automated_run(max_backtests=2, backtest_count=2)

        prepared = self._prepare_formulas(
            run_id,
            formulas=("rank(close)", "rank(open)"),
            created_at="2026-08-29T00:01:00+00:00",
        )

        with open_database(self.database_path) as connection:
            formulas = tuple(
                row["formula"]
                for row in connection.execute(
                    "SELECT formula FROM backtest_tasks ORDER BY formula"
                )
            )
        self.assertEqual(formulas, ("rank(close)", "rank(open)"))
        self.assertEqual(len(prepared), 2)

    def test_automated_batch_uses_frozen_plan_and_derives_usage(self) -> None:
        run_id = self._start_automated_run(max_backtests=4, backtest_count=2)
        changed_policy = self._settings_policy()
        changed_policy["decay"] = 5
        self.settings_path.write_text(
            json.dumps(changed_policy),
            encoding="utf-8",
        )

        prepared = self._prepare_formulas(
            run_id,
            formulas=("rank(close)", "rank(open)"),
            created_at="2026-08-30T00:02:00+08:00",
        )
        usage = load_automated_run_backtest_usage(self.database_path, run_id)

        self.assertEqual(len(prepared), 2)
        self.assertTrue(
            all(
                snapshot.task.settings_json
                == json.dumps(
                    self.settings.as_platform_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for snapshot in prepared
            )
        )
        self.assertEqual(usage.prepared_backtests, 2)
        self.assertEqual(usage.attempted_backtests, 0)
        self.assertEqual(usage.remaining_backtests, 2)
        self.assertEqual(usage.cycle_number, 1)
        self.assertEqual(usage.cycle_remaining_backtests, 0)

    def test_unlimited_run_reports_no_remaining_total_limit(self) -> None:
        run_id = self._start_automated_run(
            max_cycles=-1,
            max_backtests=0,
            backtest_count=2,
        )

        self._prepare_formulas(
            run_id,
            formulas=("rank(close)", "rank(open)"),
            created_at="2026-08-30T00:02:00+08:00",
        )
        usage = load_automated_run_backtest_usage(self.database_path, run_id)

        self.assertEqual(usage.prepared_backtests, 2)
        self.assertIsNone(usage.remaining_backtests)
        self.assertEqual(usage.cycle_remaining_backtests, 0)

    def test_automated_batch_accepts_settings_within_the_frozen_policy(self) -> None:
        run_id = self._start_automated_run(max_backtests=1, backtest_count=1)
        settings = replace(self.settings, neutralization="SUBINDUSTRY")

        prepared = self._prepare_formulas(
            run_id,
            formulas=("rank(close)",),
            settings=settings,
            created_at="2026-08-30T00:02:00+08:00",
        )

        settings = json.loads(prepared[0].task.settings_json)
        self.assertEqual(settings["neutralization"], "SUBINDUSTRY")

    def test_automated_batch_rejects_settings_outside_the_frozen_policy(self) -> None:
        run_id = self._start_automated_run(max_backtests=1, backtest_count=1)
        settings = replace(self.settings, decay=5)

        with self.assertRaisesRegex(
            ValueError,
            "automated_run_settings_not_allowed",
        ):
            self._prepare_formulas(
                run_id,
                formulas=("rank(close)",),
                settings=settings,
                created_at="2026-08-30T00:02:00+08:00",
            )

        self.assertEqual(self._task_count(), 0)

    def test_unknown_submission_counts_as_attempted(self) -> None:
        run_id = self._start_automated_run(max_backtests=2, backtest_count=2)
        prepared = self._prepare_formulas(
            run_id,
            formulas=("rank(close)",),
            created_at="2026-08-30T00:02:00+08:00",
        )
        with open_database(self.database_path) as connection:
            record_submission_unknown(
                connection,
                prepared[0].task.task_id,
                observed_at="2026-08-30T00:03:00+08:00",
            )

        usage = load_automated_run_backtest_usage(self.database_path, run_id)

        self.assertEqual(usage.prepared_backtests, 1)
        self.assertEqual(usage.attempted_backtests, 1)

    def test_automated_batch_limit_failure_rolls_back_new_facts(self) -> None:
        run_id = self._start_automated_run(max_backtests=2, backtest_count=2)
        self._prepare_formulas(
            run_id,
            formulas=("rank(close)", "rank(open)"),
            created_at="2026-08-30T00:02:00+08:00",
        )

        with self.assertRaisesRegex(
            ValueError,
            "automated_run_backtest_limit_reached",
        ):
            self._prepare_formulas(
                run_id,
                formulas=("rank(high)",),
                created_at="2026-08-30T00:03:00+08:00",
            )

        usage = load_automated_run_backtest_usage(self.database_path, run_id)
        self.assertEqual(self._task_count(), 2)
        self.assertEqual(usage.prepared_backtests, 2)

    def test_automated_batch_enforces_the_current_cycle_limit(self) -> None:
        run_id = self._start_automated_run(max_backtests=4, backtest_count=2)
        self._prepare_formulas(
            run_id,
            formulas=("rank(close)", "rank(open)"),
            created_at="2026-08-30T00:02:00+08:00",
        )

        with self.assertRaisesRegex(
            ValueError,
            "automated_run_cycle_backtest_limit_reached",
        ):
            self._prepare_formulas(
                run_id,
                formulas=("rank(high)",),
                created_at="2026-08-30T00:03:00+08:00",
            )

        usage = load_automated_run_backtest_usage(self.database_path, run_id)
        self.assertEqual(self._task_count(), 2)
        self.assertEqual(usage.prepared_backtests, 2)

    def test_same_automated_batch_retry_is_idempotent(self) -> None:
        run_id = self._start_automated_run(max_backtests=2, backtest_count=2)
        first = self._prepare_formulas(
            run_id,
            formulas=("rank(close)", "rank(open)"),
            created_at="2026-08-30T00:02:00+08:00",
        )
        retried = self._prepare_formulas(
            run_id,
            formulas=("rank(close)", "rank(open)"),
            created_at="2026-08-30T00:02:00+08:00",
        )

        self.assertEqual(retried, first)
        self.assertEqual(self._task_count(), 2)

    def test_automated_batch_requires_preapproved_real_backtests(self) -> None:
        run_id = self._start_automated_run(
            max_backtests=0,
            backtest_count=2,
            authorized=False,
        )

        with self.assertRaisesRegex(
            ValueError,
            "automated_run_backtests_not_authorized",
        ):
            self._prepare_formulas(
                run_id,
                formulas=("rank(close)",),
                created_at="2026-08-30T00:02:00+08:00",
            )

        self.assertEqual(self._task_count(), 0)

    def test_automated_mutation_batch_saves_only_submitted_lineage_facts(self) -> None:
        parent = self._signal_seed("group-account")
        run_id = self._start_automated_run(max_backtests=1, backtest_count=1)
        candidate = mutation_candidate(
            parse_formula("rank(open)").expression,
            parent_task_id=parent.task.task_id,
            parent_formula_fingerprint=parent.task.formula_fingerprint,
            change=CandidateChange(
                action="field_swap",
                location="formula.arguments[0]",
                before="close",
                after="open",
            ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "backtest_mutation_settings_mismatch",
        ):
            self._prepare_candidates(
                run_id,
                candidates=(candidate,),
                settings=replace(
                    self.settings,
                    neutralization="SUBINDUSTRY",
                ),
                created_at="2026-08-30T00:03:00+08:00",
            )
        self.assertEqual(self._task_count(), 1)

        prepared = self._prepare_candidates(
            run_id,
            candidates=(candidate,),
            created_at="2026-08-30T00:03:00+08:00",
        )

        with open_database(self.database_path) as connection:
            mutation = get_backtest_mutation(
                connection,
                prepared[0].task.task_id,
            )
        self.assertIsNotNone(mutation)
        self.assertEqual(mutation.parent_task_id, parent.task.task_id)
        self.assertEqual(mutation.action, "field_swap")
        self.assertEqual(self._task_count(), 2)

    def test_automated_mutation_batch_rejects_task_outside_parent_pool(self) -> None:
        source = prepare_real_backtest(
            self.database_path,
            account_scope="group-account",
            formula="rank(close)",
            settings=self.settings,
            created_at="2026-08-30T00:00:00+08:00",
        )
        run_id = self._start_automated_run(max_backtests=1, backtest_count=1)
        candidate = mutation_candidate(
            parse_formula("rank(open)").expression,
            parent_task_id=source.task.task_id,
            parent_formula_fingerprint=source.task.formula_fingerprint,
            change=CandidateChange(
                action="field_swap",
                location="formula.arguments[0]",
                before="close",
                after="open",
            ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "backtest_mutation_parent_not_eligible",
        ):
            self._prepare_candidates(
                run_id,
                candidates=(candidate,),
                created_at="2026-08-30T00:03:00+08:00",
            )

        self.assertEqual(self._task_count(), 1)

    def test_automated_mutation_batch_rejects_a_parent_from_another_account(
        self,
    ) -> None:
        parent = self._signal_seed("other-account")
        run_id = self._start_automated_run(max_backtests=1, backtest_count=1)
        candidate = mutation_candidate(
            parse_formula("rank(open)").expression,
            parent_task_id=parent.task.task_id,
            parent_formula_fingerprint=parent.task.formula_fingerprint,
            change=CandidateChange(
                action="field_swap",
                location="formula.arguments[0]",
                before="close",
                after="open",
            ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "backtest_mutation_parent_account_mismatch",
        ):
            self._prepare_candidates(
                run_id,
                candidates=(candidate,),
                created_at="2026-08-30T00:03:00+08:00",
            )

        self.assertEqual(self._task_count(), 1)

    def test_mixed_automated_batch_rolls_back_when_mutation_parent_is_invalid(self) -> None:
        run_id = self._start_automated_run(max_backtests=2, backtest_count=2)
        exploration = exploration_candidate(
            parse_formula("rank(close)").expression
        )
        mutation = mutation_candidate(
            parse_formula("rank(open)").expression,
            parent_task_id="missing-parent",
            parent_formula_fingerprint="missing-fingerprint",
            change=CandidateChange(
                action="field_swap",
                location="formula.arguments[0]",
                before="close",
                after="open",
            ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "backtest_mutation_parent_not_eligible",
        ):
            self._prepare_candidates(
                run_id,
                candidates=(exploration, mutation),
                created_at="2026-08-30T00:02:00+08:00",
            )

        usage = load_automated_run_backtest_usage(self.database_path, run_id)
        self.assertEqual(self._task_count(), 0)
        self.assertEqual(usage.prepared_backtests, 0)

    def test_direction_batch_keeps_lineage_and_blocks_repeat_unknown_request(self):
        source = self._signal_seed("group-account", sharpe=-1.4, fitness=-1.1)
        candidate = reverse_direction_candidate(
            parse_formula(source.task.formula).expression,
            parent_task_id=source.task.task_id,
        )
        run = self._start_automated_run(max_backtests=2, backtest_count=2)
        created = self._prepare_candidates(
            run, candidates=(candidate,), created_at="2026-08-30T00:03:00+08:00"
        )
        with open_database(self.database_path) as connection:
            mutation = get_backtest_mutation(connection, created[0].task.task_id)
            self.assertEqual(mutation.parent_task_id, source.task.task_id)
            record_submission_unknown(
                connection,
                created[0].task.task_id,
                observed_at="2026-08-30T00:04:00+08:00",
            )
        with self.assertRaisesRegex(ValueError, "direction_request_already_exists"):
            self._prepare_candidates(
                run, candidates=(candidate,), created_at="2026-08-30T00:05:00+08:00"
            )
    def test_direction_label_cannot_disguise_another_formula_or_settings(self):
        source = self._signal_seed("group-account", sharpe=-1.4, fitness=-1.1)
        expected = reverse_direction_candidate(
            parse_formula(source.task.formula).expression,
            parent_task_id=source.task.task_id,
        )
        forged = replace(expected, expression=parse_formula("rank(open)").expression)
        run = self._start_automated_run(max_backtests=2, backtest_count=2)
        with self.assertRaisesRegex(ValueError, "direction_source_invalid"):
            self._prepare_candidates(
                run, candidates=(forged,), created_at="2026-08-30T00:03:00+08:00"
            )
        self.assertEqual(self._task_count(), 1)
        with self.assertRaisesRegex(ValueError, "settings_mismatch"):
            self._prepare_candidates(
                run, candidates=(expected,), created_at="2026-08-30T00:03:00+08:00",
                settings=replace(self.settings, neutralization="SUBINDUSTRY"),
            )
        self.assertEqual(self._task_count(), 1)

    def test_cancelled_direction_can_join_a_new_batch_without_changing_old_lineage(self):
        source = self._signal_seed("group-account", sharpe=-1.4, fitness=-1.1)
        candidate = reverse_direction_candidate(parse_formula(source.task.formula).expression,
                                                parent_task_id=source.task.task_id)
        old_run = self._start_automated_run(max_backtests=1, backtest_count=1)
        old = self._prepare_candidates(old_run, candidates=(candidate,),
            created_at="2026-08-30T00:03:00+08:00")[0]
        fail_automated_run(self.database_path, old_run, failed_at="2026-08-30T00:04:00+08:00",
                          reason="replaced_by_new_run")
        new_run = self._start_automated_run(max_backtests=2, backtest_count=2)
        retry = self._prepare_candidates(new_run, candidates=(candidate,),
            created_at="2026-08-30T00:05:00+08:00")[0]
        self.assertNotEqual(old.task.task_id, retry.task.task_id)
        with open_database(self.database_path) as connection:
            self.assertEqual(get_automated_run_backtest_by_task(connection, old.task.task_id).run_id, old_run)
            self.assertEqual(get_automated_run_backtest_by_task(connection, retry.task.task_id).run_id, new_run)
            for task_id in (old.task.task_id, retry.task.task_id):
                self.assertEqual(get_backtest_mutation(connection, task_id).parent_task_id, source.task.task_id)
            record_submission_unknown(connection, retry.task.task_id, observed_at="2026-08-30T00:06:00+08:00")
        with self.assertRaisesRegex(ValueError, "direction_request_already_exists"):
            self._prepare_candidates(new_run, candidates=(candidate,), created_at="2026-08-30T00:07:00+08:00")

    def test_direction_rejects_positive_parent_even_when_active(self):
        source = self._signal_seed("group-account")
        candidate = reverse_direction_candidate(
            parse_formula(source.task.formula).expression,
            parent_task_id=source.task.task_id,
        )
        run = self._start_automated_run(max_backtests=1, backtest_count=1)
        with self.assertRaisesRegex(ValueError, "direction_source_invalid"):
            self._prepare_candidates(
                run, candidates=(candidate,), created_at="2026-08-30T00:03:00+08:00"
            )
        self.assertEqual(self._task_count(), 1)

    def _prepare_formulas(
        self,
        run_id: str,
        *,
        formulas: tuple[str, ...],
        created_at: str,
        settings: BacktestSettings | None = None,
    ):
        return self._prepare_candidates(
            run_id,
            candidates=tuple(self._candidate(formula) for formula in formulas),
            created_at=created_at,
            settings=settings,
        )

    def _prepare_candidates(
        self,
        run_id: str,
        *,
        candidates: tuple[FormulaCandidate, ...],
        created_at: str,
        settings: BacktestSettings | None = None,
    ):
        selected_settings = settings or self.settings
        return prepare_automated_candidate_backtest_batch(
            self.database_path,
            run_id=run_id,
            candidates=tuple(
                AutomatedCandidateBacktest(candidate, selected_settings)
                for candidate in candidates
            ),
            created_at=created_at,
        )

    @staticmethod
    def _candidate(formula: str) -> FormulaCandidate:
        return exploration_candidate(parse_formula(formula).expression)

    def _signal_seed(self, account_scope: str, *, sharpe=1.3, fitness=0.9):
        parent = prepare_real_backtest(
            self.database_path,
            account_scope=account_scope,
            formula="rank(close)",
            settings=self.settings,
            created_at="2026-08-30T00:00:00+08:00",
        )
        with open_database(self.database_path) as connection:
            record_submission_accepted(
                connection,
                parent.task.task_id,
                remote_id="simulation-parent",
                observed_at="2026-08-30T00:01:00+08:00",
            )
            apply_backtest_detail(
                connection,
                parent.task.task_id,
                BacktestDetail(
                    platform_alpha_id="alpha-parent",
                    sharpe=sharpe,
                    fitness=fitness,
                    turnover=0.08,
                    returns=0.01,
                    drawdown=0.1,
                    margin=0.0002,
                    book_size=None,
                    pnl=None,
                    checks=_checks(
                        {
                            "LOW_SHARPE": "FAIL" if sharpe < 0 else "PASS",
                            "LOW_FITNESS": "FAIL",
                            "LOW_SUB_UNIVERSE_SHARPE": "PASS",
                            "CONCENTRATED_WEIGHT": "PASS",
                            "LOW_TURNOVER": "PASS",
                            "HIGH_TURNOVER": "PASS",
                            "MATCHES_COMPETITION": "PASS",
                            "SELF_CORRELATION": "PENDING",
                        }
                    ),
                ),
                observed_at="2026-08-30T00:02:00+08:00",
            )
            record_backtest_yearly_stats(
                connection,
                parent.task.task_id,
                (),
                observed_at="2026-08-30T00:02:00+08:00",
            )
            synchronize_signal_seeds(
                connection,
                candidate_task_ids=(parent.task.task_id,),
            )
        return parent

    @staticmethod
    def _settings_policy() -> dict[str, object]:
        return {
            "catalogContext": {
                "instrumentType": "EQUITY",
                "region": "USA",
                "universe": "TOP3000",
                "delay": 1,
            },
            "decay": 4,
            "neutralization": {
                "default": "SECTOR",
                "rootGroupNeutralize": "NONE",
                "byFieldCategory": {
                    "sample": "SECTOR",
                    "alternate": "SUBINDUSTRY",
                },
            },
            "truncation": {"default": 0.08, "tailRisk": 0.05},
            "pasteurization": "ON",
            "unitHandling": "VERIFY",
            "nanHandling": "OFF",
            "language": "FASTEXPR",
            "visualization": False,
            "maxTrade": "OFF",
            "maxPosition": "OFF",
        }

    def _task_count(self) -> int:
        with open_database(self.database_path) as connection:
            return connection.execute(
                "SELECT COUNT(*) FROM backtest_tasks"
            ).fetchone()[0]

    def _start_automated_run(
        self,
        *,
        max_cycles: int = 2,
        max_backtests: int,
        backtest_count: int,
        authorized: bool = True,
        optimization_only: bool = False,
    ) -> str:
        prepared = prepare_automated_run(
            self.database_path,
            self.settings_path,
            account_scope="group-account",
            limits=AutomatedRunLimits(
                exploration_percent=30,
                self_correlation_percent=30,
                mutation_percent=40,
                direction_validation_percent=3,
                generation_count=10,
                backtest_count=backtest_count,
                max_cycles=max_cycles,
                max_backtests=max_backtests,
                max_pending_seconds=3600,
                max_consecutive_failures=2,
                max_request_failures=3,
                max_in_flight_backtests=3,
                exploration_seed_attempt_multiplier=4,
                real_backtests_authorized=authorized,
                optimization_only=optimization_only,
            ),
            created_at="2026-08-30T00:00:00+08:00",
        )
        start_automated_run(
            self.database_path,
            prepared.run_id,
            started_at="2026-08-30T00:01:00+08:00",
        )
        return prepared.run_id


if __name__ == "__main__":
    unittest.main()
