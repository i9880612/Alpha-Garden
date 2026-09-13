from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from execution.generation import (
    BatchShortfall,
    ExclusionCount,
    generate_exploration_batch,
    generate_unsubmitted_exploration_batch,
)
from execution.backtests import prepare_backtest_task
from generation.catalog import (
    CatalogContext,
    FieldDefinition,
    GenerationCatalog,
    OperatorDefinition,
    OperatorParameter,
    WindowDefinition,
)
from generation.exploration import ExplorationResult, explore_formula
from generation.formula import formula_fingerprint, render_formula
from persistence.backtests import initialize_backtest_schema
from persistence.database import open_database


class ExplorationBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = GenerationCatalog(
            context=CatalogContext("EQUITY", "USA", "TOP3000", 1),
            fields=tuple(
                self._field(field_id)
                for field_id in ("close", "open", "volume", "returns")
            ),
            operators=(
                self._operator(
                    "rank",
                    "Cross Sectional",
                    OperatorParameter("x", "expr"),
                ),
                self._operator(
                    "zscore",
                    "Cross Sectional",
                    OperatorParameter("x", "expr"),
                ),
                self._operator(
                    "ts_mean",
                    "Time Series",
                    OperatorParameter("x", "expr"),
                    OperatorParameter("d", "window"),
                ),
                self._operator(
                    "ts_rank",
                    "Time Series",
                    OperatorParameter("x", "expr"),
                    OperatorParameter("d", "window"),
                ),
            ),
            windows=tuple(
                WindowDefinition(value, period)
                for value, period in (
                    (5, "week"),
                    (22, "month"),
                    (66, "quarter"),
                    (120, "half_year"),
                    (250, "year"),
                )
            ),
        )
        self.fields = ("close", "open", "volume", "returns")

    def test_builds_target_count_with_unique_formulas_and_source_seeds(self) -> None:
        batch = self._generate(target_count=5, seeds=tuple(range(20)))

        self.assertEqual(len(batch.candidates), 5)
        self.assertEqual(batch.attempted_seed_count, 5)
        self.assertIsNone(batch.shortfall)
        self.assertEqual(
            len(
                {
                    formula_fingerprint(candidate.expression)
                    for candidate in batch.candidates
                }
            ),
            5,
        )
        self.assertEqual(
            tuple(candidate.seed for candidate in batch.candidates),
            (0, 1, 2, 3, 4),
        )

    def test_existing_formula_is_normalized_excluded_and_later_seed_is_used(
        self,
    ) -> None:
        existing = self._single_formula(seed=0)

        batch = self._generate(
            target_count=1,
            seeds=(0, 1, 2),
            existing_formulas=(f"  {existing}  ",),
        )

        self.assertEqual(len(batch.candidates), 1)
        self.assertNotEqual(batch.candidates[0].seed, 0)
        self.assertEqual(batch.attempted_seed_count, 2)
        self.assertEqual(
            batch.exclusions,
            (ExclusionCount("backtest_formula_duplicate", 1),),
        )
        self.assertIsNone(batch.shortfall)

    def test_batch_duplicates_are_excluded_and_reported_on_shortfall(self) -> None:
        duplicate_seeds = self._find_duplicate_seeds()

        batch = self._generate(
            target_count=2,
            seeds=duplicate_seeds,
        )

        self.assertEqual(len(batch.candidates), 1)
        self.assertEqual(batch.attempted_seed_count, 2)
        self.assertEqual(
            batch.shortfall,
            BatchShortfall(
                missing_count=1,
            ),
        )
        self.assertEqual(
            batch.exclusions,
            (ExclusionCount("batch_formula_duplicate", 1),),
        )

    def test_seed_exhaustion_returns_shortfall_without_hidden_attempts(self) -> None:
        batch = self._generate(target_count=4, seeds=(0, 1))

        self.assertEqual(batch.attempted_seed_count, 2)
        self.assertEqual(len(batch.candidates), 2)
        self.assertEqual(batch.shortfall.missing_count, 2)
        self.assertEqual(batch.exclusions, ())

    def test_exploration_failures_are_aggregated_in_the_shortfall(self) -> None:
        batch = self._generate(
            target_count=2,
            seeds=(0, 1, 2),
            field_candidates=(),
        )

        self.assertEqual(batch.candidates, ())
        self.assertEqual(batch.attempted_seed_count, 3)
        self.assertEqual(
            batch.shortfall,
            BatchShortfall(
                missing_count=2,
            ),
        )
        self.assertEqual(
            batch.exclusions,
            (
                ExclusionCount(
                    "exploration_failure:signal_field_candidates_empty",
                    3,
                ),
            ),
        )

    def test_zero_target_is_a_quiet_noop(self) -> None:
        batch = self._generate(target_count=0, seeds=(0, 1))

        self.assertEqual(batch.candidates, ())
        self.assertEqual(batch.attempted_seed_count, 0)
        self.assertEqual(batch.exclusions, ())
        self.assertIsNone(batch.shortfall)

    def test_invalid_batch_inputs_and_existing_formula_fail_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_target_count_invalid"):
            self._generate(target_count=True, seeds=(0,))
        with self.assertRaisesRegex(ValueError, "batch_seeds_duplicate"):
            self._generate(target_count=1, seeds=(0, 0))
        with self.assertRaisesRegex(
            ValueError,
            "batch_existing_formula_invalid:0:missing_expression",
        ):
            self._generate(
                target_count=1,
                seeds=(0,),
                existing_formulas=("close+",),
            )

    def test_inputs_and_catalog_remain_unchanged(self) -> None:
        seeds = tuple(range(10))
        existing = ("rank(close-open)",)
        before = (
            self.catalog.fingerprint,
            seeds,
            self.fields,
            existing,
        )

        self._generate(
            target_count=3,
            seeds=seeds,
            existing_formulas=existing,
        )

        self.assertEqual(
            (
                self.catalog.fingerprint,
                seeds,
                self.fields,
                existing,
            ),
            before,
        )

    def test_database_generation_returns_in_memory_pool_without_candidate_tables(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "backtests.sqlite3"
            with open_database(database_path) as connection:
                initialize_backtest_schema(connection)
                batch = generate_unsubmitted_exploration_batch(
                    connection,
                    self.catalog,
                    target_count=2,
                    seeds=(0, 1, 2),
                    field_candidates=self.fields,
                )
                tables = tuple(
                    row["name"]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
                    )
                )

        self.assertEqual(len(batch.candidates), 2)
        self.assertNotIn("candidates", tables)
        self.assertNotIn("candidate_batches", tables)

    def test_batch_can_use_vector_and_group_candidates(self) -> None:
        catalog = GenerationCatalog(
            context=self.catalog.context,
            fields=(
                *self.catalog.fields,
                self._field("news", "VECTOR"),
                self._field("sector", "GROUP"),
            ),
            operators=(
                *self.catalog.operators,
                self._operator(
                    "vec_avg",
                    "Vector",
                    OperatorParameter("x", "expr"),
                ),
                self._operator(
                    "group_rank",
                    "Group",
                    OperatorParameter("x", "expr"),
                    OperatorParameter("group", "group"),
                ),
            ),
            windows=self.catalog.windows,
        )

        batch = generate_exploration_batch(
            catalog,
            target_count=40,
            seeds=tuple(range(200)),
            field_candidates=(*self.fields, "news"),
            group_candidates=("sector",),
        )
        formulas = tuple(item.candidate.formula for item in batch.candidates)

        self.assertEqual(len(formulas), 40)
        self.assertIsNone(batch.shortfall)
        self.assertTrue(any("vec_avg(news)" in formula for formula in formulas))
        self.assertTrue(any("group_rank(" in formula for formula in formulas))
        self.assertTrue(any("sector" in formula for formula in formulas))

    def test_only_prepared_backtest_formula_is_excluded_from_later_generation(
        self,
    ) -> None:
        existing_formula = self._single_formula(seed=0)
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "backtests.sqlite3"
            with open_database(database_path) as connection:
                initialize_backtest_schema(connection)
                prepare_backtest_task(
                    connection,
                    account_scope="group-account",
                    formula=existing_formula,
                    settings={"delay": 1},
                    created_at="2026-08-28T00:00:00+00:00",
                )
                batch = generate_unsubmitted_exploration_batch(
                    connection,
                    self.catalog,
                    target_count=1,
                    seeds=(0, 1, 2),
                    field_candidates=self.fields,
                )

        self.assertEqual(len(batch.candidates), 1)
        self.assertNotEqual(batch.candidates[0].candidate.formula, existing_formula)
        self.assertEqual(batch.attempted_seed_count, 2)

    def _generate(self, **overrides):
        arguments = {
            "target_count": 3,
            "seeds": tuple(range(20)),
            "field_candidates": self.fields,
            "operator_candidates": (),
            "existing_formulas": (),
        }
        arguments.update(overrides)
        return generate_exploration_batch(self.catalog, **arguments)

    def _single_formula(self, seed: int) -> str:
        outcome = explore_formula(
            self.catalog,
            seed=seed,
            field_candidates=self.fields,
        )
        self.assertIsInstance(outcome, ExplorationResult)
        assert isinstance(outcome, ExplorationResult)
        return render_formula(outcome.expression)

    def _find_duplicate_seeds(self) -> tuple[int, int]:
        seen: dict[str, int] = {}
        for seed in range(500):
            formula = self._single_formula(seed)
            if formula in seen:
                return seen[formula], seed
            seen[formula] = seed
        self.fail("duplicate seed fixture not found")

    @staticmethod
    def _field(field_id: str, field_type: str = "MATRIX") -> FieldDefinition:
        return FieldDefinition(
            field_id=field_id,
            dataset_id="dataset",
            category="sample",
            subcategory=None,
            field_type=field_type,
            coverage=1.0,
        )

    @staticmethod
    def _operator(
        name: str,
        category: str,
        *parameters: OperatorParameter,
    ) -> OperatorDefinition:
        return OperatorDefinition(
            name=name,
            category=category,
            scope=("REGULAR",),
            parameters=tuple(parameters),
            roles=(
                ("cross_sectional_normalization",)
                if category == "Cross Sectional"
                else (
                    ("time_series_normalization",) if category == "Time Series" else ()
                )
            ),
            output_kind="signal",
        )


if __name__ == "__main__":
    unittest.main()
