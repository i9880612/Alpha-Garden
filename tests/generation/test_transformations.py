from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from generation.catalog import (
    CatalogContext,
    FieldDefinition,
    GenerationCatalog,
    OperatorDefinition,
    OperatorParameter,
    WindowDefinition,
)
from generation.formula import analyze_formula
from generation.parser import parse_formula
from generation.structure_limits import (
    FORMULA_MAX_COMPLEXITY,
    FORMULA_MAX_DEPTH,
)
from generation.transformations import (
    COMPLEMENTARY_SIGNAL_REFRAME,
    DISTRIBUTION_STABILIZATION,
    GROUP_RELATIVE_REFRAME,
    MAX_TRANSFORMATION_LEAVES_PER_FAMILY,
    STRUCTURAL_TRANSFORMATION_FAMILIES,
    TEMPORAL_CHANGE_REFRAME,
    TEMPORAL_PERSISTENCE_REFRAME,
    iter_transformation_leaves,
)
from generation.validation import validate_formula


class StructuralTransformationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = GenerationCatalog(
            context=CatalogContext("EQUITY", "USA", "TOP3000", 1),
            fields=(
                self._field("close", "MATRIX"),
                self._field("open", "MATRIX"),
                self._field("volume", "MATRIX"),
                self._field("sector", "GROUP"),
            ),
            operators=self._base_operators(),
            windows=(
                WindowDefinition(5, "week"),
                WindowDefinition(22, "month"),
            ),
        )
        self.parent = parse_formula("rank(close)").expression

    def test_small_catalog_has_exact_finite_leaves_for_all_families(self) -> None:
        leaves = tuple(
            iter_transformation_leaves(
                self.parent,
                self.catalog,
                field_candidates=("close", "open", "volume"),
                group_candidates=("sector",),
            )
        )

        self.assertEqual(
            Counter(leaf.family for leaf in leaves),
            Counter(
                {
                    TEMPORAL_CHANGE_REFRAME: 4,
                    TEMPORAL_PERSISTENCE_REFRAME: 4,
                    DISTRIBUTION_STABILIZATION: 2,
                    GROUP_RELATIVE_REFRAME: 2,
                    COMPLEMENTARY_SIGNAL_REFRAME: 16,
                }
            ),
        )
        self.assertEqual(len(leaves), 28)

    def test_repeated_enumeration_is_stable_unique_and_legal(self) -> None:
        arguments = {
            "field_candidates": ("close", "open", "volume"),
            "group_candidates": ("sector",),
        }

        first = tuple(
            iter_transformation_leaves(
                self.parent,
                self.catalog,
                **arguments,
            )
        )
        repeated = tuple(
            iter_transformation_leaves(
                self.parent,
                self.catalog,
                **arguments,
            )
        )

        self.assertEqual(first, repeated)
        fingerprints = tuple(leaf.formula_fingerprint for leaf in first)
        self.assertEqual(len(fingerprints), len(set(fingerprints)))
        for leaf in first:
            with self.subTest(leaf_id=leaf.leaf_id):
                self.assertEqual(
                    leaf.leaf_id,
                    f"{leaf.family}:{leaf.formula_fingerprint}",
                )
                self.assertTrue(
                    validate_formula(leaf.expression, self.catalog).is_valid
                )
                self.assertEqual(leaf.change.location, "formula")
                self.assertEqual(leaf.change.before, "rank(close)")
                self.assertNotEqual(leaf.change.before, leaf.change.after)

    def test_every_family_is_capped_at_sixty_four_leaves(self) -> None:
        catalog = self._large_catalog()
        leaves = tuple(
            iter_transformation_leaves(
                parse_formula("rank(close)").expression,
                catalog,
            )
        )

        counts = Counter(leaf.family for leaf in leaves)
        self.assertEqual(
            counts,
            Counter(
                {
                    family: MAX_TRANSFORMATION_LEAVES_PER_FAMILY
                    for family in STRUCTURAL_TRANSFORMATION_FAMILIES
                }
            ),
        )

    def test_capped_families_cover_each_parameter_dimension(self) -> None:
        catalog = self._large_catalog()
        leaves = tuple(
            iter_transformation_leaves(
                self.parent,
                catalog,
            )
        )
        groups = tuple(leaf for leaf in leaves if leaf.family == GROUP_RELATIVE_REFRAME)
        complements = tuple(
            leaf for leaf in leaves if leaf.family == COMPLEMENTARY_SIGNAL_REFRAME
        )

        self.assertGreater(len({leaf.spec.input_normalizer for leaf in groups}), 1)
        self.assertGreater(len({leaf.spec.group_operator for leaf in groups}), 1)
        self.assertNotEqual(
            {leaf.spec.group_field for leaf in groups},
            {f"group_{index:03d}" for index in range(64)},
        )
        self.assertGreater(
            len({leaf.spec.temporal_normalizer for leaf in complements}),
            1,
        )
        self.assertGreater(len({leaf.spec.window for leaf in complements}), 1)
        self.assertNotEqual(
            {leaf.spec.complement_field for leaf in complements},
            {f"signal_{index:03d}" for index in range(64)},
        )

    def test_capped_family_balances_dimensions_for_adversarial_parent_identity(
        self,
    ) -> None:
        catalog = self._large_catalog(group_count=1000)
        parent = parse_formula(
            "zscore(rank(rank(rank(zscore(rank(close))))))"
        ).expression

        leaves = tuple(
            iter_transformation_leaves(
                parent,
                catalog,
                families=(GROUP_RELATIVE_REFRAME,),
            )
        )

        self.assertEqual(len(leaves), MAX_TRANSFORMATION_LEAVES_PER_FAMILY)
        self.assertEqual(len({leaf.spec.input_normalizer for leaf in leaves}), 4)
        self.assertEqual(len({leaf.spec.group_operator for leaf in leaves}), 2)
        self.assertEqual(len({leaf.spec.group_field for leaf in leaves}), 64)

    def test_transformation_leaves_respect_absolute_structure_limits(self) -> None:
        nested = "close"
        for _ in range(FORMULA_MAX_DEPTH - 2):
            nested = f"rank({nested})"
        parent = parse_formula(nested).expression

        leaves = tuple(
            iter_transformation_leaves(
                parent,
                self.catalog,
                field_candidates=("close", "open", "volume"),
                group_candidates=("sector",),
            )
        )

        self.assertEqual(leaves, ())

    def test_transformation_leaves_reject_wide_formula_over_complexity_limit(
        self,
    ) -> None:
        parent = parse_formula(self._balanced_sum(("close", "open") * 16)).expression
        facts = analyze_formula(parent)

        leaves = tuple(
            iter_transformation_leaves(
                parent,
                self.catalog,
                field_candidates=("close", "open", "volume"),
                group_candidates=("sector",),
            )
        )

        self.assertLessEqual(facts.depth, FORMULA_MAX_DEPTH)
        self.assertGreater(facts.complexity, FORMULA_MAX_COMPLEXITY)
        self.assertEqual(leaves, ())

    def test_vector_parent_uses_matrix_sibling_from_same_dataset(self) -> None:
        catalog = GenerationCatalog(
            context=self.catalog.context,
            fields=(
                self._field(
                    "news_vector",
                    "VECTOR",
                    dataset_id="news",
                    category="News",
                    subcategory="Sentiment",
                ),
                self._field(
                    "news_score",
                    "MATRIX",
                    dataset_id="news",
                    category="News",
                    subcategory="Sentiment",
                ),
                self._field(
                    "unrelated_score",
                    "MATRIX",
                    dataset_id="other",
                    category="Model",
                    subcategory="Risk",
                ),
            ),
            operators=(
                *self._base_operators(),
                self._operator(
                    "vec_avg",
                    (),
                    OperatorParameter("x", "expr"),
                ),
            ),
            windows=self.catalog.windows,
        )

        leaves = tuple(
            iter_transformation_leaves(
                parse_formula("vec_avg(news_vector)").expression,
                catalog,
                families=(COMPLEMENTARY_SIGNAL_REFRAME,),
                field_candidates=("news_score", "unrelated_score"),
            )
        )

        self.assertTrue(leaves)
        self.assertEqual(
            {leaf.spec.complement_field for leaf in leaves},
            {"news_score"},
        )
        self.assertTrue(
            all("unrelated_score" not in leaf.change.after for leaf in leaves)
        )

    def test_complementary_family_is_empty_without_any_catalog_relation(
        self,
    ) -> None:
        catalog = GenerationCatalog(
            context=self.catalog.context,
            fields=(
                self._field(
                    "close",
                    "MATRIX",
                    dataset_id="prices",
                    category="Price Volume",
                    subcategory="Price",
                ),
                self._field(
                    "unrelated",
                    "MATRIX",
                    dataset_id="model",
                    category="Model",
                    subcategory="Risk",
                ),
            ),
            operators=self._base_operators(),
            windows=self.catalog.windows,
        )

        leaves = tuple(
            iter_transformation_leaves(
                parse_formula("rank(close)").expression,
                catalog,
                families=(COMPLEMENTARY_SIGNAL_REFRAME,),
                field_candidates=("unrelated",),
            )
        )

        self.assertEqual(leaves, ())

    def test_complementary_family_ignores_blank_catalog_relations(self) -> None:
        catalog = GenerationCatalog(
            context=self.catalog.context,
            fields=(
                self._field(
                    "close",
                    "MATRIX",
                    dataset_id="",
                    category=" ",
                    subcategory="",
                ),
                self._field(
                    "unrelated",
                    "MATRIX",
                    dataset_id="",
                    category=" ",
                    subcategory="",
                ),
            ),
            operators=self._base_operators(),
            windows=self.catalog.windows,
        )

        leaves = tuple(
            iter_transformation_leaves(
                parse_formula("rank(close)").expression,
                catalog,
                families=(COMPLEMENTARY_SIGNAL_REFRAME,),
                field_candidates=("unrelated",),
            )
        )

        self.assertEqual(leaves, ())

    def test_missing_catalog_capabilities_leave_each_family_empty(self) -> None:
        catalog = GenerationCatalog(
            context=self.catalog.context,
            fields=self.catalog.fields,
            operators=(
                self._operator(
                    "rank",
                    ("cross_sectional_normalization",),
                    OperatorParameter("x", "expr"),
                ),
            ),
            windows=self.catalog.windows,
        )

        for family in STRUCTURAL_TRANSFORMATION_FAMILIES:
            with self.subTest(family=family):
                leaves = tuple(
                    iter_transformation_leaves(
                        parse_formula("rank(close)").expression,
                        catalog,
                        families=(family,),
                        field_candidates=("open",),
                        group_candidates=("sector",),
                    )
                )
                self.assertEqual(leaves, ())

    def test_request_boundary_validation_fails_closed(self) -> None:
        for families in (
            (),
            (TEMPORAL_CHANGE_REFRAME, TEMPORAL_CHANGE_REFRAME),
            ("unsupported",),
        ):
            with (
                self.subTest(families=families),
                self.assertRaisesRegex(
                    ValueError,
                    "structural_transformation_families_invalid",
                ),
            ):
                tuple(
                    iter_transformation_leaves(
                        self.parent,
                        self.catalog,
                        families=families,
                    )
                )

        for keyword, value, error in (
            (
                "field_candidates",
                ("close", "close"),
                "structural_transformation_field_candidates_invalid",
            ),
            (
                "field_candidates",
                ("missing",),
                "structural_transformation_field_candidates_invalid:missing",
            ),
            (
                "field_candidates",
                ("sector",),
                "structural_transformation_field_candidates_invalid:sector",
            ),
            (
                "group_candidates",
                ("close",),
                "structural_transformation_group_candidates_invalid:close",
            ),
        ):
            with (
                self.subTest(keyword=keyword, value=value),
                self.assertRaisesRegex(
                    ValueError,
                    error,
                ),
            ):
                tuple(
                    iter_transformation_leaves(
                        self.parent,
                        self.catalog,
                        **{keyword: value},
                    )
                )

        with self.assertRaisesRegex(
            ValueError,
            "structural_transformation_parent_invalid",
        ):
            tuple(
                iter_transformation_leaves(
                    parse_formula("missing").expression,
                    self.catalog,
                )
            )

    def _large_catalog(
        self,
        *,
        signal_count: int = 100,
        group_count: int = 100,
    ) -> GenerationCatalog:
        return GenerationCatalog(
            context=self.catalog.context,
            fields=(
                self._field("close", "MATRIX"),
                *(
                    self._field(f"signal_{index:03d}", "MATRIX")
                    for index in range(signal_count)
                ),
                *(
                    self._field(f"group_{index:03d}", "GROUP")
                    for index in range(group_count)
                ),
            ),
            operators=(
                self._operator(
                    "rank",
                    ("cross_sectional_normalization",),
                    OperatorParameter("x", "expr"),
                ),
                self._operator(
                    "zscore",
                    ("cross_sectional_normalization",),
                    OperatorParameter("x", "expr"),
                ),
                self._operator(
                    "normalize_a",
                    ("cross_sectional_normalization",),
                    OperatorParameter("x", "expr"),
                ),
                self._operator(
                    "normalize_b",
                    ("cross_sectional_normalization",),
                    OperatorParameter("x", "expr"),
                ),
                *(
                    self._operator(
                        f"change_{index}",
                        ("time_series_change",),
                        OperatorParameter("x", "expr"),
                        OperatorParameter("d", "window"),
                    )
                    for index in range(2)
                ),
                *(
                    self._operator(
                        f"smooth_{index}",
                        ("time_series_smoothing",),
                        OperatorParameter("x", "expr"),
                        OperatorParameter("d", "window"),
                    )
                    for index in range(2)
                ),
                *(
                    self._operator(
                        f"control_{index:02d}",
                        ("cross_sectional_outlier_control",),
                        OperatorParameter("x", "expr"),
                    )
                    for index in range(33)
                ),
                self._operator(
                    "ts_rank",
                    ("time_series_normalization",),
                    OperatorParameter("x", "expr"),
                    OperatorParameter("d", "window"),
                ),
                self._operator(
                    "ts_zscore",
                    ("time_series_normalization",),
                    OperatorParameter("x", "expr"),
                    OperatorParameter("d", "window"),
                ),
                self._operator(
                    "group_rank",
                    (),
                    OperatorParameter("x", "expr"),
                    OperatorParameter("group", "group"),
                ),
                self._operator(
                    "group_zscore",
                    (),
                    OperatorParameter("x", "expr"),
                    OperatorParameter("group", "group"),
                ),
            ),
            windows=tuple(
                WindowDefinition(index, f"window_{index}") for index in range(1, 18)
            ),
        )

    @staticmethod
    def _balanced_sum(terms: tuple[str, ...]) -> str:
        level = terms
        while len(level) > 1:
            level = tuple(
                f"({level[index]}+{level[index + 1]})"
                for index in range(0, len(level), 2)
            )
        return level[0]

    def _base_operators(self) -> tuple[OperatorDefinition, ...]:
        return (
            self._operator(
                "rank",
                ("cross_sectional_normalization",),
                OperatorParameter("x", "expr"),
            ),
            self._operator(
                "zscore",
                ("cross_sectional_normalization",),
                OperatorParameter("x", "expr"),
            ),
            self._operator(
                "winsorize",
                ("cross_sectional_outlier_control",),
                OperatorParameter("x", "expr"),
                OperatorParameter("std", "float", optional=True),
            ),
            self._operator(
                "ts_delta",
                ("time_series_change",),
                OperatorParameter("x", "expr"),
                OperatorParameter("d", "window"),
            ),
            self._operator(
                "ts_mean",
                ("time_series_smoothing",),
                OperatorParameter("x", "expr"),
                OperatorParameter("d", "window"),
            ),
            self._operator(
                "ts_rank",
                ("time_series_normalization",),
                OperatorParameter("x", "expr"),
                OperatorParameter("d", "window"),
            ),
            self._operator(
                "group_rank",
                (),
                OperatorParameter("x", "expr"),
                OperatorParameter("group", "group"),
            ),
        )

    @staticmethod
    def _field(
        field_id: str,
        field_type: str,
        *,
        dataset_id: str = "dataset",
        category: str = "sample",
        subcategory: str | None = None,
        coverage: float | None = 1.0,
    ) -> FieldDefinition:
        return FieldDefinition(
            field_id=field_id,
            dataset_id=dataset_id,
            category=category,
            subcategory=subcategory,
            field_type=field_type,
            coverage=coverage,
        )

    @staticmethod
    def _operator(
        name: str,
        roles: tuple[str, ...],
        *parameters: OperatorParameter,
    ) -> OperatorDefinition:
        return OperatorDefinition(
            name=name,
            category="sample",
            scope=("REGULAR",),
            parameters=tuple(parameters),
            roles=roles,
            output_kind="signal",
        )


if __name__ == "__main__":
    unittest.main()
