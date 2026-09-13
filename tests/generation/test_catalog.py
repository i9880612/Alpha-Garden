from __future__ import annotations

import sys
import unittest
from dataclasses import FrozenInstanceError
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
)


class GenerationCatalogTests(unittest.TestCase):
    def test_catalog_is_sorted_lookupable_and_immutable(self) -> None:
        catalog = self._catalog(
            fields=(self._field("volume"), self._field("close")),
        )

        self.assertEqual([item.field_id for item in catalog.fields], ["close", "volume"])
        self.assertEqual(catalog.field("close"), self._field("close"))
        self.assertIsNone(catalog.field("missing"))
        with self.assertRaises(FrozenInstanceError):
            catalog.context = CatalogContext("EQUITY", "EUR", "TOP3000", 1)  # type: ignore[misc]

    def test_fingerprint_is_independent_of_input_order(self) -> None:
        left = self._catalog(
            fields=(self._field("close"), self._field("volume")),
        )
        right = self._catalog(
            fields=(self._field("volume"), self._field("close")),
        )

        self.assertEqual(left.fingerprint, right.fingerprint)

    def test_duplicate_field_identity_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "generation_field_id_duplicate"):
            self._catalog(fields=(self._field("close"), self._field("close")))

    def test_unsupported_parameter_kind_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "generation_operator_parameter_kind_unsupported:unknown",
        ):
            GenerationCatalog(
                context=CatalogContext("EQUITY", "USA", "TOP3000", 1),
                fields=(self._field("close"),),
                operators=(
                    OperatorDefinition(
                        name="rank",
                        category="Cross Sectional",
                        scope=("REGULAR",),
                        parameters=(OperatorParameter("x", "unknown"),),
                    ),
                ),
            )

    def _catalog(
        self,
        *,
        fields: tuple[FieldDefinition, ...],
    ) -> GenerationCatalog:
        return GenerationCatalog(
            context=CatalogContext("EQUITY", "USA", "TOP3000", 1),
            fields=fields,
            operators=(
                OperatorDefinition(
                    name="rank",
                    category="Cross Sectional",
                    scope=("REGULAR",),
                    parameters=(OperatorParameter("x", "expr"),),
                ),
            ),
        )

    @staticmethod
    def _field(field_id: str) -> FieldDefinition:
        return FieldDefinition(
            field_id=field_id,
            dataset_id="dataset",
            category="market",
            subcategory="price",
            field_type="MATRIX",
            coverage=1.0,
        )


if __name__ == "__main__":
    unittest.main()
