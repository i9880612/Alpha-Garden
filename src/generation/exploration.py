from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from generation.catalog import GenerationCatalog
from generation.formula import Expression, analyze_formula, render_formula
from generation.logic import prepare_formula_logic
from generation.parser import parse_formula
from generation.structure_limits import (
    FORMULA_MAX_DEPTH,
    fits_formula_structure_limits,
)
from generation.structure_builder import build_structure_expression
from generation.unit_validation import find_coarse_unit_issues
from generation.validation import ValidationIssue, validate_formula


@dataclass(frozen=True, slots=True)
class ExplorationResult:
    expression: Expression


@dataclass(frozen=True, slots=True)
class ExplorationFailure:
    code: str
    issues: tuple[ValidationIssue, ...] = ()


ExplorationOutcome: TypeAlias = ExplorationResult | ExplorationFailure


def explore_formula(
    catalog: GenerationCatalog,
    *,
    seed: int,
    field_candidates: tuple[str, ...],
    operator_candidates: tuple[str, ...] = (),
    group_candidates: tuple[str, ...] = (),
) -> ExplorationOutcome:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("exploration_seed_invalid")
    fields = tuple(
        field
        for field_id in field_candidates
        if (field := catalog.field(field_id)) is not None
        and field.field_type in {"MATRIX", "VECTOR"}
        and field.coverage is not None
        and field.coverage > 0
    )
    groups = tuple(
        field
        for field_id in group_candidates
        if (field := catalog.field(field_id)) is not None
        and field.field_type == "GROUP"
    )
    if not fields:
        return ExplorationFailure("signal_field_candidates_empty")
    if not catalog.window_values():
        return ExplorationFailure("window_catalog_empty")

    expression = build_structure_expression(
        catalog,
        seed=seed,
        fields=fields,
        groups=groups,
        max_depth=max(2, FORMULA_MAX_DEPTH // 2),
        operator_candidates=operator_candidates,
    )
    if expression is None:
        return ExplorationFailure("signal_operator_candidates_empty")

    facts = analyze_formula(expression)
    if not fits_formula_structure_limits(expression):
        return ExplorationFailure("structure_budget_exceeded")
    if not any(
        (field := catalog.field(name)) is not None
        and field.field_type in {"MATRIX", "VECTOR"}
        for name in facts.referenced_names
    ):
        return ExplorationFailure("structure_signal_field_missing")

    validation = validate_formula(expression, catalog)
    if not validation.is_valid:
        return ExplorationFailure(
            "explored_formula_invalid",
            validation.issues,
        )
    unit_issues = find_coarse_unit_issues(expression, catalog)
    if unit_issues:
        return ExplorationFailure(
            "deterministic_unit_incompatible",
            unit_issues,
        )
    logic = prepare_formula_logic(expression)
    if logic.issue is not None:
        return ExplorationFailure(
            f"deterministic_logic:{logic.issue.code}",
            (
                ValidationIssue(
                    code=logic.issue.code,
                    location=logic.issue.location,
                ),
            ),
        )
    canonical = parse_formula(render_formula(logic.expression)).expression
    return ExplorationResult(canonical)
