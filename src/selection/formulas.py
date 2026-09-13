from __future__ import annotations

from dataclasses import replace

from generation.arguments import bind_operator_arguments
from generation.catalog import GenerationCatalog
from generation.formula import (
    Binary,
    Call,
    Expression,
    Literal,
    Prefix,
    analyze_formula,
    formula_fingerprint,
    formula_structure_fingerprint,
)
from selection.candidates import DiversityDimension, SelectionCandidate
from selection.risks import assess_candidate_risk
from selection.settings import backtest_settings_fingerprint
from worldquant.backtests import BacktestSettings


_POLISHING_PARENT_SAFETY_CHECKS = frozenset(
    {"CONCENTRATED_WEIGHT", "LOW_SUB_UNIVERSE_SHARPE"}
)
_INHERITABLE_POLISHING_RISK = "uncontrolled_tail"


def formula_selection_candidate(
    expression: Expression,
    catalog: GenerationCatalog,
    settings: BacktestSettings,
) -> SelectionCandidate:
    facts = analyze_formula(expression)
    operators = tuple(
        operator
        for name in dict.fromkeys(facts.operator_names)
        if (operator := catalog.operator(name)) is not None
    )
    fields = tuple(
        field
        for name in dict.fromkeys(facts.referenced_names)
        if (field := catalog.field(name)) is not None
    )
    signal_fields = tuple(
        field for field in fields if field.field_type in {"MATRIX", "VECTOR"}
    )
    dimensions = tuple(
        DiversityDimension(name=name, values=values)
        for name, values in (
            ("operator", tuple(sorted({item.name for item in operators}))),
            (
                "operator_category",
                tuple(
                    sorted(
                        {
                            item.category
                            for item in operators
                            if item.category is not None
                        }
                    )
                ),
            ),
            (
                "operator_output",
                tuple(
                    sorted(
                        {
                            item.output_kind
                            for item in operators
                            if item.output_kind is not None
                        }
                    )
                ),
            ),
            ("field", tuple(sorted({item.field_id for item in fields}))),
            (
                "field_type",
                tuple(
                    sorted(
                        {
                            item.field_type
                            for item in fields
                            if item.field_type is not None
                        }
                    )
                ),
            ),
            (
                "field_category",
                tuple(
                    sorted(
                        {
                            item.category
                            for item in signal_fields
                            if item.category is not None
                        }
                    )
                ),
            ),
            (
                "field_subcategory",
                tuple(
                    sorted(
                        {
                            item.subcategory
                            for item in signal_fields
                            if item.subcategory is not None
                        }
                    )
                ),
            ),
            (
                "field_source",
                tuple(
                    sorted(
                        {
                            item.dataset_id
                            for item in signal_fields
                            if item.dataset_id is not None
                        }
                    )
                ),
            ),
            ("window", _window_values(expression, catalog)),
            (
                "formula_structure",
                (formula_structure_fingerprint(expression, catalog),),
            ),
            ("depth", (str(facts.depth),)),
            ("neutralization", (settings.neutralization,)),
            ("decay", (str(settings.decay),)),
            ("truncation", (format(settings.truncation, ".12g"),)),
            ("delay", (str(settings.delay),)),
        )
        if values
    )
    fingerprint = formula_fingerprint(expression)
    settings_fingerprint = backtest_settings_fingerprint(settings)
    return SelectionCandidate(
        candidate_id=f"{fingerprint}:{settings_fingerprint}",
        family_root_candidate_id=fingerprint,
        diversity=dimensions,
        rejection_reasons=assess_candidate_risk(
            expression,
            catalog,
        ).rejection_reasons,
    )


def formula_polishing_candidate(
    expression: Expression,
    parent_expression: Expression,
    catalog: GenerationCatalog,
    settings: BacktestSettings,
    *,
    parent_passed_checks: tuple[str, ...],
) -> SelectionCandidate:
    if (
        not isinstance(parent_passed_checks, tuple)
        or any(
            not isinstance(check_name, str) or not check_name.strip()
            for check_name in parent_passed_checks
        )
        or len(set(parent_passed_checks)) != len(parent_passed_checks)
    ):
        raise ValueError("polishing_parent_passed_checks_invalid")
    candidate = formula_selection_candidate(expression, catalog, settings)
    if _POLISHING_PARENT_SAFETY_CHECKS - set(parent_passed_checks):
        return candidate
    parent_risks = set(
        assess_candidate_risk(parent_expression, catalog).rejection_reasons
    )
    rejections = tuple(
        reason
        for reason in candidate.rejection_reasons
        if reason != _INHERITABLE_POLISHING_RISK or reason not in parent_risks
    )
    return replace(candidate, rejection_reasons=rejections)


def _window_values(
    expression: Expression,
    catalog: GenerationCatalog,
) -> tuple[str, ...]:
    values: set[int] = set()

    def visit(node: Expression) -> None:
        if isinstance(node, Call):
            operator = catalog.operator(node.operator)
            if operator is not None:
                binding = bind_operator_arguments(
                    operator.parameters,
                    tuple(argument.name for argument in node.arguments),
                )
                for assignment in binding.assignments:
                    argument = node.arguments[assignment.argument_index]
                    if (
                        assignment.parameter.kind == "window"
                        and isinstance(argument.value, Literal)
                        and argument.value.kind == "integer"
                    ):
                        values.add(int(argument.value.value))
            for argument in node.arguments:
                visit(argument.value)
        elif isinstance(node, Binary):
            visit(node.left)
            visit(node.right)
        elif isinstance(node, Prefix):
            visit(node.operand)

    visit(expression)
    return tuple(str(value) for value in sorted(values))
