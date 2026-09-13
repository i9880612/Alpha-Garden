from __future__ import annotations

from generation.arguments import bind_operator_arguments
from generation.catalog import GenerationCatalog
from generation.formula import Binary, Call, Expression, Literal, Prefix
from generation.validation import ValidationIssue


_COMPARISONS = frozenset({">", ">=", "<", "<=", "==", "!="})
_COMPARISON_CALLS = frozenset(
    {"equal", "greater", "greater_equal", "less", "less_equal", "not_equal"}
)
_DIMENSIONLESS_INPUT_INDEXES = {
    "bucket": (0,),
    "group_backfill": (0,),
    "hump": (0,),
    "inverse": (0,),
    "log": (0,),
    "power": (0,),
    "signed_power": (0,),
    "sqrt": (0,),
    "ts_product": (0,),
    "ts_regression": (0, 1),
}


def find_coarse_unit_issues(
    expression: Expression,
    catalog: GenerationCatalog,
) -> tuple[ValidationIssue, ...]:
    """Reject shapes whose unit compatibility cannot be established locally."""
    issues: list[ValidationIssue] = []

    def require_dimensionless(value: Expression, location: str) -> None:
        if not is_proven_dimensionless(value, catalog):
            issues.append(
                ValidationIssue(
                    code="unit_compatibility_unproven",
                    location=location,
                )
            )

    def visit(value: Expression, location: str) -> None:
        if isinstance(value, Binary):
            if value.operator in _COMPARISONS:
                require_dimensionless(value.left, f"{location}.left")
                require_dimensionless(value.right, f"{location}.right")
            visit(value.left, f"{location}.left")
            visit(value.right, f"{location}.right")
            return
        if isinstance(value, Prefix):
            visit(value.operand, f"{location}.operand")
            return
        if not isinstance(value, Call):
            return

        expression_arguments = _call_expression_arguments(value, catalog)
        if value.operator in _COMPARISON_CALLS:
            for index, argument in expression_arguments:
                require_dimensionless(argument, f"{location}.arguments[{index}]")
        elif value.operator in {"add", "max", "min", "subtract", "vector_neut"}:
            for index, argument in expression_arguments:
                require_dimensionless(argument, f"{location}.arguments[{index}]")
        elif value.operator == "if_else":
            for index, argument in expression_arguments[1:3]:
                require_dimensionless(argument, f"{location}.arguments[{index}]")
        elif value.operator == "group_mean" and len(expression_arguments) >= 2:
            index, argument = expression_arguments[1]
            require_dimensionless(argument, f"{location}.arguments[{index}]")
        else:
            for input_index in dimensionless_input_indexes(value.operator):
                if input_index >= len(expression_arguments):
                    continue
                index, argument = expression_arguments[input_index]
                require_dimensionless(argument, f"{location}.arguments[{index}]")

        for index, argument in enumerate(value.arguments):
            visit(argument.value, f"{location}.arguments[{index}]")

    visit(expression, "$")
    return tuple(issues)


def dimensionless_input_indexes(operator_name: str) -> tuple[int, ...]:
    return _DIMENSIONLESS_INPUT_INDEXES.get(operator_name, ())


def is_proven_dimensionless(
    expression: Expression,
    catalog: GenerationCatalog,
) -> bool:
    if isinstance(expression, Literal):
        return True
    if isinstance(expression, Binary):
        if expression.operator in _COMPARISONS:
            return True
        return is_proven_dimensionless(
            expression.left,
            catalog,
        ) and is_proven_dimensionless(expression.right, catalog)
    if isinstance(expression, Prefix):
        return is_proven_dimensionless(expression.operand, catalog)
    if not isinstance(expression, Call):
        return False

    definition = catalog.operator(expression.operator)
    if definition is None:
        return False
    if definition.output_kind == "condition":
        return True
    if "cross_sectional_normalization" in definition.roles:
        return True
    expression_arguments = _call_expression_arguments(expression, catalog)
    if not expression_arguments:
        return expression.operator == "ts_step"
    return all(
        is_proven_dimensionless(argument, catalog)
        for _, argument in expression_arguments
    )


def _call_expression_arguments(
    expression: Call,
    catalog: GenerationCatalog,
) -> list[tuple[int, Expression]]:
    definition = catalog.operator(expression.operator)
    if definition is None:
        return []
    binding = bind_operator_arguments(
        definition.parameters,
        tuple(argument.name for argument in expression.arguments),
    )
    assignments = [
        (
            assignment.parameter_index,
            assignment.argument_index,
            expression.arguments[assignment.argument_index].value,
        )
        for assignment in binding.assignments
        if assignment.parameter.kind == "expr"
    ]
    assignments.sort(key=lambda item: (item[0], item[1]))
    return [(argument_index, value) for _, argument_index, value in assignments]
