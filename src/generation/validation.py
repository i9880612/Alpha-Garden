from __future__ import annotations

from dataclasses import dataclass

from generation.arguments import bind_operator_arguments
from generation.catalog import GenerationCatalog, OperatorDefinition, OperatorParameter
from generation.expression_types import infer_formula_type
from generation.formula import Binary, Call, Expression, Literal, Name, Prefix


_REQUIRED_EXPLICIT_PARAMETERS = {
    "ts_backfill": frozenset({"lookback"}),
}


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    location: str


@dataclass(frozen=True, slots=True)
class FormulaValidation:
    issues: tuple[ValidationIssue, ...]

    @property
    def is_valid(self) -> bool:
        return not self.issues


def validate_formula(
    expression: Expression,
    catalog: GenerationCatalog,
) -> FormulaValidation:
    issues: list[ValidationIssue] = []
    _validate_expression(
        expression,
        catalog,
        issues,
        location="formula",
        expected_kind="expr",
    )
    if not issues:
        inferred = infer_formula_type(expression, catalog)
        issues.extend(
            ValidationIssue(code=issue.code, location=issue.location)
            for issue in inferred.issues
        )
    return FormulaValidation(tuple(issues))


def _validate_expression(
    expression: Expression,
    catalog: GenerationCatalog,
    issues: list[ValidationIssue],
    *,
    location: str,
    expected_kind: str,
) -> None:
    if not _matches_kind(expression, expected_kind):
        issues.append(
            ValidationIssue(
                code=f"argument_type_mismatch:{expected_kind}",
                location=location,
            )
        )
        return

    if isinstance(expression, Name):
        if expected_kind in {"expr", "group"}:
            field = catalog.field(expression.value)
            if field is None:
                issues.append(
                    ValidationIssue(
                        code=f"unknown_field:{expression.value}",
                        location=location,
                    )
                )
            elif not _field_matches_kind(field.field_type, expected_kind):
                issues.append(
                    ValidationIssue(
                        code=(
                            f"field_type_mismatch:{expected_kind}:"
                            f"{field.field_type or 'unknown'}"
                        ),
                        location=location,
                    )
                )
        return
    if isinstance(expression, Literal):
        return
    if isinstance(expression, Binary):
        _validate_expression(
            expression.left,
            catalog,
            issues,
            location=f"{location}.left",
            expected_kind="expr",
        )
        _validate_expression(
            expression.right,
            catalog,
            issues,
            location=f"{location}.right",
            expected_kind="expr",
        )
        return
    if isinstance(expression, Prefix):
        _validate_expression(
            expression.operand,
            catalog,
            issues,
            location=f"{location}.operand",
            expected_kind=expected_kind,
        )
        return
    if isinstance(expression, Call):
        _validate_call(expression, catalog, issues, location=location)
        return
    raise TypeError(f"unsupported_expression:{type(expression).__name__}")


def _validate_call(
    call: Call,
    catalog: GenerationCatalog,
    issues: list[ValidationIssue],
    *,
    location: str,
) -> None:
    operator = catalog.operator(call.operator)
    if operator is None:
        issues.append(
            ValidationIssue(
                code=f"unknown_operator:{call.operator}",
                location=location,
            )
        )
        return

    assignments = _assign_arguments(call, operator, issues, location=location)
    assigned_parameter_names = {parameter.name for parameter, _, _ in assignments}
    for parameter_name in _REQUIRED_EXPLICIT_PARAMETERS.get(operator.name, ()):
        if parameter_name not in assigned_parameter_names:
            issues.append(
                ValidationIssue(
                    code=(
                        f"missing_explicit_argument:{operator.name}:"
                        f"{parameter_name}"
                    ),
                    location=location,
                )
            )
    for parameter, argument, argument_location in assignments:
        _validate_expression(
            argument,
            catalog,
            issues,
            location=argument_location,
            expected_kind=(
                "group"
                if operator.name == "densify" and parameter.name == "x"
                else parameter.kind
            ),
        )


def _assign_arguments(
    call: Call,
    operator: OperatorDefinition,
    issues: list[ValidationIssue],
    *,
    location: str,
) -> list[tuple[OperatorParameter, Expression, str]]:
    binding = bind_operator_arguments(
        operator.parameters,
        tuple(argument.name for argument in call.arguments),
    )
    for issue in binding.issues:
        suffix = (
            f":{issue.parameter_name}"
            if issue.parameter_name is not None
            else ""
        )
        issue_location = (
            location
            if issue.argument_index is None
            else f"{location}.arguments[{issue.argument_index}]"
        )
        issues.append(
            ValidationIssue(
                code=f"{issue.code}:{operator.name}{suffix}",
                location=issue_location,
            )
        )
    return [
        (
            assignment.parameter,
            call.arguments[assignment.argument_index].value,
            f"{location}.arguments[{assignment.argument_index}]",
        )
        for assignment in binding.assignments
    ]


def _matches_kind(expression: Expression, kind: str) -> bool:
    if kind == "expr":
        return not isinstance(expression, Literal) or expression.kind in {
            "integer",
            "float",
        }
    if kind == "group":
        return not isinstance(expression, Literal)
    if kind == "window" or kind == "int":
        return _is_signed_literal(expression, {"integer"})
    if kind == "float":
        return _is_signed_literal(expression, {"integer", "float"})
    if kind == "bool":
        return isinstance(expression, Literal) and expression.kind == "boolean"
    if kind == "string":
        return isinstance(expression, Name) or (
            isinstance(expression, Literal) and expression.kind == "string"
        )
    raise ValueError(f"unsupported_parameter_kind:{kind}")


def _is_signed_literal(expression: Expression, kinds: set[str]) -> bool:
    if isinstance(expression, Literal):
        return expression.kind in kinds
    return (
        isinstance(expression, Prefix)
        and expression.operator in {"+", "-"}
        and isinstance(expression.operand, Literal)
        and expression.operand.kind in kinds
    )


def _field_matches_kind(field_type: str | None, expected_kind: str) -> bool:
    if expected_kind == "group":
        return field_type == "GROUP"
    if expected_kind == "expr":
        return field_type in {"MATRIX", "VECTOR"}
    raise ValueError(f"unsupported_field_kind:{expected_kind}")
