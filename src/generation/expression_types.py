from __future__ import annotations

from dataclasses import dataclass

from generation.arguments import bind_operator_arguments
from generation.catalog import GenerationCatalog, OperatorDefinition, OperatorParameter
from generation.formula import Binary, Call, Expression, Literal, Name, Prefix


@dataclass(frozen=True, slots=True)
class ExpressionTypeIssue:
    code: str
    location: str


@dataclass(frozen=True, slots=True)
class ExpressionTypeResult:
    kind: str | None
    issues: tuple[ExpressionTypeIssue, ...]

    @property
    def is_valid(self) -> bool:
        return not self.issues


def infer_formula_type(
    expression: Expression,
    catalog: GenerationCatalog,
) -> ExpressionTypeResult:
    issues: list[ExpressionTypeIssue] = []
    kind = _infer(expression, catalog, issues, location="formula")
    if kind is not None and kind != "signal":
        issues.append(
            ExpressionTypeIssue(
                code=f"formula_output_type_mismatch:{kind}",
                location="formula",
            )
        )
    return ExpressionTypeResult(kind=kind, issues=tuple(issues))


def _infer(
    expression: Expression,
    catalog: GenerationCatalog,
    issues: list[ExpressionTypeIssue],
    *,
    location: str,
) -> str | None:
    if isinstance(expression, Name):
        field = catalog.field(expression.value)
        if field is None:
            return None
        return {
            "MATRIX": "signal",
            "VECTOR": "vector",
            "GROUP": "group",
        }.get(field.field_type or "")
    if isinstance(expression, Literal):
        return {
            "integer": "scalar",
            "float": "scalar",
            "boolean": "condition",
            "string": "string",
        }.get(expression.kind)
    if isinstance(expression, Prefix):
        operand_kind = _infer(
            expression.operand,
            catalog,
            issues,
            location=f"{location}.operand",
        )
        if operand_kind in {"group", "string", "vector"}:
            issues.append(
                ExpressionTypeIssue(
                    code=f"prefix_operand_type_mismatch:{operand_kind}",
                    location=f"{location}.operand",
                )
            )
            return None
        return operand_kind
    if isinstance(expression, Binary):
        return _infer_binary(expression, catalog, issues, location=location)
    if isinstance(expression, Call):
        return _infer_call(expression, catalog, issues, location=location)
    raise TypeError(f"unsupported_expression:{type(expression).__name__}")


def _infer_binary(
    expression: Binary,
    catalog: GenerationCatalog,
    issues: list[ExpressionTypeIssue],
    *,
    location: str,
) -> str | None:
    left_kind = _infer(
        expression.left,
        catalog,
        issues,
        location=f"{location}.left",
    )
    right_kind = _infer(
        expression.right,
        catalog,
        issues,
        location=f"{location}.right",
    )
    allowed = {"condition", "scalar", "signal"}
    for kind, operand_location in (
        (left_kind, f"{location}.left"),
        (right_kind, f"{location}.right"),
    ):
        if kind is not None and kind not in allowed:
            issues.append(
                ExpressionTypeIssue(
                    code=f"binary_operand_type_mismatch:{kind}",
                    location=operand_location,
                )
            )
    if left_kind is None or right_kind is None:
        return None
    if left_kind not in allowed or right_kind not in allowed:
        return None
    if expression.operator in {">", ">=", "<", "<=", "==", "!="}:
        return "condition"
    if left_kind == "scalar" and right_kind == "scalar":
        return "scalar"
    return "signal"


def _infer_call(
    expression: Call,
    catalog: GenerationCatalog,
    issues: list[ExpressionTypeIssue],
    *,
    location: str,
) -> str | None:
    operator = catalog.operator(expression.operator)
    if operator is None:
        return None
    assignments = _bind_arguments(expression, operator)
    inferred: dict[str, str | None] = {}
    for parameter, argument, argument_index in assignments:
        argument_location = f"{location}.arguments[{argument_index}]"
        if parameter.kind == "string" and isinstance(argument, Name):
            argument_kind = "string"
        else:
            argument_kind = _infer(
                argument,
                catalog,
                issues,
                location=argument_location,
            )
        inferred[parameter.name] = argument_kind
        _validate_argument_kind(
            operator,
            parameter,
            argument_kind,
            issues,
            location=argument_location,
        )
    _validate_condition_arguments(operator, inferred, issues, location=location)
    return operator.output_kind


def _validate_argument_kind(
    operator: OperatorDefinition,
    parameter: OperatorParameter,
    actual: str | None,
    issues: list[ExpressionTypeIssue],
    *,
    location: str,
) -> None:
    if actual is None:
        return
    if parameter.kind == "group":
        allowed = {"group"}
    elif parameter.kind == "expr":
        if operator.name in {"vec_avg", "vec_sum"} and parameter.name == "x":
            allowed = {"vector"}
        elif operator.name == "densify" and parameter.name == "x":
            allowed = {"group"}
        else:
            allowed = {"condition", "scalar", "signal"}
    elif parameter.kind in {"float", "int", "window"}:
        allowed = {"scalar"}
    elif parameter.kind == "bool":
        allowed = {"condition"}
    elif parameter.kind == "string":
        allowed = {"string"}
    else:
        return
    if actual not in allowed:
        issues.append(
            ExpressionTypeIssue(
                code=(
                    f"operator_argument_type_mismatch:{operator.name}:"
                    f"{parameter.name}:{actual}"
                ),
                location=location,
            )
        )


def _validate_condition_arguments(
    operator: OperatorDefinition,
    inferred: dict[str, str | None],
    issues: list[ExpressionTypeIssue],
    *,
    location: str,
) -> None:
    required = {
        "and": ("input1", "input2"),
        "if_else": ("input1",),
        "not": ("x",),
        "or": ("input1", "input2"),
        "trade_when": ("x", "z"),
    }.get(operator.name, ())
    for parameter_name in required:
        actual = inferred.get(parameter_name)
        if actual is not None and actual != "condition":
            issues.append(
                ExpressionTypeIssue(
                    code=(
                        f"condition_argument_required:{operator.name}:"
                        f"{parameter_name}:{actual}"
                    ),
                    location=location,
                )
            )


def _bind_arguments(
    call: Call,
    operator: OperatorDefinition,
) -> tuple[tuple[OperatorParameter, Expression, int], ...]:
    binding = bind_operator_arguments(
        operator.parameters,
        tuple(argument.name for argument in call.arguments),
    )
    return tuple(
        (
            assignment.parameter,
            call.arguments[assignment.argument_index].value,
            assignment.argument_index,
        )
        for assignment in binding.assignments
    )
