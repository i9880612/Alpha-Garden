from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from generation.formula import (
    Binary,
    Call,
    CallArgument,
    Expression,
    Literal,
    Name,
    Prefix,
)


@dataclass(frozen=True, slots=True)
class LogicIssue:
    code: str
    location: str


@dataclass(frozen=True, slots=True)
class LogicResult:
    expression: Expression
    issue: LogicIssue | None = None

    @property
    def is_valid(self) -> bool:
        return self.issue is None


def prepare_formula_logic(expression: Expression) -> LogicResult:
    simplified, issue = _simplify(expression, location="formula")
    if issue is not None:
        return LogicResult(expression=simplified, issue=issue)
    if _is_constant_expression(simplified):
        return LogicResult(
            expression=simplified,
            issue=LogicIssue(code="constant_expression", location="formula"),
        )
    return LogicResult(expression=simplified)


def _simplify(
    expression: Expression,
    *,
    location: str,
) -> tuple[Expression, LogicIssue | None]:
    if isinstance(expression, (Name, Literal)):
        return expression, None
    if isinstance(expression, Prefix):
        operand, issue = _simplify(
            expression.operand,
            location=f"{location}.operand",
        )
        if issue is not None:
            return Prefix(expression.operator, operand), issue
        if expression.operator == "+":
            return operand, None
        if expression.operator == "-" and isinstance(operand, Prefix):
            if operand.operator == "-":
                return operand.operand, None
        value = _numeric_value(operand)
        if expression.operator == "-" and value is not None:
            return _numeric_literal(-value), None
        return Prefix(expression.operator, operand), None
    if isinstance(expression, Binary):
        return _simplify_binary(expression, location=location)
    if isinstance(expression, Call):
        return _simplify_call(expression, location=location)
    raise TypeError(f"unsupported_expression:{type(expression).__name__}")


def _simplify_binary(
    expression: Binary,
    *,
    location: str,
) -> tuple[Expression, LogicIssue | None]:
    left, issue = _simplify(expression.left, location=f"{location}.left")
    if issue is not None:
        return Binary(expression.operator, left, expression.right), issue
    right, issue = _simplify(expression.right, location=f"{location}.right")
    simplified = Binary(expression.operator, left, right)
    if issue is not None:
        if expression.operator == "/" and _is_zero(right):
            return simplified, LogicIssue(
                code="division_by_zero",
                location=f"{location}.right",
            )
        return simplified, issue

    if expression.operator == "/":
        if _is_zero(right):
            return simplified, LogicIssue(
                code="division_by_zero",
                location=f"{location}.right",
            )
        if left == right:
            return simplified, LogicIssue(
                code="self_division",
                location=location,
            )
        if _is_one(right):
            return left, None

    if expression.operator == "-":
        if left == right:
            return _numeric_literal(Decimal(0)), LogicIssue(
                code="complete_cancellation",
                location=location,
            )
        if _is_zero(right):
            return left, None

    if expression.operator == "+":
        if _are_additive_inverses(left, right):
            return _numeric_literal(Decimal(0)), LogicIssue(
                code="complete_cancellation",
                location=location,
            )
        if _is_zero(left):
            return right, None
        if _is_zero(right):
            return left, None

    if expression.operator == "*":
        if _is_zero(left) or _is_zero(right):
            return _numeric_literal(Decimal(0)), None
        if _is_one(left):
            return right, None
        if _is_one(right):
            return left, None

    left_value = _numeric_value(left)
    right_value = _numeric_value(right)
    if left_value is not None and right_value is not None:
        if expression.operator == "+":
            return _numeric_literal(left_value + right_value), None
        if expression.operator == "-":
            return _numeric_literal(left_value - right_value), None
        if expression.operator == "*":
            return _numeric_literal(left_value * right_value), None
    return simplified, None


def _simplify_call(
    expression: Call,
    *,
    location: str,
) -> tuple[Expression, LogicIssue | None]:
    arguments: list[CallArgument] = []
    first_issue: LogicIssue | None = None
    for index, argument in enumerate(expression.arguments):
        value, issue = _simplify(
            argument.value,
            location=f"{location}.arguments[{index}]",
        )
        arguments.append(CallArgument(value=value, name=argument.name))
        if issue is not None and first_issue is None:
            first_issue = issue

    call = Call(expression.operator, tuple(arguments))
    values = _known_call_values(call)
    if values is None:
        if first_issue is not None:
            return call, first_issue
        return call, None

    if expression.operator == "divide" and len(values) == 2:
        if _is_zero(values[1]):
            return call, LogicIssue(
                code="division_by_zero",
                location=f"{location}.arguments[1]",
            )
        if first_issue is not None:
            return call, first_issue
        if values[0] == values[1]:
            return call, LogicIssue(code="self_division", location=location)
        if _is_one(values[1]):
            return values[0], None
    if expression.operator == "inverse" and len(values) == 1:
        if _is_zero(values[0]):
            return call, LogicIssue(
                code="division_by_zero",
                location=f"{location}.arguments[0]",
            )
        if first_issue is not None:
            return call, first_issue
    if first_issue is not None:
        return call, first_issue
    if expression.operator == "subtract" and len(values) == 2:
        if values[0] == values[1]:
            return _numeric_literal(Decimal(0)), LogicIssue(
                code="complete_cancellation",
                location=location,
            )
        if _is_zero(values[1]):
            return values[0], None
    if expression.operator == "add" and len(values) == 2:
        if _are_additive_inverses(values[0], values[1]):
            return _numeric_literal(Decimal(0)), LogicIssue(
                code="complete_cancellation",
                location=location,
            )
        if _is_zero(values[0]):
            return values[1], None
        if _is_zero(values[1]):
            return values[0], None
    if expression.operator == "multiply" and len(values) == 2:
        if _is_zero(values[0]) or _is_zero(values[1]):
            return _numeric_literal(Decimal(0)), None
        if _is_one(values[0]):
            return values[1], None
        if _is_one(values[1]):
            return values[0], None
    if expression.operator == "reverse" and len(values) == 1:
        if isinstance(values[0], Call) and values[0].operator == "reverse":
            if len(values[0].arguments) == 1:
                return values[0].arguments[0].value, None
        value = _numeric_value(values[0])
        if value is not None:
            return _numeric_literal(-value), None
    return call, None


def _known_call_values(call: Call) -> tuple[Expression, ...] | None:
    parameter_names = {
        "add": ("x", "y"),
        "divide": ("x", "y"),
        "inverse": ("x",),
        "multiply": ("x", "y"),
        "reverse": ("x",),
        "subtract": ("x", "y"),
    }.get(call.operator)
    if parameter_names is None or len(call.arguments) != len(parameter_names):
        return None

    assigned: dict[str, Expression] = {}
    positional_index = 0
    for argument in call.arguments:
        if argument.name is None:
            while (
                positional_index < len(parameter_names)
                and parameter_names[positional_index] in assigned
            ):
                positional_index += 1
            if positional_index == len(parameter_names):
                return None
            name = parameter_names[positional_index]
            positional_index += 1
        else:
            name = argument.name
            if name not in parameter_names:
                return None
        if name in assigned:
            return None
        assigned[name] = argument.value
    if set(assigned) != set(parameter_names):
        return None
    return tuple(assigned[name] for name in parameter_names)


def _are_additive_inverses(left: Expression, right: Expression) -> bool:
    return (
        isinstance(left, Prefix)
        and left.operator == "-"
        and left.operand == right
    ) or (
        isinstance(right, Prefix)
        and right.operator == "-"
        and right.operand == left
    ) or (
        isinstance(left, Call)
        and left.operator == "reverse"
        and len(left.arguments) == 1
        and left.arguments[0].name is None
        and left.arguments[0].value == right
    ) or (
        isinstance(right, Call)
        and right.operator == "reverse"
        and len(right.arguments) == 1
        and right.arguments[0].name is None
        and right.arguments[0].value == left
    )


def _is_constant_expression(expression: Expression) -> bool:
    if isinstance(expression, Literal):
        return expression.kind in {"integer", "float", "boolean", "string"}
    if isinstance(expression, Name):
        return False
    if isinstance(expression, Prefix):
        return _is_constant_expression(expression.operand)
    if isinstance(expression, Binary):
        return _is_constant_expression(expression.left) and _is_constant_expression(
            expression.right
        )
    if isinstance(expression, Call):
        if expression.operator not in {
            "abs",
            "add",
            "divide",
            "inverse",
            "max",
            "min",
            "multiply",
            "power",
            "reverse",
            "sign",
            "signed_power",
            "sqrt",
            "subtract",
        }:
            return False
        return bool(expression.arguments) and all(
            _is_constant_expression(argument.value)
            for argument in expression.arguments
        )
    raise TypeError(f"unsupported_expression:{type(expression).__name__}")


def _numeric_value(expression: Expression) -> Decimal | None:
    if not isinstance(expression, Literal) or expression.kind not in {
        "integer",
        "float",
    }:
        return None
    try:
        value = Decimal(expression.value)
    except InvalidOperation:
        return None
    return value if value.is_finite() else None


def _is_zero(expression: Expression) -> bool:
    return _numeric_value(expression) == 0


def _is_one(expression: Expression) -> bool:
    return _numeric_value(expression) == 1


def _numeric_literal(value: Decimal) -> Literal:
    if value == value.to_integral_value():
        return Literal(str(int(value)), "integer")
    rendered = format(value.normalize(), "f")
    return Literal(rendered, "float")
