from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import TypeAlias

from generation.arguments import bind_operator_arguments
from generation.catalog import GenerationCatalog


@dataclass(frozen=True, slots=True)
class Name:
    value: str


@dataclass(frozen=True, slots=True)
class Literal:
    value: str
    kind: str


@dataclass(frozen=True, slots=True)
class CallArgument:
    value: Expression
    name: str | None = None


@dataclass(frozen=True, slots=True)
class Call:
    operator: str
    arguments: tuple[CallArgument, ...]


@dataclass(frozen=True, slots=True)
class Binary:
    operator: str
    left: Expression
    right: Expression


@dataclass(frozen=True, slots=True)
class Prefix:
    operator: str
    operand: Expression


Expression: TypeAlias = Name | Literal | Call | Binary | Prefix


@dataclass(frozen=True, slots=True)
class FormulaFacts:
    referenced_names: tuple[str, ...]
    operator_names: tuple[str, ...]
    depth: int
    complexity: int


_BINARY_PRECEDENCE = {
    ">": 10,
    ">=": 10,
    "<": 10,
    "<=": 10,
    "==": 10,
    "!=": 10,
    "+": 20,
    "-": 20,
    "*": 30,
    "/": 30,
}
_PREFIX_PRECEDENCE = 40
_ATOM_PRECEDENCE = 100


def render_formula(expression: Expression) -> str:
    return _render(expression)


def formula_fingerprint(expression: Expression) -> str:
    normalized = render_formula(expression)
    return sha256(normalized.encode("utf-8")).hexdigest()


def formula_structure_signature(
    expression: Expression,
    catalog: GenerationCatalog,
) -> str:
    """Describe a formula tree while treating window choices as one family."""
    if not isinstance(catalog, GenerationCatalog):
        raise ValueError("formula_structure_catalog_invalid")
    payload = _structure_payload(expression, catalog)
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def formula_structure_fingerprint(
    expression: Expression,
    catalog: GenerationCatalog,
) -> str:
    signature = formula_structure_signature(expression, catalog)
    return sha256(signature.encode("utf-8")).hexdigest()


def analyze_formula(expression: Expression) -> FormulaFacts:
    referenced_names: list[str] = []
    operator_names: list[str] = []

    def visit(node: Expression) -> tuple[int, int]:
        if isinstance(node, Name):
            referenced_names.append(node.value)
            return 1, 1
        if isinstance(node, Literal):
            return 1, 1
        if isinstance(node, Call):
            operator_names.append(node.operator)
            child_metrics = [visit(argument.value) for argument in node.arguments]
            depth = 1 + max((item[0] for item in child_metrics), default=0)
            complexity = 1 + sum(item[1] for item in child_metrics)
            return depth, complexity
        if isinstance(node, Binary):
            operator_names.append(node.operator)
            left_depth, left_complexity = visit(node.left)
            right_depth, right_complexity = visit(node.right)
            return (
                1 + max(left_depth, right_depth),
                1 + left_complexity + right_complexity,
            )
        if isinstance(node, Prefix):
            operator_names.append(node.operator)
            depth, complexity = visit(node.operand)
            return 1 + depth, 1 + complexity
        raise TypeError(f"unsupported_expression:{type(node).__name__}")

    depth, complexity = visit(expression)
    return FormulaFacts(
        referenced_names=tuple(referenced_names),
        operator_names=tuple(operator_names),
        depth=depth,
        complexity=complexity,
    )


def _structure_payload(
    expression: Expression,
    catalog: GenerationCatalog,
    *,
    parameter_kind: str | None = None,
) -> object:
    if parameter_kind == "window":
        return ["window"]
    if isinstance(expression, Name):
        return ["name", expression.value]
    if isinstance(expression, Literal):
        return ["literal", expression.kind, expression.value]
    if isinstance(expression, Prefix):
        return [
            "prefix",
            expression.operator,
            _structure_payload(expression.operand, catalog),
        ]
    if isinstance(expression, Binary):
        return [
            "binary",
            expression.operator,
            _structure_payload(expression.left, catalog),
            _structure_payload(expression.right, catalog),
        ]
    if isinstance(expression, Call):
        operator = catalog.operator(expression.operator)
        if operator is None:
            return [
                "call",
                expression.operator,
                [
                    [
                        "unbound",
                        index,
                        argument.name,
                        _structure_payload(argument.value, catalog),
                    ]
                    for index, argument in enumerate(expression.arguments)
                ],
            ]
        binding = bind_operator_arguments(
            operator.parameters,
            tuple(argument.name for argument in expression.arguments),
        )
        arguments = [
            [
                assignment.parameter.name,
                _structure_payload(
                    expression.arguments[assignment.argument_index].value,
                    catalog,
                    parameter_kind=assignment.parameter.kind,
                ),
            ]
            for assignment in sorted(
                binding.assignments,
                key=lambda item: (item.parameter_index, item.argument_index),
            )
        ]
        assigned_argument_indexes = {
            assignment.argument_index for assignment in binding.assignments
        }
        arguments.extend(
            [
                "unbound",
                index,
                argument.name,
                _structure_payload(argument.value, catalog),
            ]
            for index, argument in enumerate(expression.arguments)
            if index not in assigned_argument_indexes
        )
        return ["call", expression.operator, arguments]
    raise TypeError(f"unsupported_expression:{type(expression).__name__}")


def _render(
    expression: Expression,
    *,
    parent_precedence: int = 0,
    is_right_child: bool = False,
) -> str:
    if isinstance(expression, Name):
        return expression.value
    if isinstance(expression, Literal):
        return expression.value
    if isinstance(expression, Call):
        rendered_arguments = []
        for argument in expression.arguments:
            rendered = _render(argument.value)
            if argument.name is not None:
                rendered = f"{argument.name}={rendered}"
            rendered_arguments.append(rendered)
        return f"{expression.operator}({','.join(rendered_arguments)})"
    if isinstance(expression, Prefix):
        rendered_operand = _render(
            expression.operand,
            parent_precedence=_PREFIX_PRECEDENCE,
            is_right_child=True,
        )
        rendered = f"{expression.operator}{rendered_operand}"
        if _PREFIX_PRECEDENCE < parent_precedence:
            return f"({rendered})"
        return rendered
    if isinstance(expression, Binary):
        precedence = _BINARY_PRECEDENCE[expression.operator]
        rendered_left = _render(
            expression.left,
            parent_precedence=precedence,
            is_right_child=False,
        )
        rendered_right = _render(
            expression.right,
            parent_precedence=precedence,
            is_right_child=True,
        )
        rendered = f"{rendered_left}{expression.operator}{rendered_right}"
        needs_grouping = precedence < parent_precedence or (
            is_right_child and precedence == parent_precedence
        )
        if needs_grouping:
            return f"({rendered})"
        return rendered
    raise TypeError(f"unsupported_expression:{type(expression).__name__}")
