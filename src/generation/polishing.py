from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TypeAlias

from generation.arguments import bind_operator_arguments
from generation.catalog import (
    GenerationCatalog,
    OperatorDefinition,
    OperatorParameter,
)
from generation.formula import (
    Binary,
    Call,
    CallArgument,
    Expression,
    Literal,
    Prefix,
    formula_fingerprint,
)
from generation.logic import prepare_formula_logic
from generation.unit_validation import find_coarse_unit_issues
from generation.validation import validate_formula


SINGLE_WINDOW_MUTATION = "single_window_mutation"
POLISHING_FAMILIES = (SINGLE_WINDOW_MUTATION,)

_Path: TypeAlias = tuple[str | int, ...]


@dataclass(frozen=True, slots=True)
class WindowMutationChange:
    location: str
    before: str
    after: str


@dataclass(frozen=True, slots=True)
class WindowMutationLeaf:
    leaf_id: str
    formula_fingerprint: str
    family: str
    expression: Expression
    change: WindowMutationChange


@dataclass(frozen=True, slots=True)
class _WindowSite:
    path: _Path
    value: int


def iter_window_mutation_leaves(
    expression: Expression,
    catalog: GenerationCatalog,
) -> Iterator[WindowMutationLeaf]:
    if not isinstance(catalog, GenerationCatalog):
        raise ValueError("polishing_catalog_invalid")
    parent_validation = validate_formula(expression, catalog)
    if not parent_validation.is_valid or find_coarse_unit_issues(expression, catalog):
        raise ValueError("polishing_parent_invalid")
    parent_logic = prepare_formula_logic(expression)
    if parent_logic.issue is not None or parent_logic.expression != expression:
        return

    windows = catalog.window_values()
    if not windows:
        return
    used_fingerprints: set[str] = set()
    for site in _window_sites(expression, catalog):
        for replacement in windows:
            if replacement == site.value:
                continue
            transformed = _replace_at(
                expression,
                site.path,
                Literal(str(replacement), "integer"),
            )
            prepared = _prepare_leaf(transformed, catalog)
            if prepared is None:
                continue
            fingerprint = formula_fingerprint(prepared)
            if fingerprint in used_fingerprints:
                continue
            used_fingerprints.add(fingerprint)
            yield WindowMutationLeaf(
                leaf_id=f"{SINGLE_WINDOW_MUTATION}:{fingerprint}",
                formula_fingerprint=fingerprint,
                family=SINGLE_WINDOW_MUTATION,
                expression=prepared,
                change=WindowMutationChange(
                    location=_render_location(site.path),
                    before=str(site.value),
                    after=str(replacement),
                ),
            )


def _window_sites(
    expression: Expression,
    catalog: GenerationCatalog,
    *,
    path: _Path = (),
) -> Iterator[_WindowSite]:
    if isinstance(expression, Call):
        operator = catalog.operator(expression.operator)
        if operator is not None:
            for parameter, value, argument_index in _bind_arguments(
                expression,
                operator,
            ):
                if parameter.kind == "window":
                    integer = _integer_value(value)
                    if integer is not None:
                        yield _WindowSite(path + (argument_index,), integer)
        for index, argument in enumerate(expression.arguments):
            yield from _window_sites(
                argument.value,
                catalog,
                path=path + (index,),
            )
        return
    if isinstance(expression, Binary):
        yield from _window_sites(
            expression.left,
            catalog,
            path=path + ("left",),
        )
        yield from _window_sites(
            expression.right,
            catalog,
            path=path + ("right",),
        )
        return
    if isinstance(expression, Prefix):
        yield from _window_sites(
            expression.operand,
            catalog,
            path=path + ("operand",),
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


def _prepare_leaf(
    expression: Expression,
    catalog: GenerationCatalog,
) -> Expression | None:
    validation = validate_formula(expression, catalog)
    if not validation.is_valid or find_coarse_unit_issues(expression, catalog):
        return None
    logic = prepare_formula_logic(expression)
    if logic.issue is not None:
        return None
    prepared_validation = validate_formula(logic.expression, catalog)
    if not prepared_validation.is_valid or find_coarse_unit_issues(
        logic.expression,
        catalog,
    ):
        return None
    return logic.expression


def _replace_at(
    expression: Expression,
    path: _Path,
    replacement: Expression,
) -> Expression:
    if not path:
        return replacement
    head, *tail = path
    remaining = tuple(tail)
    if isinstance(expression, Binary) and head == "left":
        return Binary(
            expression.operator,
            _replace_at(expression.left, remaining, replacement),
            expression.right,
        )
    if isinstance(expression, Binary) and head == "right":
        return Binary(
            expression.operator,
            expression.left,
            _replace_at(expression.right, remaining, replacement),
        )
    if isinstance(expression, Prefix) and head == "operand":
        return Prefix(
            expression.operator,
            _replace_at(expression.operand, remaining, replacement),
        )
    if isinstance(expression, Call) and isinstance(head, int):
        arguments = list(expression.arguments)
        argument = arguments[head]
        arguments[head] = CallArgument(
            _replace_at(argument.value, remaining, replacement),
            name=argument.name,
        )
        return Call(expression.operator, tuple(arguments))
    raise ValueError("polishing_path_invalid")


def _integer_value(expression: Expression) -> int | None:
    if isinstance(expression, Literal) and expression.kind == "integer":
        try:
            return int(expression.value)
        except ValueError:
            return None
    if (
        isinstance(expression, Prefix)
        and expression.operator in {"+", "-"}
        and isinstance(expression.operand, Literal)
        and expression.operand.kind == "integer"
    ):
        try:
            value = int(expression.operand.value)
        except ValueError:
            return None
        return value if expression.operator == "+" else -value
    return None


def _render_location(path: _Path) -> str:
    location = "formula"
    for part in path:
        if isinstance(part, int):
            location += f".arguments[{part}]"
        else:
            location += f".{part}"
    return location
