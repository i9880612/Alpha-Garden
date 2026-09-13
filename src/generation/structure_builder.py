from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from random import Random
from typing import TypeVar

from generation.catalog import FieldDefinition, GenerationCatalog, OperatorDefinition
from generation.formula import Binary, Call, CallArgument, Expression, Literal, Name
from generation.unit_validation import (
    dimensionless_input_indexes,
    is_proven_dimensionless,
)


_T = TypeVar("_T")
_COMPARISONS = (">", ">=", "<", "<=")
_COMPARISON_CALLS = frozenset(
    {"greater", "greater_equal", "less", "less_equal"}
)
_SCALAR_THRESHOLDS = ("0.25", "0.5", "0.75")
_SAME_UNIT_SIGNAL_OPERATORS = frozenset(
    {"add", "if_else", "max", "min", "subtract", "vector_neut"}
)
_DOMAIN_DEPENDENCIES = {
    "divide": ("abs", "add"),
    "group_mean": ("abs", "add"),
    "inverse": ("abs", "add"),
    "log": ("abs", "add"),
    "power": ("abs",),
    "sqrt": ("abs",),
}
_CONDITION_INPUTS = {
    "and": frozenset({"input1", "input2"}),
    "if_else": frozenset({"input1"}),
    "not": frozenset({"x"}),
    "or": frozenset({"input1", "input2"}),
    "trade_when": frozenset({"x", "z"}),
}


def build_structure_expression(
    catalog: GenerationCatalog,
    *,
    seed: int,
    fields: tuple[FieldDefinition, ...],
    groups: tuple[FieldDefinition, ...],
    max_depth: int,
    operator_candidates: tuple[str, ...] = (),
) -> Expression | None:
    builder = _ExpressionBuilder(
        catalog,
        fields=fields,
        groups=groups,
        max_depth=max_depth,
        random=Random(seed),
        allowed_operators=(
            frozenset(operator_candidates) if operator_candidates else None
        ),
    )
    if not builder.matrix_fields and not (
        builder.vector_fields and builder.vector_reducers
    ):
        return None
    return builder.signal(depth=1, allow_atom=False)


class _ExpressionBuilder:
    def __init__(
        self,
        catalog: GenerationCatalog,
        *,
        fields: tuple[FieldDefinition, ...],
        groups: tuple[FieldDefinition, ...],
        max_depth: int,
        random: Random,
        allowed_operators: frozenset[str] | None,
    ) -> None:
        self.catalog = catalog
        self.matrix_fields = tuple(
            field for field in fields if field.field_type == "MATRIX"
        )
        self.vector_fields = tuple(
            field for field in fields if field.field_type == "VECTOR"
        )
        self.groups = groups
        self.windows = catalog.window_values()
        self.max_depth = max_depth
        self.random = random
        self.allowed_operators = allowed_operators
        self.vector_reducers = tuple(
            operator
            for name in ("vec_avg", "vec_sum")
            if self._operator_enabled(name)
            and (operator := catalog.operator(name)) is not None
        )
        self.signal_fields = self.matrix_fields + (
            self.vector_fields if self.vector_reducers else ()
        )
        self.dimensionless_normalizers = tuple(
            operator
            for operator in catalog.operators
            if self._operator_enabled(operator.name)
            and operator.output_kind == "signal"
            and "cross_sectional_normalization" in operator.roles
            and self._required_parameters_supported(operator)
        )
        self.rank_operator = next(
            (
                operator
                for operator in self.dimensionless_normalizers
                if operator.name == "rank"
            ),
            None,
        )
        self.signal_operators = tuple(
            operator
            for operator in catalog.operators
            if self._operator_enabled(operator.name)
            and self._can_build_signal_operator(operator)
        )
        self.condition_operators = tuple(
            operator
            for operator in catalog.operators
            if self._operator_enabled(operator.name)
            and operator.name not in {"equal", "not_equal"}
            and self.dimensionless_normalizers
            and operator.output_kind == "condition"
            and self._required_parameters_supported(operator)
        )
        self.group_operators = tuple(
            operator
            for operator in catalog.operators
            if self._operator_enabled(operator.name)
            and operator.output_kind == "group"
            and operator.name in {"bucket", "densify"}
            and (operator.name != "bucket" or self.rank_operator is not None)
        )

    def signal(self, *, depth: int, allow_atom: bool = True) -> Expression | None:
        if depth >= self.max_depth:
            return self._signal_atom()
        if allow_atom and self.random.random() < 0.28:
            return self._signal_atom()

        categories: dict[str, list[OperatorDefinition]] = defaultdict(list)
        for operator in self.signal_operators:
            categories[operator.category or "Uncategorized"].append(operator)
        if not categories:
            return self._signal_atom() if allow_atom else None
        category = _pick(tuple(sorted(categories)), self.random)
        operator = _pick(
            tuple(sorted(categories[category], key=lambda item: item.name)),
            self.random,
        )
        return self._signal_call(operator, depth=depth)

    def condition(self, *, depth: int) -> Expression:
        if depth >= self.max_depth or not self.condition_operators:
            return self._base_condition(depth=depth)
        if self.random.random() < 0.45:
            return self._base_condition(depth=depth)
        operator = _pick(self.condition_operators, self.random)
        if operator.name in _COMPARISON_CALLS:
            left, right = self._comparison_operands(depth=depth + 1)
            return Call(
                operator.name,
                (CallArgument(left), CallArgument(right)),
            )
        arguments: list[CallArgument] = []
        for parameter in operator.parameters:
            if parameter.optional:
                continue
            expected = (
                "condition"
                if parameter.name in _CONDITION_INPUTS.get(operator.name, ())
                else "signal"
            )
            value = (
                self.condition(depth=depth + 1)
                if expected == "condition"
                else self._required_signal(depth=depth + 1)
            )
            arguments.append(CallArgument(value))
        return Call(operator.name, tuple(arguments))

    def group(self, *, depth: int) -> Expression:
        if depth >= self.max_depth or not self.group_operators:
            return Name(_pick(self.groups, self.random).field_id)
        if self.random.random() < 0.55:
            return Name(_pick(self.groups, self.random).field_id)
        operator = _pick(self.group_operators, self.random)
        if operator.name == "bucket":
            return self._bucket_group(depth=depth + 1)
        bucket = next(
            (item for item in self.group_operators if item.name == "bucket"),
            None,
        )
        if bucket is not None and self.random.random() < 0.35:
            child: Expression = self._bucket_group(depth=depth + 1)
        else:
            child = Name(_pick(self.groups, self.random).field_id)
        return Call(operator.name, (CallArgument(child),))

    def _bucket_group(self, *, depth: int) -> Call:
        if self.rank_operator is None:
            raise ValueError("structure_bucket_rank_unavailable")
        ranked = Call(
            self.rank_operator.name,
            (CallArgument(self._required_signal(depth=depth)),),
        )
        return Call(
            "bucket",
            (
                CallArgument(ranked),
                CallArgument(Literal('"0,1,0.1"', "string"), name="range"),
            ),
        )

    def _signal_call(self, operator: OperatorDefinition, *, depth: int) -> Expression:
        name = operator.name
        if name in {"vec_avg", "vec_sum"}:
            field = _pick_field(self.vector_fields, self.random)
            return Call(name, (CallArgument(Name(field.field_id)),))
        if name == "divide":
            return Call(
                name,
                (
                    CallArgument(self._required_signal(depth=depth + 1)),
                    CallArgument(self._positive_signal(depth=depth + 1)),
                ),
            )
        if name in {"inverse", "log"}:
            return Call(
                name,
                (CallArgument(self._positive_signal(depth=depth + 1)),),
            )
        if name == "sqrt":
            return Call(
                name,
                (CallArgument(self._absolute_signal(depth=depth + 1)),),
            )
        if name in {"power", "signed_power"}:
            base = self._dimensionless_signal(depth=depth + 1)
            if name == "power":
                base = self._absolute(base)
            exponent = _pick(("0.5", "2"), self.random)
            return Call(
                name,
                (
                    CallArgument(base),
                    CallArgument(
                        Literal(
                            exponent,
                            "float" if "." in exponent else "integer",
                        )
                    ),
                ),
            )
        if name == "kth_element":
            window = _pick(self.windows, self.random)
            k = _pick((1, min(2, window)), self.random)
            return Call(
                name,
                (
                    CallArgument(self._required_signal(depth=depth + 1)),
                    CallArgument(_integer(window)),
                    CallArgument(_integer(k), name="k"),
                    CallArgument(Literal('"NaN"', "string"), name="ignore"),
                ),
            )
        if name == "ts_step":
            return Call(name, (CallArgument(_integer(1)),))
        if name == "ts_backfill":
            return Call(
                name,
                (
                    CallArgument(self._required_signal(depth=depth + 1)),
                    CallArgument(
                        _integer(_pick(self.windows, self.random)),
                        name="lookback",
                    ),
                ),
            )

        arguments: list[CallArgument] = []
        for parameter in operator.parameters:
            if parameter.optional:
                continue
            occurrences = 1
            if parameter.variadic and self.random.random() < 0.5:
                occurrences += 1
            for _ in range(occurrences):
                arguments.append(
                    CallArgument(
                        self._argument(
                            operator,
                            parameter.name,
                            parameter.kind,
                            depth=depth + 1,
                        )
                    )
                )

        if name in {"subtract", "vector_neut", "max", "min", "multiply"}:
            arguments = self._make_first_pair_distinct(arguments)
        if name in _SAME_UNIT_SIGNAL_OPERATORS:
            start = 1 if name == "if_else" else 0
            for index in range(start, len(arguments)):
                arguments[index] = CallArgument(
                    self._dimensionless(arguments[index].value)
                )
        for index in dimensionless_input_indexes(name):
            if index >= len(arguments):
                continue
            argument = arguments[index]
            arguments[index] = CallArgument(
                self._dimensionless(argument.value),
                name=argument.name,
            )
        if name == "group_mean" and len(arguments) >= 2:
            arguments[1] = CallArgument(self._positive(arguments[1].value))
        if name == "trade_when" and len(arguments) == 3:
            if arguments[0].value == arguments[2].value:
                arguments[2] = CallArgument(
                    Binary("==", arguments[2].value, Literal("0", "integer"))
                )
        return Call(name, tuple(arguments))

    def _argument(
        self,
        operator: OperatorDefinition,
        parameter_name: str,
        kind: str,
        *,
        depth: int,
    ) -> Expression:
        if parameter_name in _CONDITION_INPUTS.get(operator.name, ()):
            return self.condition(depth=depth)
        if kind == "expr":
            return self._required_signal(depth=depth)
        if kind == "group":
            return self.group(depth=depth)
        if kind == "window":
            return _integer(_pick(self.windows, self.random))
        raise ValueError(
            f"structure_parameter_rule_missing:{operator.name}:{parameter_name}"
        )

    def _required_signal(self, *, depth: int) -> Expression:
        expression = self.signal(depth=depth)
        if expression is None:
            return self._signal_atom()
        return expression

    def _signal_atom(self) -> Expression:
        field = _pick_field(self.signal_fields, self.random)
        return _pick(self._field_atoms(field), self.random)

    def _base_condition(self, *, depth: int) -> Binary:
        left, alternative = self._comparison_operands(depth=depth + 1)
        if self.random.random() < 0.6:
            right: Expression = Literal(
                _pick(_SCALAR_THRESHOLDS, self.random),
                "float",
            )
        else:
            right = alternative
        return Binary(_pick(_COMPARISONS, self.random), left, right)

    def _comparison_operands(self, *, depth: int) -> tuple[Expression, Expression]:
        left = self._required_signal(depth=depth)
        right = self._required_signal(depth=depth)
        if right == left:
            right = self._different_atom(left)
        return self._dimensionless(left), self._dimensionless(right)

    def _positive_signal(self, *, depth: int) -> Expression:
        return self._positive(self._required_signal(depth=depth))

    def _absolute_signal(self, *, depth: int) -> Expression:
        return self._absolute(self._dimensionless_signal(depth=depth))

    def _dimensionless_signal(self, *, depth: int) -> Expression:
        return self._dimensionless(self._required_signal(depth=depth))

    def _positive(self, expression: Expression) -> Expression:
        return Call(
            "add",
            (
                CallArgument(self._absolute(self._dimensionless(expression))),
                CallArgument(Literal("0.0001", "float")),
            ),
        )

    def _dimensionless(self, expression: Expression) -> Expression:
        if is_proven_dimensionless(expression, self.catalog):
            return expression
        operator = _pick(self.dimensionless_normalizers, self.random)
        return Call(operator.name, (CallArgument(expression),))

    @staticmethod
    def _absolute(expression: Expression) -> Expression:
        return Call("abs", (CallArgument(expression),))

    def _make_first_pair_distinct(
        self,
        arguments: list[CallArgument],
    ) -> list[CallArgument]:
        if len(arguments) >= 2 and arguments[0].value == arguments[1].value:
            arguments[1] = CallArgument(self._different_atom(arguments[0].value))
        return arguments

    def _different_atom(self, expression: Expression) -> Expression:
        fields = tuple(
            field
            for field in self.signal_fields
            if any(atom != expression for atom in self._field_atoms(field))
        )
        if not fields:
            return Literal("1", "integer")
        field = _pick_field(fields, self.random)
        return _pick(
            tuple(
                atom for atom in self._field_atoms(field) if atom != expression
            ),
            self.random,
        )

    def _field_atoms(self, field: FieldDefinition) -> tuple[Expression, ...]:
        if field.field_type == "MATRIX":
            return (Name(field.field_id),)
        return tuple(
            Call(reducer.name, (CallArgument(Name(field.field_id)),))
            for reducer in self.vector_reducers
        )

    def _can_build_signal_operator(self, operator: OperatorDefinition) -> bool:
        if operator.output_kind != "signal":
            return False
        if operator.name in {"vec_avg", "vec_sum"} and not self.vector_fields:
            return False
        if any(
            not self._operator_enabled(name) or self.catalog.operator(name) is None
            for name in _DOMAIN_DEPENDENCIES.get(operator.name, ())
        ):
            return False
        if (
            (
                operator.name
                in (
                    _COMPARISON_CALLS
                    | _SAME_UNIT_SIGNAL_OPERATORS
                    | {"divide", "group_mean", "trade_when"}
                )
                or dimensionless_input_indexes(operator.name)
            )
            and not self.dimensionless_normalizers
        ):
            return False
        if any(
            parameter.kind == "group" and not self.groups
            for parameter in operator.parameters
            if not parameter.optional
        ):
            return False
        return self._required_parameters_supported(operator)

    def _operator_enabled(self, name: str) -> bool:
        return self.allowed_operators is None or name in self.allowed_operators

    @staticmethod
    def _required_parameters_supported(operator: OperatorDefinition) -> bool:
        for parameter in operator.parameters:
            if parameter.optional:
                continue
            if parameter.kind in {"expr", "group", "window"}:
                continue
            if operator.name in {"kth_element", "ts_step"} and parameter.kind == "int":
                continue
            return False
        return True


def _integer(value: int) -> Literal:
    return Literal(str(value), "integer")


def _pick(items: Sequence[_T], random: Random) -> _T:
    if len(items) == 1:
        return items[0]
    return items[random.randrange(len(items))]


def _pick_field(
    fields: Sequence[FieldDefinition],
    random: Random,
) -> FieldDefinition:
    by_category: dict[str, list[FieldDefinition]] = defaultdict(list)
    for field in fields:
        by_category[field.category or ""].append(field)
    category = _pick(tuple(sorted(by_category)), random)

    by_dataset: dict[str, list[FieldDefinition]] = defaultdict(list)
    for field in by_category[category]:
        by_dataset[field.dataset_id or ""].append(field)
    dataset = _pick(tuple(sorted(by_dataset)), random)
    return _pick(
        tuple(sorted(by_dataset[dataset], key=lambda field: field.field_id)),
        random,
    )
