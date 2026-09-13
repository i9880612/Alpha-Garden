from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from types import MappingProxyType
from typing import Mapping


_SUPPORTED_PARAMETER_KINDS = frozenset(
    {"bool", "expr", "float", "group", "int", "string", "window"}
)
_SUPPORTED_OUTPUT_KINDS = frozenset({"condition", "group", "signal"})
TIME_SERIES_SIGNAL_ROLES = frozenset(
    {
        "time_series_aggregation",
        "time_series_change",
        "time_series_dispersion",
        "time_series_event_age",
        "time_series_lag",
        "time_series_normalization",
        "time_series_smoothing",
    }
)
ADDITION_SAFE_TIME_SERIES_ROLES = frozenset({"time_series_normalization"})
CROSS_SECTIONAL_SIGNAL_ROLES = frozenset(
    {
        "cross_sectional_normalization",
        "cross_sectional_outlier_control",
        "cross_sectional_scaling",
    }
)


@dataclass(frozen=True, slots=True)
class CatalogContext:
    instrument_type: str
    region: str
    universe: str
    delay: int


@dataclass(frozen=True, slots=True)
class FieldDefinition:
    field_id: str
    dataset_id: str | None
    category: str | None
    subcategory: str | None
    field_type: str | None
    coverage: float | None


@dataclass(frozen=True, slots=True)
class OperatorParameter:
    name: str
    kind: str
    optional: bool = False
    variadic: bool = False


@dataclass(frozen=True, slots=True)
class OperatorDefinition:
    name: str
    category: str | None
    scope: tuple[str, ...]
    parameters: tuple[OperatorParameter, ...]
    roles: tuple[str, ...] = ()
    output_kind: str | None = None

    @property
    def uses_window(self) -> bool:
        return any(parameter.kind == "window" for parameter in self.parameters)

    def has_any_role(self, roles: frozenset[str]) -> bool:
        return bool(roles.intersection(self.roles))


@dataclass(frozen=True, slots=True)
class WindowDefinition:
    value: int
    horizon: str


@dataclass(frozen=True, slots=True)
class GenerationCatalog:
    context: CatalogContext
    fields: tuple[FieldDefinition, ...]
    operators: tuple[OperatorDefinition, ...]
    windows: tuple[WindowDefinition, ...] = ()
    fingerprint: str = field(init=False)
    _fields_by_id: Mapping[str, FieldDefinition] = field(init=False, repr=False)
    _operators_by_name: Mapping[str, OperatorDefinition] = field(
        init=False,
        repr=False,
    )
    _windows_by_value: Mapping[int, WindowDefinition] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        _validate_context(self.context)
        fields = tuple(sorted(self.fields, key=lambda item: item.field_id))
        operators = tuple(sorted(self.operators, key=lambda item: item.name))
        windows = tuple(sorted(self.windows, key=lambda item: item.value))
        _validate_fields(fields)
        _validate_operators(operators)
        _validate_windows(windows)
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "operators", operators)
        object.__setattr__(self, "windows", windows)
        object.__setattr__(
            self,
            "_fields_by_id",
            MappingProxyType({item.field_id: item for item in fields}),
        )
        object.__setattr__(
            self,
            "_operators_by_name",
            MappingProxyType({item.name: item for item in operators}),
        )
        object.__setattr__(
            self,
            "_windows_by_value",
            MappingProxyType({item.value: item for item in windows}),
        )
        object.__setattr__(self, "fingerprint", _catalog_fingerprint(self))

    def field(self, field_id: str) -> FieldDefinition | None:
        return self._fields_by_id.get(field_id)

    def operator(self, name: str) -> OperatorDefinition | None:
        return self._operators_by_name.get(name)

    def window(self, value: int) -> WindowDefinition | None:
        return self._windows_by_value.get(value)

    def window_values(self, horizon: str | None = None) -> tuple[int, ...]:
        return tuple(
            item.value
            for item in self.windows
            if horizon is None or item.horizon == horizon
        )


def _validate_context(context: CatalogContext) -> None:
    for value in (context.instrument_type, context.region, context.universe):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("generation_catalog_context_invalid")
    if (
        isinstance(context.delay, bool)
        or not isinstance(context.delay, int)
        or context.delay < 0
    ):
        raise ValueError("generation_catalog_context_invalid")


def _validate_fields(fields: tuple[FieldDefinition, ...]) -> None:
    if not fields:
        raise ValueError("generation_field_catalog_empty")
    identifiers = [item.field_id for item in fields]
    if any(not isinstance(item, str) or not item.strip() for item in identifiers):
        raise ValueError("generation_field_id_invalid")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("generation_field_id_duplicate")


def _validate_operators(operators: tuple[OperatorDefinition, ...]) -> None:
    if not operators:
        raise ValueError("generation_operator_catalog_empty")
    names = [item.name for item in operators]
    if any(not isinstance(item, str) or not item.strip() for item in names):
        raise ValueError("generation_operator_name_invalid")
    if len(set(names)) != len(names):
        raise ValueError("generation_operator_name_duplicate")
    for operator in operators:
        if not isinstance(operator.scope, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in operator.scope
        ):
            raise ValueError("generation_operator_scope_invalid")
        if not isinstance(operator.parameters, tuple):
            raise ValueError("generation_operator_parameters_invalid")
        if (
            not isinstance(operator.roles, tuple)
            or any(
                not isinstance(role, str) or not role.strip()
                for role in operator.roles
            )
            or len(set(operator.roles)) != len(operator.roles)
        ):
            raise ValueError("generation_operator_roles_invalid")
        if (
            operator.output_kind is not None
            and operator.output_kind not in _SUPPORTED_OUTPUT_KINDS
        ):
            raise ValueError(
                f"generation_operator_output_kind_unsupported:{operator.name}"
            )
        parameter_names = [item.name for item in operator.parameters]
        if any(
            not isinstance(parameter.name, str)
            or not parameter.name.strip()
            or not isinstance(parameter.kind, str)
            or not parameter.kind.strip()
            or not isinstance(parameter.optional, bool)
            or not isinstance(parameter.variadic, bool)
            for parameter in operator.parameters
        ):
            raise ValueError("generation_operator_parameter_invalid")
        unsupported_kinds = {
            item.kind
            for item in operator.parameters
            if item.kind not in _SUPPORTED_PARAMETER_KINDS
        }
        if unsupported_kinds:
            raise ValueError(
                "generation_operator_parameter_kind_unsupported:"
                + ",".join(sorted(unsupported_kinds))
            )
        if len(set(parameter_names)) != len(parameter_names):
            raise ValueError("generation_operator_parameter_duplicate")
        variadic_indexes = [
            index
            for index, parameter in enumerate(operator.parameters)
            if parameter.variadic
        ]
        if len(variadic_indexes) > 1:
            raise ValueError("generation_operator_variadic_parameter_invalid")


def _validate_windows(windows: tuple[WindowDefinition, ...]) -> None:
    values = [item.value for item in windows]
    if any(
        isinstance(item.value, bool)
        or not isinstance(item.value, int)
        or item.value <= 0
        or not isinstance(item.horizon, str)
        or not item.horizon.strip()
        for item in windows
    ):
        raise ValueError("generation_window_invalid")
    if len(set(values)) != len(values):
        raise ValueError("generation_window_value_duplicate")


def _catalog_fingerprint(catalog: GenerationCatalog) -> str:
    payload = {
        "context": {
            "instrument_type": catalog.context.instrument_type,
            "region": catalog.context.region,
            "universe": catalog.context.universe,
            "delay": catalog.context.delay,
        },
        "fields": [
            {
                "field_id": item.field_id,
                "dataset_id": item.dataset_id,
                "category": item.category,
                "subcategory": item.subcategory,
                "field_type": item.field_type,
                "coverage": item.coverage,
            }
            for item in catalog.fields
        ],
        "operators": [
            {
                "name": item.name,
                "category": item.category,
                "scope": list(item.scope),
                "roles": list(item.roles),
                "output_kind": item.output_kind,
                "parameters": [
                    {
                        "name": parameter.name,
                        "kind": parameter.kind,
                        "optional": parameter.optional,
                        "variadic": parameter.variadic,
                    }
                    for parameter in item.parameters
                ],
            }
            for item in catalog.operators
        ],
        "windows": [
            {"value": item.value, "horizon": item.horizon}
            for item in catalog.windows
        ],
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(canonical.encode("utf-8")).hexdigest()
