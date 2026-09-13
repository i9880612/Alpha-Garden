from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CatalogContext:
    instrument_type: str
    region: str
    universe: str
    delay: int

    def __post_init__(self) -> None:
        for value in (self.instrument_type, self.region, self.universe):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("worldquant_catalog_context_invalid")
        if (
            isinstance(self.delay, bool)
            or not isinstance(self.delay, int)
            or self.delay < 0
        ):
            raise ValueError("worldquant_catalog_context_invalid")


@dataclass(frozen=True, slots=True)
class DataField:
    field_id: str
    dataset_id: str | None
    dataset_name: str | None
    category: str | None
    category_id: str | None
    subcategory: str | None
    subcategory_id: str | None
    field_type: str | None
    coverage: float | None
    description: str | None
    raw_payload: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class DataFieldPage:
    total_count: int
    fields: tuple[DataField, ...]


@dataclass(frozen=True, slots=True)
class Operator:
    name: str
    category: str | None
    definition: str | None
    description: str | None
    documentation: str | None
    level: str | None
    scope: tuple[str, ...]
    parameters: tuple[Mapping[str, Any], ...]
    raw_payload: Mapping[str, Any]


class WorldQuantCatalogProtocolError(ValueError):
    pass


def parse_data_field_page(
    payload: object,
    *,
    expected_context: CatalogContext,
) -> DataFieldPage:
    if not isinstance(payload, Mapping):
        raise WorldQuantCatalogProtocolError(
            "worldquant_data_fields_payload_invalid"
        )
    total_count = payload.get("count")
    results = payload.get("results")
    if (
        isinstance(total_count, bool)
        or not isinstance(total_count, int)
        or total_count <= 0
    ):
        raise WorldQuantCatalogProtocolError(
            "worldquant_data_fields_total_invalid"
        )
    if not isinstance(results, list):
        raise WorldQuantCatalogProtocolError(
            "worldquant_data_fields_results_invalid"
        )
    fields = tuple(
        _parse_data_field(item, expected_context=expected_context)
        for item in results
    )
    identifiers = [field.field_id for field in fields]
    if len(set(identifiers)) != len(identifiers):
        raise WorldQuantCatalogProtocolError(
            "worldquant_data_fields_duplicate_id"
        )
    return DataFieldPage(total_count=total_count, fields=fields)


def parse_operators(payload: object) -> tuple[Operator, ...]:
    if isinstance(payload, Mapping):
        results = payload.get("results")
    else:
        results = payload
    if not isinstance(results, list) or not results:
        raise WorldQuantCatalogProtocolError(
            "worldquant_operators_payload_invalid"
        )
    operators = tuple(_parse_operator(item) for item in results)
    names = [operator.name for operator in operators]
    if len(set(names)) != len(names):
        raise WorldQuantCatalogProtocolError(
            "worldquant_operators_duplicate_name"
        )
    return operators


def _parse_data_field(
    payload: object,
    *,
    expected_context: CatalogContext,
) -> DataField:
    if not isinstance(payload, Mapping):
        raise WorldQuantCatalogProtocolError(
            "worldquant_data_field_payload_invalid"
        )
    field_id = payload.get("id")
    if not isinstance(field_id, str) or not field_id.strip():
        raise WorldQuantCatalogProtocolError("worldquant_data_field_id_invalid")
    if (
        payload.get("region") != expected_context.region
        or payload.get("universe") != expected_context.universe
        or payload.get("delay") != expected_context.delay
    ):
        raise WorldQuantCatalogProtocolError(
            "worldquant_data_field_context_mismatch"
        )
    dataset_id, dataset_name = _nested_text_pair(payload, "dataset")
    category_id, category_name = _nested_text_pair(payload, "category")
    subcategory_id, subcategory_name = _nested_text_pair(payload, "subcategory")
    coverage = _optional_number(
        payload.get("coverage"),
        "worldquant_data_field_coverage_invalid",
    )
    return DataField(
        field_id=field_id.strip(),
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        category=category_name,
        category_id=category_id,
        subcategory=subcategory_name,
        subcategory_id=subcategory_id,
        field_type=_optional_text(
            payload.get("type"),
            "worldquant_data_field_type_invalid",
        ),
        coverage=coverage,
        description=_optional_text(
            payload.get("description"),
            "worldquant_data_field_description_invalid",
        ),
        raw_payload=dict(payload),
    )


def _parse_operator(payload: object) -> Operator:
    if not isinstance(payload, Mapping):
        raise WorldQuantCatalogProtocolError(
            "worldquant_operator_payload_invalid"
        )
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise WorldQuantCatalogProtocolError("worldquant_operator_name_invalid")
    scope = payload.get("scope")
    if scope is None:
        normalized_scope: tuple[str, ...] = ()
    elif isinstance(scope, list) and all(
        isinstance(item, str) and item.strip() for item in scope
    ):
        normalized_scope = tuple(item.strip() for item in scope)
    else:
        raise WorldQuantCatalogProtocolError("worldquant_operator_scope_invalid")
    definition = _optional_text(
        payload.get("definition"),
        "worldquant_operator_definition_invalid",
    )
    raw_parameters = payload.get("parameters")
    if raw_parameters is None:
        parameters = _operator_parameters_from_definition(
            operator_name=name.strip(),
            definition=definition,
        )
    else:
        if not isinstance(raw_parameters, list) or any(
            not isinstance(item, Mapping) for item in raw_parameters
        ):
            raise WorldQuantCatalogProtocolError(
                "worldquant_operator_parameters_invalid"
            )
        parameters = tuple(dict(item) for item in raw_parameters)
        if not parameters:
            parameters = _operator_parameters_from_definition(
                operator_name=name.strip(),
                definition=definition,
            )
    return Operator(
        name=name.strip(),
        category=_optional_text(
            payload.get("category"),
            "worldquant_operator_category_invalid",
        ),
        definition=definition,
        description=_optional_text(
            payload.get("description"),
            "worldquant_operator_description_invalid",
        ),
        documentation=_optional_text(
            payload.get("documentation"),
            "worldquant_operator_documentation_invalid",
        ),
        level=_optional_text(
            payload.get("level"),
            "worldquant_operator_level_invalid",
        ),
        scope=normalized_scope,
        parameters=parameters,
        raw_payload=dict(payload),
    )


def _nested_text_pair(
    payload: Mapping[str, object],
    key: str,
) -> tuple[str | None, str | None]:
    value = payload.get(key)
    if value is None:
        return None, None
    if not isinstance(value, Mapping):
        raise WorldQuantCatalogProtocolError(
            f"worldquant_data_field_{key}_invalid"
        )
    return (
        _optional_text(
            value.get("id"),
            f"worldquant_data_field_{key}_id_invalid",
        ),
        _optional_text(
            value.get("name"),
            f"worldquant_data_field_{key}_name_invalid",
        ),
    )


def _optional_text(value: object, error: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise WorldQuantCatalogProtocolError(error)
    return value.strip() or None


def _optional_number(value: object, error: str) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise WorldQuantCatalogProtocolError(error)
    return float(value)


def _operator_parameters_from_definition(
    *,
    operator_name: str,
    definition: str | None,
) -> tuple[Mapping[str, Any], ...]:
    if not definition:
        return ()
    normalized = definition.replace("“", '"').replace("”", '"')
    signature = _first_operator_signature(
        operator_name=operator_name,
        definition=normalized,
    )
    if signature is None:
        return _expression_operator_parameters(normalized)
    return _signature_parameters(signature)


def _first_operator_signature(
    *,
    operator_name: str,
    definition: str,
) -> str | None:
    marker = f"{operator_name}("
    start = definition.find(marker)
    if start < 0:
        return None
    inner_start = start + len(marker)
    depth = 1
    quote: str | None = None
    for index in range(inner_start, len(definition)):
        character = definition[index]
        if quote is not None:
            if character == quote:
                quote = None
            continue
        if character in {'"', "'"}:
            quote = character
        elif character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return definition[inner_start:index]
    return None


def _signature_parameters(signature: str) -> tuple[Mapping[str, Any], ...]:
    parameters: list[dict[str, Any]] = []
    for raw_item in _split_signature_arguments(signature):
        item = _normalize_signature_parameter(raw_item)
        if not item:
            continue
        if item in {"..", "..."}:
            _mark_previous_variadic(parameters)
            continue
        variadic = False
        for suffix in ("...", ".."):
            if item.endswith(suffix):
                item = item[: -len(suffix)].strip()
                variadic = True
                break
        if "=" in item:
            raw_name, raw_default = item.split("=", 1)
            default = raw_default.strip()
        else:
            raw_name = item
            default = None
        parameter_name = _normalize_parameter_name(raw_name)
        if parameter_name is None:
            if variadic:
                _mark_previous_variadic(parameters)
            continue
        parameter: dict[str, Any] = {
            "name": parameter_name,
            "kind": _infer_parameter_kind(parameter_name, default),
        }
        if default is not None:
            parameter["optional"] = True
        if variadic:
            parameter["variadic"] = True
        parameters.append(parameter)
    return tuple(parameters)


def _split_signature_arguments(signature: str) -> tuple[str, ...]:
    arguments: list[str] = []
    depth = 0
    quote: str | None = None
    start = 0
    for index, character in enumerate(signature):
        if quote is not None:
            if character == quote:
                quote = None
            continue
        if character in {'"', "'"}:
            quote = character
        elif character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        elif character == "," and depth == 0:
            arguments.append(signature[start:index])
            start = index + 1
    arguments.append(signature[start:])
    return tuple(arguments)


def _normalize_signature_parameter(raw_item: str) -> str:
    item = raw_item.strip().replace("“", '"').replace("”", '"')
    if not item:
        return ""
    wrapped_call = re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)",
        item,
    )
    return wrapped_call.group(1) if wrapped_call else item


def _normalize_parameter_name(raw_name: str) -> str | None:
    name = raw_name.strip().replace(" ", "")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        return None
    return name


def _mark_previous_variadic(parameters: list[dict[str, Any]]) -> None:
    if parameters:
        parameters[-1]["variadic"] = True


def _expression_operator_parameters(
    definition: str,
) -> tuple[Mapping[str, Any], ...]:
    names = tuple(dict.fromkeys(re.findall(r"\binput\s*\d+\b", definition)))
    return tuple(
        {"name": name.replace(" ", ""), "kind": "expr"}
        for name in names
    )


def _infer_parameter_kind(name: str, default: str | None) -> str:
    normalized = name.lower()
    default_text = (default or "").strip().strip('"').strip("'").lower()
    if normalized in {"d", "lookback", "window"}:
        return "window"
    if normalized == "group":
        return "group"
    if normalized in {"filter", "dense", "usestd", "skipboth", "nangroup"}:
        return "bool"
    if default_text in {"true", "false"}:
        return "bool"
    if normalized in {"k", "lag", "rettype"}:
        return "int"
    if normalized in {"driver", "ignore", "range", "buckets"}:
        return "string"
    if default_text and _is_number(default_text):
        return "float"
    return "expr"


def _is_number(value: str) -> bool:
    try:
        parsed = float(value)
    except ValueError:
        return False
    return math.isfinite(parsed)
