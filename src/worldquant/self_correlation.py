from __future__ import annotations

import math
from collections.abc import Mapping


def maximum_self_correlation_reference(payload: object) -> str | None:
    """Read a reported maximum peer; unavailable detail is not an SC result."""
    metrics = payload.get("is") if isinstance(payload, Mapping) else None
    detail = metrics.get("selfCorrelated") if isinstance(metrics, Mapping) else None
    if not isinstance(detail, Mapping):
        return None
    maximum = detail.get("max")
    if not _correlation(maximum) or maximum <= 0:
        return None
    checks = metrics.get("checks")
    if not isinstance(checks, list):
        return None
    sc_checks = [
        item
        for item in checks
        if isinstance(item, Mapping) and item.get("name") == "SELF_CORRELATION"
    ]
    if len(sc_checks) != 1:
        return None
    value, limit = sc_checks[0].get("value"), sc_checks[0].get("limit")
    if (
        not _correlation(value)
        or not _correlation(limit)
        or value != maximum
        or value <= limit
    ):
        return None
    schema = detail.get("schema")
    properties = schema.get("properties") if isinstance(schema, Mapping) else None
    records = detail.get("records")
    if not isinstance(properties, list) or not isinstance(records, list):
        return None
    names = [
        item.get("name") if isinstance(item, Mapping) else None for item in properties
    ]
    if names.count("id") != 1 or names.count("correlation") != 1:
        return None
    id_index = names.index("id")
    correlation_index = names.index("correlation")
    references: set[str] = set()
    for row in records:
        if not isinstance(row, list) or len(row) != len(names):
            return None
        reference, correlation = row[id_index], row[correlation_index]
        if (
            not isinstance(reference, str)
            or not reference.strip()
            or not _correlation(correlation)
            or correlation > maximum
        ):
            return None
        if correlation == maximum:
            references.add(reference)
    # Deterministic tie breaking is not a quality preference.
    return min(references) if references else None


def _correlation(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and -1 <= value <= 1
    )
