from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class PnlObservation:
    points: tuple[tuple[str, float], ...] | None
    retry_after_seconds: float | None = None


def parse_pnl(payload: object) -> tuple[tuple[str, float], ...]:
    """Parse cumulative daily PnL; never fill gaps or invent a zero baseline."""
    if not isinstance(payload, dict):
        raise ValueError("worldquant_pnl_payload_invalid")
    schema = payload.get("schema")
    properties = schema.get("properties") if isinstance(schema, dict) else None
    records = payload.get("records")
    if not isinstance(properties, list) or not isinstance(records, list):
        raise ValueError("worldquant_pnl_schema_invalid")
    names = [
        item.get("name") if isinstance(item, dict) else None for item in properties
    ]
    if names.count("date") != 1 or names.count("pnl") != 1:
        raise ValueError("worldquant_pnl_columns_invalid")
    points = []
    for row in records:
        if not isinstance(row, list) or len(row) != len(names):
            raise ValueError("worldquant_pnl_row_invalid")
        day, value = row[names.index("date")], row[names.index("pnl")]
        if (
            not isinstance(day, str)
            or date.fromisoformat(day).isoformat() != day
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("worldquant_pnl_point_invalid")
        if points and day <= points[-1][0]:
            raise ValueError("worldquant_pnl_dates_not_increasing")
        points.append((day, float(value)))
    return tuple(points)
