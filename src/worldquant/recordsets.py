"""Chart recordsets preserve gaps and the platform's original value units."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

RECORDSET_UNITS = {"pnl": "amount", "sharpe": "decimal", "turnover": "percent"}


@dataclass(frozen=True)
class RecordsetObservation:
    points: tuple[tuple[str, float | None], ...] | None
    retry_after_seconds: float | None = None


def parse_recordset(payload, metric):
    if metric not in RECORDSET_UNITS:
        raise ValueError("worldquant_recordset_metric_invalid")
    if not isinstance(payload, dict) or not isinstance(payload.get("schema"), dict):
        raise ValueError("worldquant_recordset_schema_invalid")
    properties, records = payload["schema"].get("properties"), payload.get("records")
    if not isinstance(properties, list) or not isinstance(records, list):
        raise ValueError("worldquant_recordset_schema_invalid")
    names = [item.get("name") if isinstance(item, dict) else None for item in properties]
    if names.count("date") != 1 or names.count(metric) != 1:
        raise ValueError("worldquant_recordset_columns_invalid")
    date_index, value_index = names.index("date"), names.index(metric)
    if properties[date_index].get("type") != "date" or properties[value_index].get("type") != RECORDSET_UNITS[metric]:
        raise ValueError("worldquant_recordset_unit_invalid")
    points = []
    for record in records:
        if not isinstance(record, list) or len(record) != len(names):
            raise ValueError("worldquant_recordset_row_invalid")
        day, value = record[date_index], record[value_index]
        if not isinstance(day, str) or date.fromisoformat(day).isoformat() != day:
            raise ValueError("worldquant_recordset_date_invalid")
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError("worldquant_recordset_value_invalid")
        if points and day <= points[-1][0]:
            raise ValueError("worldquant_recordset_dates_not_increasing")
        points.append((day, float(value) if value is not None else None))
    return tuple(points)
