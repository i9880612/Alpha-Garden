"""Local submission screening; the platform still owns the final check."""
from collections.abc import Mapping
from decimal import Decimal
import math


CORRELATION_CUTOFF = 0.7
MIN_CORRELATION_INTERVALS = 252


def sharpe_improves(candidate: float | None, reference: float | None) -> bool | None:
    """Use reported precision, including the exact 10% boundary, without rounding."""
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value <= 0 for value in (candidate, reference)):
        return None
    return Decimal(str(candidate)) >= Decimal(str(reference)) * Decimal("1.10")


def correlation_check(correlation: float | None, candidate_sharpe: float | None,
                      reference_sharpe: float | None) -> str:
    if correlation is None:
        return "pending"
    if correlation < CORRELATION_CUTOFF:
        return "passed"
    improves = sharpe_improves(candidate_sharpe, reference_sharpe)
    return "pending" if improves is None else "passed" if improves else "failed"


def submitted_sharpe(payload: Mapping) -> float | None:
    metrics = payload.get("is")
    value = metrics.get("sharpe") if isinstance(metrics, Mapping) else None
    return value if type(value) in (float, int) and math.isfinite(value) else None
