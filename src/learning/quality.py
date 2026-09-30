"""Quality objectives use platform grades and metrics, independently of failed checks."""
from __future__ import annotations

import math
from learning.evidence import LearningEvidenceRecord
from worldquant.backtests import STANDARD_NON_SC_CHECK_NAMES


QUALITY_IMPROVEMENT = "QUALITY_IMPROVEMENT"
QUALITY_LINEAGE_ATTEMPT_BUDGET = 80
QUALITY_GRADES = ("INFERIOR", "AVERAGE", "GOOD", "EXCELLENT", "SPECTACULAR")


def quality_metrics_ready(record: LearningEvidenceRecord) -> bool:
    if not (record.grade in QUALITY_GRADES and record.non_sc_check_set_complete
            and not record.unexpected_checks and STANDARD_NON_SC_CHECK_NAMES <= set(record.passed_checks)
            and all(math.isfinite(value) for value in (record.sharpe, record.fitness, record.turnover))):
        return False
    for check in record.checks:
        if check.name == "LOW_TURNOVER" and check.threshold is not None and record.turnover < check.threshold:
            return False
        if (check.name == "LOW_SUB_UNIVERSE_SHARPE" and check.threshold is not None
                and check.actual is not None and check.actual < check.threshold):
            return False
    return True


def quality_progress(parent: LearningEvidenceRecord, child: LearningEvidenceRecord) -> str:
    """Small Pareto improvements teach preferences; they never grant attempts."""
    if not quality_metrics_ready(child):
        return "conflict" if child.failed_checks else "unresolved"
    if parent.grade not in QUALITY_GRADES:
        return "unresolved"
    before, after = QUALITY_GRADES.index(parent.grade), QUALITY_GRADES.index(child.grade)
    if after < before:
        return "conflict"
    if after > before:
        return "safe_progress"
    if child.sharpe < parent.sharpe - 1e-12 or child.fitness < parent.fitness - 1e-12:
        return "conflict"
    return ("safe_progress" if child.sharpe > parent.sharpe + 1e-12
            or child.fitness > parent.fitness + 1e-12 else "no_progress")
