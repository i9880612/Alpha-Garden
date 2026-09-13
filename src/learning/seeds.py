from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from evaluation.backtests import evaluate_backtest
from persistence.backtests import BacktestSnapshot

if TYPE_CHECKING:
    from learning.evidence import LearningEvidenceRecord


SIGNAL_TARGET_CHECKS = frozenset(
    {
        "LOW_SHARPE",
        "LOW_FITNESS",
        "LOW_SUB_UNIVERSE_SHARPE",
    }
)
SIGNAL_REQUIRED_PASS_CHECKS = frozenset(
    {
        "CONCENTRATED_WEIGHT",
    }
)
SIGNAL_ALLOWED_RISK_CHECKS = frozenset({"SELF_CORRELATION"})
SIGNAL_MIN_SHARPE = 1.0
SIGNAL_MIN_FITNESS = 0.7
SIGNAL_MIN_TURNOVER = 0.01
SIGNAL_MAX_TURNOVER = 0.70


@dataclass(frozen=True, slots=True)
class SignalSeedAssessment:
    eligible: bool
    self_correlation_status: str
    rejection_reasons: tuple[str, ...]


def assess_signal_seed(snapshot: BacktestSnapshot) -> SignalSeedAssessment:
    evaluation = evaluate_backtest(snapshot)
    assert snapshot.result is not None
    reasons: list[str] = []
    if snapshot.result.fitness < SIGNAL_MIN_FITNESS:
        reasons.append("signal_seed_fitness_below_positive_threshold")
    if snapshot.result.sharpe < SIGNAL_MIN_SHARPE:
        reasons.append("signal_seed_sharpe_below_positive_threshold")
    if not signal_turnover_is_eligible(snapshot.result.turnover):
        reasons.append("signal_seed_turnover_out_of_range")
    if not evaluation.non_sc_check_set_complete:
        reasons.append("signal_seed_check_set_incomplete")

    passed = set(evaluation.passed_checks)
    failed = set(evaluation.failed_checks)
    pending = set(evaluation.pending_checks)
    observed_signal_checks = passed | failed
    for check_name in sorted(SIGNAL_TARGET_CHECKS - observed_signal_checks):
        reasons.append(f"signal_seed_check_not_resolved:{check_name}")
    for check_name in sorted(SIGNAL_REQUIRED_PASS_CHECKS - passed):
        reasons.append(f"signal_seed_check_not_passed:{check_name}")
    for check_name in sorted(
        (failed | pending)
        - SIGNAL_ALLOWED_RISK_CHECKS
        - SIGNAL_TARGET_CHECKS
        - SIGNAL_REQUIRED_PASS_CHECKS
    ):
        reasons.append(f"signal_seed_other_check_not_passed:{check_name}")
    if "SELF_CORRELATION" in passed:
        self_correlation_status = "passed"
    elif "SELF_CORRELATION" in failed:
        self_correlation_status = "failed"
    elif "SELF_CORRELATION" in pending:
        self_correlation_status = "pending"
    else:
        self_correlation_status = "missing"

    return SignalSeedAssessment(
        eligible=not reasons,
        self_correlation_status=self_correlation_status,
        rejection_reasons=tuple(reasons),
    )


def signal_turnover_is_eligible(turnover: float) -> bool:
    return SIGNAL_MIN_TURNOVER < turnover < SIGNAL_MAX_TURNOVER


def signal_branch_is_safe(record: LearningEvidenceRecord) -> bool:
    passed = set(record.passed_checks)
    failed = set(record.failed_checks)
    pending = set(record.pending_checks)
    if record.sharpe < SIGNAL_MIN_SHARPE or record.fitness < SIGNAL_MIN_FITNESS:
        return False
    if not signal_turnover_is_eligible(record.turnover):
        return False
    if not record.non_sc_check_set_complete:
        return False
    if not SIGNAL_REQUIRED_PASS_CHECKS <= passed:
        return False
    if not SIGNAL_TARGET_CHECKS <= (passed | failed):
        return False
    allowed_unresolved = SIGNAL_TARGET_CHECKS | SIGNAL_ALLOWED_RISK_CHECKS
    return not ((failed | pending) - allowed_unresolved)
