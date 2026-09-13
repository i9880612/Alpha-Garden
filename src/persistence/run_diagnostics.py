from __future__ import annotations

import json
from dataclasses import dataclass


CANDIDATE_PLANNING_STOP_REASONS = frozenset(
    {
        "generation_attempt_budget_exhausted",
        "selection_rejection_shortfall",
    }
)


@dataclass(frozen=True, slots=True)
class CandidatePlanningExclusionCount:
    code: str
    count: int


@dataclass(frozen=True, slots=True)
class CandidatePlanningDiagnostic:
    cycle_number: int
    exploration_generation_target_count: int
    exploration_backtest_target_count: int
    seed_attempt_limit: int
    attempted_seed_count: int
    generated_candidate_count: int
    selected_candidate_count: int | None
    generation_exclusions: tuple[CandidatePlanningExclusionCount, ...]
    selection_rejections: tuple[CandidatePlanningExclusionCount, ...]

    def canonical_json(self, *, stop_reason: str) -> str:
        _validate_diagnostic(self, stop_reason=stop_reason)
        return json.dumps(
            {
                "attemptedSeedCount": self.attempted_seed_count,
                "cycleNumber": self.cycle_number,
                "explorationBacktestTargetCount": (
                    self.exploration_backtest_target_count
                ),
                "explorationGenerationTargetCount": (
                    self.exploration_generation_target_count
                ),
                "generatedCandidateCount": self.generated_candidate_count,
                "generationExclusionCounts": {
                    item.code: item.count for item in self.generation_exclusions
                },
                "seedAttemptLimit": self.seed_attempt_limit,
                "selectedCandidateCount": self.selected_candidate_count,
                "selectionRejectionCounts": {
                    item.code: item.count for item in self.selection_rejections
                },
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @classmethod
    def from_canonical_json(
        cls,
        value: str,
        *,
        stop_reason: str,
    ) -> CandidatePlanningDiagnostic:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("candidate_planning_diagnostic_invalid")
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("candidate_planning_diagnostic_invalid") from exc
        if not isinstance(payload, dict) or set(payload) != {
            "attemptedSeedCount",
            "cycleNumber",
            "explorationBacktestTargetCount",
            "explorationGenerationTargetCount",
            "generatedCandidateCount",
            "generationExclusionCounts",
            "seedAttemptLimit",
            "selectedCandidateCount",
            "selectionRejectionCounts",
        }:
            raise ValueError("candidate_planning_diagnostic_invalid")
        diagnostic = cls(
            cycle_number=payload["cycleNumber"],
            exploration_generation_target_count=payload[
                "explorationGenerationTargetCount"
            ],
            exploration_backtest_target_count=payload[
                "explorationBacktestTargetCount"
            ],
            seed_attempt_limit=payload["seedAttemptLimit"],
            attempted_seed_count=payload["attemptedSeedCount"],
            generated_candidate_count=payload["generatedCandidateCount"],
            selected_candidate_count=payload["selectedCandidateCount"],
            generation_exclusions=_exclusion_counts(
                payload["generationExclusionCounts"]
            ),
            selection_rejections=_exclusion_counts(
                payload["selectionRejectionCounts"]
            ),
        )
        if diagnostic.canonical_json(stop_reason=stop_reason) != value:
            raise ValueError("candidate_planning_diagnostic_invalid")
        return diagnostic


def _validate_diagnostic(
    diagnostic: CandidatePlanningDiagnostic,
    *,
    stop_reason: str,
) -> None:
    if stop_reason not in CANDIDATE_PLANNING_STOP_REASONS:
        raise ValueError("candidate_planning_stop_reason_invalid")
    _positive_integer(diagnostic.cycle_number)
    _positive_integer(diagnostic.exploration_generation_target_count)
    _positive_integer(diagnostic.exploration_backtest_target_count)
    _positive_integer(diagnostic.seed_attempt_limit)
    _non_negative_integer(diagnostic.attempted_seed_count)
    _non_negative_integer(diagnostic.generated_candidate_count)
    if (
        diagnostic.exploration_backtest_target_count
        > diagnostic.exploration_generation_target_count
        or diagnostic.seed_attempt_limit
        < diagnostic.exploration_generation_target_count
        or diagnostic.attempted_seed_count > diagnostic.seed_attempt_limit
        or diagnostic.generated_candidate_count
        > diagnostic.exploration_generation_target_count
        or diagnostic.generated_candidate_count > diagnostic.attempted_seed_count
    ):
        raise ValueError("candidate_planning_diagnostic_count_invalid")
    _validate_exclusion_counts(diagnostic.generation_exclusions)
    _validate_exclusion_counts(diagnostic.selection_rejections)
    if sum(item.count for item in diagnostic.generation_exclusions) != (
        diagnostic.attempted_seed_count - diagnostic.generated_candidate_count
    ):
        raise ValueError("candidate_planning_generation_exclusions_invalid")

    if stop_reason == "generation_attempt_budget_exhausted":
        if (
            diagnostic.attempted_seed_count != diagnostic.seed_attempt_limit
            or diagnostic.generated_candidate_count
            >= diagnostic.exploration_generation_target_count
            or diagnostic.selected_candidate_count is not None
            or diagnostic.selection_rejections
        ):
            raise ValueError("candidate_planning_generation_stop_invalid")
        return

    selected_count = diagnostic.selected_candidate_count
    if (
        isinstance(selected_count, bool)
        or not isinstance(selected_count, int)
        or selected_count < 0
        or selected_count >= diagnostic.exploration_backtest_target_count
        or diagnostic.generated_candidate_count
        != diagnostic.exploration_generation_target_count
    ):
        raise ValueError("candidate_planning_selection_stop_invalid")
    rejected_candidate_count = diagnostic.generated_candidate_count - selected_count
    if (
        not diagnostic.selection_rejections
        or sum(item.count for item in diagnostic.selection_rejections)
        < rejected_candidate_count
        or any(
            item.count > rejected_candidate_count
            for item in diagnostic.selection_rejections
        )
    ):
        raise ValueError("candidate_planning_selection_rejections_invalid")


def _exclusion_counts(value: object) -> tuple[CandidatePlanningExclusionCount, ...]:
    if not isinstance(value, dict):
        raise ValueError("candidate_planning_exclusions_invalid")
    return tuple(
        CandidatePlanningExclusionCount(code=code, count=count)
        for code, count in sorted(value.items())
    )


def _validate_exclusion_counts(
    counts: tuple[CandidatePlanningExclusionCount, ...],
) -> None:
    if (
        not isinstance(counts, tuple)
        or any(not isinstance(item, CandidatePlanningExclusionCount) for item in counts)
        or tuple(item.code for item in counts)
        != tuple(sorted(item.code for item in counts))
        or len({item.code for item in counts}) != len(counts)
    ):
        raise ValueError("candidate_planning_exclusions_invalid")
    for item in counts:
        if not isinstance(item.code, str) or not item.code.strip():
            raise ValueError("candidate_planning_exclusion_code_invalid")
        _positive_integer(item.count)


def _positive_integer(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("candidate_planning_diagnostic_count_invalid")


def _non_negative_integer(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("candidate_planning_diagnostic_count_invalid")
