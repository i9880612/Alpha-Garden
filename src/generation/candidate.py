from __future__ import annotations

from dataclasses import dataclass

from generation.formula import (
    Expression,
    formula_fingerprint,
    render_formula,
)
from generation.logic import prepare_formula_logic


@dataclass(frozen=True, slots=True)
class CandidateChange:
    action: str
    location: str
    before: str
    after: str
    parameters: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.action, "candidate_change_action_missing")
        _require_text(self.location, "candidate_change_location_missing")
        _require_text(self.before, "candidate_change_before_missing")
        _require_text(self.after, "candidate_change_after_missing")
        names: list[str] = []
        for name, value in self.parameters:
            _require_text(name, "candidate_change_parameter_name_missing")
            _require_text(value, "candidate_change_parameter_value_missing")
            names.append(name)
        if len(set(names)) != len(names):
            raise ValueError("candidate_change_parameter_duplicate")


@dataclass(frozen=True, slots=True)
class FormulaCandidate:
    expression: Expression
    generation_action: str
    parent_task_id: str | None = None
    parent_formula_fingerprint: str | None = None
    change: CandidateChange | None = None
    strategy_label: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.generation_action, "candidate_generation_action_missing")
        if self.strategy_label is not None:
            _require_text(self.strategy_label, "candidate_strategy_label_invalid")
        has_parent_task = self.parent_task_id is not None
        has_parent_formula = self.parent_formula_fingerprint is not None
        if has_parent_task:
            _require_text(self.parent_task_id, "candidate_parent_task_id_invalid")
        if has_parent_formula:
            _require_text(
                self.parent_formula_fingerprint,
                "candidate_parent_formula_fingerprint_invalid",
            )
        if len(
            {
                has_parent_task,
                has_parent_formula,
                self.change is not None,
            }
        ) != 1:
            raise ValueError("candidate_lineage_incomplete")
        logic = prepare_formula_logic(self.expression)
        if logic.issue is not None:
            raise ValueError(f"candidate_logic_invalid:{logic.issue.code}")
        object.__setattr__(self, "expression", logic.expression)

    @property
    def formula(self) -> str:
        return render_formula(self.expression)

    @property
    def fingerprint(self) -> str:
        return formula_fingerprint(self.expression)


def exploration_candidate(
    expression: Expression,
    *,
    strategy_label: str | None = None,
) -> FormulaCandidate:
    return FormulaCandidate(
        expression=expression,
        generation_action="exploration",
        strategy_label=strategy_label,
    )


def mutation_candidate(
    expression: Expression,
    *,
    parent_task_id: str,
    parent_formula_fingerprint: str,
    change: CandidateChange,
) -> FormulaCandidate:
    return FormulaCandidate(
        expression=expression,
        generation_action="mutation",
        parent_task_id=parent_task_id,
        parent_formula_fingerprint=parent_formula_fingerprint,
        change=change,
    )


def _require_text(value: object, error: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(error)
