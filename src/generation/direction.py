from __future__ import annotations

from generation.candidate import CandidateChange, FormulaCandidate, mutation_candidate
from generation.formula import Expression, Prefix, formula_fingerprint, render_formula
from generation.logic import prepare_formula_logic


DIRECTION_REVERSAL = "direction_reversal"


def reverse_direction_candidate(
    expression: Expression, *, parent_task_id: str
) -> FormulaCandidate:
    logic = prepare_formula_logic(Prefix("-", expression))
    if not logic.is_valid:
        raise ValueError("direction_formula_invalid")
    return mutation_candidate(
        logic.expression,
        parent_task_id=parent_task_id,
        parent_formula_fingerprint=formula_fingerprint(expression),
        change=CandidateChange(
            action=DIRECTION_REVERSAL,
            location="formula",
            before=render_formula(expression),
            after=render_formula(logic.expression),
        ),
    )
