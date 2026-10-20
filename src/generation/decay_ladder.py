"""Decay ladder: the formula-level version of raising the platform's decay setting.

The platform's decay setting is a linear decay of the final alpha values, so
``ts_decay_linear(parent, n)`` adds n days of decay without touching the run's settings
(lineages, budgets and learning are all keyed by settings and stay comparable). Rungs are
offered smallest first: the first round showed that heavy smoothing cuts turnover but
also halves returns, so a high-turnover parent climbs only as far as it needs.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from generation.catalog import GenerationCatalog
from generation.formula import (
    Call,
    CallArgument,
    Expression,
    Literal,
    formula_fingerprint,
    render_formula,
)
from generation.logic import prepare_formula_logic
from generation.structure_limits import fits_formula_structure_limits
from generation.transformations import TransformationChange
from generation.unit_validation import find_coarse_unit_issues
from generation.validation import validate_formula

SIGNAL_DECAY_LADDER = "signal_decay_ladder"
DECAY_LADDER_FAMILIES = (SIGNAL_DECAY_LADDER,)
DECAY_LADDER_RUNGS = (3, 6, 10, 15)
_DECAY_OPERATOR = "ts_decay_linear"


@dataclass(frozen=True, slots=True)
class DecayLadderLeaf:
    leaf_id: str
    expression: Expression
    change: TransformationChange
    rung: int
    family: str = SIGNAL_DECAY_LADDER


def iter_decay_ladder_leaves(
    expression: Expression,
    catalog: GenerationCatalog,
) -> Iterator[DecayLadderLeaf]:
    """One leaf per rung, smallest first; none if the catalog lacks ts_decay_linear."""
    if catalog.operator(_DECAY_OPERATOR) is None:
        return
    if isinstance(expression, Call) and expression.operator == _DECAY_OPERATOR:
        # Already decayed at the top: another layer is what the next rung would be,
        # and stacking them hides how much decay the formula really carries.
        return
    before = render_formula(expression)
    parent_fingerprint = formula_fingerprint(expression)
    for rung in DECAY_LADDER_RUNGS:
        child = Call(_DECAY_OPERATOR, (
            CallArgument(expression), CallArgument(Literal(str(rung), "integer"))))
        if not validate_formula(child, catalog).is_valid or find_coarse_unit_issues(child, catalog):
            continue
        logic = prepare_formula_logic(child)
        if logic.issue is not None or not fits_formula_structure_limits(logic.expression):
            continue
        after = render_formula(logic.expression)
        yield DecayLadderLeaf(
            leaf_id=f"{SIGNAL_DECAY_LADDER}:{parent_fingerprint}:{rung}",
            expression=logic.expression,
            change=TransformationChange(location="formula", before=before, after=after),
            rung=rung,
        )
