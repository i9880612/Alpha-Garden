from __future__ import annotations

from generation.formula import Expression, analyze_formula


FORMULA_MAX_DEPTH = 10
FORMULA_MAX_COMPLEXITY = 48


def fits_formula_structure_limits(
    expression: Expression,
) -> bool:
    facts = analyze_formula(expression)
    return (
        facts.depth <= FORMULA_MAX_DEPTH and facts.complexity <= FORMULA_MAX_COMPLEXITY
    )
