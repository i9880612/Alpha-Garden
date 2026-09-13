from __future__ import annotations

from dataclasses import dataclass

from generation.catalog import GenerationCatalog
from generation.formula import Binary, Call, Expression, Literal, Name, Prefix
from generation.structure_limits import fits_formula_structure_limits


_LOW_COVERAGE_THRESHOLD = 0.70
_COVERAGE_REPAIR_OPERATORS = frozenset({"group_backfill", "kth_element", "ts_backfill"})
_TAIL_CONTROL_OPERATORS = frozenset(
    {
        "group_rank",
        "group_scale",
        "group_zscore",
        "normalize",
        "quantile",
        "rank",
        "ts_quantile",
        "ts_rank",
        "winsorize",
        "zscore",
    }
)
_TAIL_RISK_OPERATORS = frozenset(
    {"divide", "inverse", "multiply", "power", "signed_power", "ts_product"}
)
_CONTINUOUS_TRANSFORMS = frozenset(
    {
        "group_rank",
        "group_scale",
        "group_zscore",
        "normalize",
        "quantile",
        "rank",
        "scale",
        "ts_quantile",
        "ts_rank",
        "ts_scale",
        "ts_zscore",
        "winsorize",
        "zscore",
    }
)


@dataclass(frozen=True, slots=True)
class CandidateRiskAssessment:
    rejection_reasons: tuple[str, ...]

    @property
    def eligible(self) -> bool:
        return not self.rejection_reasons


def assess_candidate_risk(
    expression: Expression,
    catalog: GenerationCatalog,
) -> CandidateRiskAssessment:
    reasons: set[str] = set()
    if not fits_formula_structure_limits(expression):
        reasons.add("formula_structure_budget_exceeded")
    sparse_unfilled_fields: set[str] = set()

    def visit(
        node: Expression,
        *,
        coverage_repaired: bool,
        tail_controlled: bool,
    ) -> None:
        if isinstance(node, Name):
            field = catalog.field(node.value)
            assert field is not None
            if field.field_type not in {"MATRIX", "VECTOR"}:
                return
            if field.coverage is None:
                reasons.add("field_coverage_missing")
            elif field.coverage == 0:
                reasons.add("field_coverage_zero")
            elif field.coverage < _LOW_COVERAGE_THRESHOLD and not coverage_repaired:
                sparse_unfilled_fields.add(field.field_id)
            return
        if isinstance(node, Literal):
            return
        if isinstance(node, Call):
            if node.operator in _TAIL_RISK_OPERATORS and not tail_controlled:
                reasons.add("uncontrolled_tail")
            if node.operator in {"equal", "not_equal"} and any(
                _contains_continuous_transform(argument.value)
                for argument in node.arguments
            ):
                reasons.add("continuous_exact_comparison")
            repaired = coverage_repaired or (
                node.operator in _COVERAGE_REPAIR_OPERATORS
            )
            controlled = tail_controlled or (node.operator in _TAIL_CONTROL_OPERATORS)
            for argument in node.arguments:
                visit(
                    argument.value,
                    coverage_repaired=repaired,
                    tail_controlled=controlled,
                )
            return
        if isinstance(node, Binary):
            if node.operator == "/" and not tail_controlled:
                reasons.add("uncontrolled_tail")
            if node.operator in {"==", "!="} and (
                _contains_continuous_transform(node.left)
                or _contains_continuous_transform(node.right)
            ):
                reasons.add("continuous_exact_comparison")
            visit(
                node.left,
                coverage_repaired=coverage_repaired,
                tail_controlled=tail_controlled,
            )
            visit(
                node.right,
                coverage_repaired=coverage_repaired,
                tail_controlled=tail_controlled,
            )
            return
        if isinstance(node, Prefix):
            visit(
                node.operand,
                coverage_repaired=coverage_repaired,
                tail_controlled=tail_controlled,
            )
            return
        raise TypeError(f"unsupported_expression:{type(node).__name__}")

    visit(expression, coverage_repaired=False, tail_controlled=False)
    if len(sparse_unfilled_fields) >= 2:
        reasons.add("multiple_sparse_inputs_without_fill")
    return CandidateRiskAssessment(tuple(sorted(reasons)))


def has_tail_risk_structure(expression: Expression) -> bool:
    if isinstance(expression, Call):
        return expression.operator in _TAIL_RISK_OPERATORS or any(
            has_tail_risk_structure(argument.value) for argument in expression.arguments
        )
    if isinstance(expression, Binary):
        return (
            expression.operator == "/"
            or has_tail_risk_structure(expression.left)
            or has_tail_risk_structure(expression.right)
        )
    if isinstance(expression, Prefix):
        return has_tail_risk_structure(expression.operand)
    return False


def _contains_continuous_transform(expression: Expression) -> bool:
    if isinstance(expression, Call):
        return expression.operator in _CONTINUOUS_TRANSFORMS or any(
            _contains_continuous_transform(argument.value)
            for argument in expression.arguments
        )
    if isinstance(expression, Binary):
        return _contains_continuous_transform(
            expression.left
        ) or _contains_continuous_transform(expression.right)
    if isinstance(expression, Prefix):
        return _contains_continuous_transform(expression.operand)
    return False
