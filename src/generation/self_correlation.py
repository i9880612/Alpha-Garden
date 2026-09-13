from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from generation.candidate import CandidateChange
from generation.catalog import GenerationCatalog
from generation.formula import (
    Binary,
    Call,
    CallArgument,
    Expression,
    Literal,
    Name,
    Prefix,
    analyze_formula,
    formula_fingerprint,
    render_formula,
)
from generation.internal_edits import (
    INTERNAL_FIELD_REPLACEMENT,
    INTERNAL_OPERATOR_REPLACEMENT,
    iter_internal_edit_candidates,
)
from generation.logic import prepare_formula_logic
from generation.structure_limits import fits_formula_structure_limits
from generation.unit_validation import find_coarse_unit_issues
from generation.validation import validate_formula
from generation.shared_rewrite import matches_shared_half


SELF_CORRELATION_REPAIR = "self_correlation_conflict_reference_residual"
SELF_CORRELATION_FIELD_REPLACEMENT = "self_correlation_shared_field_replacement"
SELF_CORRELATION_FIELD_OPERATOR_REPLACEMENT = "self_correlation_shared_field_operator_replacement"
SELF_CORRELATION_INTERNAL_FAMILIES = (
    SELF_CORRELATION_FIELD_REPLACEMENT,
    SELF_CORRELATION_FIELD_OPERATOR_REPLACEMENT,
)
SELF_CORRELATION_HALF_FAMILIES = (
    "self_correlation_shared_left_replacement",
    "self_correlation_shared_right_replacement",
)
SELF_CORRELATION_LIGHT_FAMILIES = (
    "self_correlation_industry_rank",
    "self_correlation_time_smoothing",
    "self_correlation_industry_neutralization",
)
SELF_CORRELATION_REPAIR_FAMILIES = (
    SELF_CORRELATION_REPAIR, *SELF_CORRELATION_INTERNAL_FAMILIES,
    *SELF_CORRELATION_HALF_FAMILIES, *SELF_CORRELATION_LIGHT_FAMILIES,
)


@dataclass(frozen=True, slots=True)
class SelfCorrelationLeaf:
    leaf_id: str
    family: str
    expression: Expression
    change: CandidateChange


def iter_self_correlation_leaves(
    expression: Expression,
    reference: Expression,
    catalog: GenerationCatalog,
    *,
    families: tuple[str, ...],
    neutralization: str,
    field_candidates: tuple[str, ...] = (),
) -> Iterator[SelfCorrelationLeaf]:
    if set(families) - set((*SELF_CORRELATION_LIGHT_FAMILIES, *SELF_CORRELATION_INTERNAL_FAMILIES)):
        raise ValueError("self_correlation_generation_family_invalid")
    if not families:
        return
    if not validate_formula(reference, catalog).is_valid:
        return
    if set(families).intersection(SELF_CORRELATION_LIGHT_FAMILIES):
        for leaf in _light_leaves(expression, catalog):
            if leaf.family not in families:
                continue
            if (leaf.family == "self_correlation_industry_neutralization"
                    and neutralization in {"INDUSTRY", "SUBINDUSTRY"}):
                continue
            yield leaf
    if set(families).intersection(SELF_CORRELATION_INTERNAL_FAMILIES):
        yield from (leaf for leaf in _internal_leaves(expression, reference, catalog, field_candidates)
                    if leaf.family in families)


def _light_leaves(
    expression: Expression, catalog: GenerationCatalog,
) -> Iterator[SelfCorrelationLeaf]:
    variants = (
        Call("group_rank", (CallArgument(expression), CallArgument(Name("industry")))),
        Call("ts_decay_linear", (CallArgument(expression), CallArgument(Literal("5", "integer")))),
        Call("group_neutralize", (CallArgument(expression), CallArgument(Name("industry")))),
    )
    for family, candidate in zip(SELF_CORRELATION_LIGHT_FAMILIES, variants):
        if (family == "self_correlation_industry_neutralization"
                and isinstance(expression, Call) and expression.operator == "group_neutralize"
                and any(isinstance(arg.value, Name) and arg.value.value in {"industry", "subindustry"}
                        for arg in expression.arguments)):
            continue
        logic = prepare_formula_logic(candidate)
        if logic.issue is not None:
            continue
        candidate = logic.expression
        if (
            candidate == expression
            or not fits_formula_structure_limits(candidate)
            or not validate_formula(candidate, catalog).is_valid
            or find_coarse_unit_issues(candidate, catalog)
        ):
            continue
        yield SelfCorrelationLeaf(
            formula_fingerprint(candidate), family, candidate,
            CandidateChange(family, "formula", render_formula(expression), render_formula(candidate)),
        )


def shared_field_replacements(
    expression: Expression, reference: Expression, catalog: GenerationCatalog,
    fields: tuple[str, ...],
) -> tuple[str, ...]:
    """Same-dataset/type search scope, not a predicted quality preference."""
    shared = set(analyze_formula(expression).referenced_names) & set(
        analyze_formula(reference).referenced_names
    )
    sources = [catalog.field(name) for name in shared]
    identities = {
        (field.dataset_id, field.field_type) for field in sources
        if field is not None and field.dataset_id
        and field.field_type in {"MATRIX", "VECTOR"}
    }
    return tuple(
        name for name in fields
        if (field := catalog.field(name)) is not None
        and (field.dataset_id, field.field_type) in identities
    )


def _residual_expression(expression: Expression, reference: Expression) -> Expression | None:
    logic = prepare_formula_logic(Call("vector_neut", (
        CallArgument(Call("zscore", (CallArgument(expression),))),
        CallArgument(Call("zscore", (CallArgument(reference),))),
    )))
    return logic.expression if logic.issue is None else None


def _nodes(
    expression: Expression, location: str = "formula",
) -> Iterator[tuple[str, Expression]]:
    yield location, expression
    if isinstance(expression, Call):
        for index, arg in enumerate(expression.arguments):
            yield from _nodes(arg.value, f"{location}.arguments[{index}]")
    elif isinstance(expression, Binary):
        for side in ("left", "right"):
            yield from _nodes(getattr(expression, side), f"{location}.{side}")
    elif isinstance(expression, Prefix):
        yield from _nodes(expression.operand, f"{location}.operand")


def _shared_operator_locations(
    expression: Expression, reference: Expression, field_location: str, field_name: str,
) -> frozenset[str]:
    reference_operators = {
        node.operator for _, node in _nodes(reference)
        if isinstance(node, Call) and field_name in analyze_formula(node).referenced_names
    }
    return frozenset(
        location for location, node in _nodes(expression)
        if location != "formula" and field_location.startswith(location + ".")
        and isinstance(node, Call) and node.operator in reference_operators
    )


def _internal_leaves(
    expression: Expression, reference: Expression, catalog: GenerationCatalog,
    fields: tuple[str, ...],
) -> Iterator[SelfCorrelationLeaf]:
    """One shared field occurrence, optionally plus one shared ancestor operator.

    Root operators and every literal/window are preserved. Both edits use the
    ordinary validator; combined candidates remain children of the real parent.
    """
    common_names = set(analyze_formula(reference).referenced_names)
    seen = {formula_fingerprint(expression), formula_fingerprint(reference)}
    for location, node in _nodes(expression):
        if not isinstance(node, Name) or node.value not in common_names:
            continue
        source = catalog.field(node.value)
        if source is None or not source.dataset_id:
            continue
        replacements = tuple(
            name for name in fields
            if (target := catalog.field(name)) is not None
            and target.dataset_id == source.dataset_id
            and target.field_type == source.field_type
        )
        if not replacements:
            continue
        operators = _shared_operator_locations(expression, reference, location, node.value)
        for field_edit in iter_internal_edit_candidates(
            expression, catalog, parent_task_id="sc-candidate",
            field_candidates=replacements, families=(INTERNAL_FIELD_REPLACEMENT,),
            locations=frozenset({location}),
        ):
            edits = [(field_edit, SELF_CORRELATION_FIELD_REPLACEMENT)]
            if operators:
                edits.extend(
                    (item, SELF_CORRELATION_FIELD_OPERATOR_REPLACEMENT)
                    for item in iter_internal_edit_candidates(
                        field_edit.expression, catalog, parent_task_id="sc-candidate",
                        families=(INTERNAL_OPERATOR_REPLACEMENT,), locations=operators,
                    )
                )
            for edit, family in edits:
                if edit.fingerprint in seen:
                    continue
                seen.add(edit.fingerprint)
                yield SelfCorrelationLeaf(
                    edit.fingerprint, family, edit.expression,
                    CandidateChange(family, location, render_formula(expression), edit.formula),
                )


def matches_self_correlation_repair(
    expression: Expression, reference: Expression, candidate: Expression,
    *, action: str, location: str,
    catalog: GenerationCatalog | None = None,
) -> bool:
    """Verify recorded repair shape; a mutation label alone grants no lineage."""
    if action in SELF_CORRELATION_LIGHT_FAMILIES:
        return bool(
            catalog is not None and location == "formula"
            and any(leaf.family == action and leaf.expression == candidate
                    for leaf in _light_leaves(expression, catalog))
        )
    if action == SELF_CORRELATION_REPAIR:
        return expression != reference and _residual_expression(expression, reference) == candidate
    if action in SELF_CORRELATION_HALF_FAMILIES:
        return bool(catalog is not None and location == "formula" and matches_shared_half(
            expression, reference, candidate, catalog,
            "left" if action == SELF_CORRELATION_HALF_FAMILIES[0] else "right",
        ))
    if action not in SELF_CORRELATION_INTERNAL_FAMILIES:
        return False
    before, after = dict(_nodes(expression)), dict(_nodes(candidate))
    if before.keys() != after.keys() or expression == candidate or candidate == reference:
        return False
    fields, operators = [], []
    for path, node in before.items():
        replacement = after[path]
        if type(node) is not type(replacement):
            return False
        if isinstance(node, Name):
            if node != replacement:
                fields.append(path)
        elif isinstance(node, Call):
            if node.operator != replacement.operator:
                operators.append(path)
            elif tuple(a.name for a in node.arguments) != tuple(a.name for a in replacement.arguments):
                return False
        elif isinstance(node, (Binary, Prefix)):
            if node.operator != replacement.operator:
                return False
        elif node != replacement:
            return False
    if fields != [location]:
        return False
    field = before[location]
    if field.value not in analyze_formula(reference).referenced_names:
        return False
    if action == SELF_CORRELATION_FIELD_REPLACEMENT:
        return not operators
    return len(operators) == 1 and operators[0] in _shared_operator_locations(
        expression, reference, location, field.value,
    )
