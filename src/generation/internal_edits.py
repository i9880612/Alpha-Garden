from __future__ import annotations

from collections.abc import Callable, Iterator
from itertools import chain
from dataclasses import replace

from generation.arguments import bind_operator_arguments
from generation.candidate import CandidateChange, FormulaCandidate, mutation_candidate
from generation.catalog import GenerationCatalog
from generation.formula import (
    Binary,
    Call,
    CallArgument,
    Expression,
    Name,
    Prefix,
    analyze_formula,
    formula_fingerprint,
    render_formula,
)
from generation.logic import prepare_formula_logic
from generation.structure_limits import fits_formula_structure_limits
from generation.transformations import signal_key
from generation.unit_validation import find_coarse_unit_issues
from generation.validation import validate_formula


INTERNAL_OPERATOR_REPLACEMENT = "internal_operator_replacement"
INTERNAL_FIELD_REPLACEMENT = "internal_field_replacement"
INTERNAL_LAYER_REMOVAL = "internal_layer_removal"
INTERNAL_EDIT_FAMILIES = (
    INTERNAL_OPERATOR_REPLACEMENT,
    INTERNAL_FIELD_REPLACEMENT,
    INTERNAL_LAYER_REMOVAL,
)
# Dropping ts_backfill on sparse data only adds gaps; it never yields a new idea.
_MISSING_VALUE_REPAIR = "missing_value_repair"
# rank and quantile order stocks the same way; their backtests moved together.
_EQUIVALENT_OPERATOR_PAIRS = ({"rank", "quantile"},)


def iter_internal_edit_candidates(
    expression: Expression,
    catalog: GenerationCatalog,
    *,
    parent_task_id: str,
    field_candidates: tuple[str, ...] = (),
    families: tuple[str, ...] = INTERNAL_EDIT_FAMILIES,
    locations: frozenset[str] | None = None,
    field_relatives: Callable[[str], tuple[str, ...]] | None = None,
) -> Iterator[FormulaCandidate]:
    """One local edit per candidate, with no increase in depth or complexity.

    Field inputs are explicitly supplied by the caller, not chosen as a quality
    preference here. Passing ``field_relatives`` turns on related-only editing: a field
    is only swapped for its own relatives (same kind, scale and cadence), every
    occurrence at once so vwap/ts_delay(vwap,3) becomes close/ts_delay(close,3), operators only
    within a shared catalog role (smoothing for smoothing, never rank for quantile), and
    missing-value repair layers are never removed. Enumeration order is
    deterministic, not a learned ranking. Selection still owns coverage/tail-risk
    checks and allocation before use.
    """
    if not isinstance(parent_task_id, str) or not parent_task_id.strip():
        raise ValueError("internal_edit_parent_task_id_invalid")
    if not isinstance(catalog, GenerationCatalog):
        raise ValueError("internal_edit_catalog_invalid")
    if not isinstance(field_candidates, tuple) or any(
        not isinstance(name, str) or catalog.field(name) is None
        for name in field_candidates
    ):
        raise ValueError("internal_edit_field_candidates_invalid")
    if not _valid(expression, catalog):
        raise ValueError("internal_edit_parent_invalid")
    if not set(families) <= set(INTERNAL_EDIT_FAMILIES):
        raise ValueError("internal_edit_families_invalid")
    facts = analyze_formula(expression)
    parent_fingerprint = formula_fingerprint(expression)
    seen = {parent_fingerprint}
    # Two edits can reach the same signal (normalize(x) and zscore(x)); test it once.
    seen_signals = {signal_key(expression)}
    edits = _edits(
        expression, catalog, tuple(sorted(set(field_candidates))), "formula",
        families, locations, field_relatives,
    )
    if field_relatives is not None and INTERNAL_FIELD_REPLACEMENT in families and locations is None:
        edits = chain(_whole_field_swaps(expression, catalog, field_relatives), edits)
    for transformed, change in edits:
        # Do not silently perform a second edit via global simplification.
        if not _valid(transformed, catalog):
            continue
        child_facts = analyze_formula(transformed)
        if child_facts.depth > facts.depth or child_facts.complexity > facts.complexity:
            continue
        fingerprint = formula_fingerprint(transformed)
        signal = signal_key(transformed)
        if fingerprint in seen or signal in seen_signals:
            continue
        seen.add(fingerprint)
        seen_signals.add(signal)
        yield mutation_candidate(
            transformed,
            parent_task_id=parent_task_id,
            parent_formula_fingerprint=parent_fingerprint,
            change=change,
        )


def _whole_field_swaps(
    expression: Expression,
    catalog: GenerationCatalog,
    field_relatives: Callable[[str], tuple[str, ...]],
) -> Iterator[tuple[Expression, CandidateChange]]:
    for name in dict.fromkeys(analyze_formula(expression).referenced_names):
        source = catalog.field(name)
        if source is None or source.field_type not in {"MATRIX", "VECTOR"}:
            continue
        for relative in field_relatives(name):
            target = catalog.field(relative)
            if target is None or relative == name or target.field_type != source.field_type:
                continue
            yield (
                _substitute(expression, name, relative),
                CandidateChange(INTERNAL_FIELD_REPLACEMENT, "formula", name, relative),
            )


def _substitute(node: Expression, old: str, new: str) -> Expression:
    if isinstance(node, Name):
        return Name(new) if node.value == old else node
    if isinstance(node, Call):
        return replace(node, arguments=tuple(
            replace(argument, value=_substitute(argument.value, old, new))
            for argument in node.arguments))
    if isinstance(node, Binary):
        return replace(node, left=_substitute(node.left, old, new),
                       right=_substitute(node.right, old, new))
    if isinstance(node, Prefix):
        return replace(node, operand=_substitute(node.operand, old, new))
    return node


def _valid(expression: Expression, catalog: GenerationCatalog) -> bool:
    if not fits_formula_structure_limits(expression):
        return False
    if not validate_formula(expression, catalog).is_valid:
        return False
    if find_coarse_unit_issues(expression, catalog):
        return False
    logic = prepare_formula_logic(expression)
    # Compare the written form: a leading negative weight (-1.7*rank(...)) parses as a
    # negated literal but normalizes to a negative literal, the same formula (10-23:
    # two such qualified combos stopped a cycle's planning as invalid parents).
    return logic.issue is None and (logic.expression == expression
                                    or render_formula(logic.expression) == render_formula(expression))


def _edits(
    node: Expression,
    catalog: GenerationCatalog,
    fields: tuple[str, ...],
    location: str,
    families: tuple[str, ...],
    locations: frozenset[str] | None,
    field_relatives: Callable[[str], tuple[str, ...]] | None = None,
) -> Iterator[tuple[Expression, CandidateChange]]:
    editable = locations is None or location in locations
    if isinstance(node, Name):
        source = catalog.field(node.value)
        if (editable and INTERNAL_FIELD_REPLACEMENT in families and field_relatives is None
                and source is not None and source.field_type in {"MATRIX", "VECTOR"}):
            names = fields if field_relatives is None else field_relatives(source.field_id)
            for name in names:
                target = catalog.field(name)
                if target is None:
                    continue
                if name != source.field_id and target.field_type == source.field_type:
                    yield (
                        Name(name),
                        CandidateChange(
                            INTERNAL_FIELD_REPLACEMENT, location, node.value, name
                        ),
                    )
        return
    if isinstance(node, Call):
        source = catalog.operator(node.operator)
        assert source is not None
        binding = bind_operator_arguments(
            source.parameters, tuple(a.name for a in node.arguments)
        )
        before = render_formula(node)
        if editable and INTERNAL_OPERATOR_REPLACEMENT in families and source.output_kind is not None and not any(
            p.variadic for p in source.parameters
        ):
            for target in catalog.operators:
                if (
                    target.name == source.name
                    or (field_relatives is not None and (
                        not set(source.roles) & set(target.roles)
                        or {source.name, target.name} in _EQUIVALENT_OPERATOR_PAIRS))
                    or target.output_kind != source.output_kind
                    or "REGULAR" not in target.scope
                    or any(p.variadic for p in target.parameters)
                ):
                    continue
                arguments = []
                for assignment in sorted(binding.assignments, key=lambda item: item.parameter_index):
                    index = assignment.parameter_index
                    if (
                        index >= len(target.parameters)
                        or target.parameters[index].kind != assignment.parameter.kind
                    ):
                        break
                    argument = node.arguments[assignment.argument_index]
                    # Follow the target signature: required inputs are positional,
                    # optional inputs are named (e.g. ts_backfill's lookback).
                    arguments.append(CallArgument(
                        argument.value,
                        target.parameters[index].name
                        if target.parameters[index].optional
                        else None,
                    ))
                else:
                    replacement = Call(target.name, tuple(arguments))
                    target_binding = bind_operator_arguments(
                        target.parameters, tuple(a.name for a in replacement.arguments)
                    )
                    if not target_binding.issues:
                        yield (
                            replacement,
                            CandidateChange(
                                INTERNAL_OPERATOR_REPLACEMENT,
                                location,
                                before,
                                render_formula(replacement),
                            ),
                        )
        inputs = [a for a in binding.assignments if a.parameter.kind == "expr"]
        if (editable and INTERNAL_LAYER_REMOVAL in families and len(inputs) == 1
                and (field_relatives is None or _MISSING_VALUE_REPAIR not in source.roles)):
            replacement = node.arguments[inputs[0].argument_index].value
            yield (
                replacement,
                CandidateChange(
                    INTERNAL_LAYER_REMOVAL,
                    location,
                    before,
                    render_formula(replacement),
                ),
            )
        for index, argument in enumerate(node.arguments):
            # String options may look like field identifiers; never edit them.
            parameter = next(
                a.parameter for a in binding.assignments if a.argument_index == index
            )
            if parameter.kind not in {"expr", "group"}:
                continue
            for replacement, change in _edits(
                argument.value, catalog, fields, f"{location}.arguments[{index}]",
                families, locations, field_relatives,
            ):
                arguments = list(node.arguments)
                arguments[index] = replace(argument, value=replacement)
                yield replace(node, arguments=tuple(arguments)), change
    elif isinstance(node, Binary):
        for side in ("left", "right"):
            for replacement, change in _edits(
                getattr(node, side), catalog, fields, f"{location}.{side}",
                families, locations, field_relatives,
            ):
                yield replace(node, **{side: replacement}), change
    elif isinstance(node, Prefix):
        for replacement, change in _edits(
            node.operand, catalog, fields, f"{location}.operand", families, locations,
            field_relatives,
        ):
            yield replace(node, operand=replacement), change
