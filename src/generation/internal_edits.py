from __future__ import annotations

from collections.abc import Iterator
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


def iter_internal_edit_candidates(
    expression: Expression,
    catalog: GenerationCatalog,
    *,
    parent_task_id: str,
    field_candidates: tuple[str, ...] = (),
    families: tuple[str, ...] = INTERNAL_EDIT_FAMILIES,
    locations: frozenset[str] | None = None,
) -> Iterator[FormulaCandidate]:
    """One local edit per candidate, with no increase in depth or complexity.

    Field inputs are explicitly supplied by the caller, not chosen as a quality
    preference here. Enumeration order is deterministic, not a learned ranking.
    Selection still owns coverage/tail-risk checks and allocation before use.
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
    for transformed, change in _edits(
        expression, catalog, tuple(sorted(set(field_candidates))), "formula",
        families, locations,
    ):
        # Do not silently perform a second edit via global simplification.
        if not _valid(transformed, catalog):
            continue
        child_facts = analyze_formula(transformed)
        if child_facts.depth > facts.depth or child_facts.complexity > facts.complexity:
            continue
        fingerprint = formula_fingerprint(transformed)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        yield mutation_candidate(
            transformed,
            parent_task_id=parent_task_id,
            parent_formula_fingerprint=parent_fingerprint,
            change=change,
        )


def _valid(expression: Expression, catalog: GenerationCatalog) -> bool:
    if not fits_formula_structure_limits(expression):
        return False
    if not validate_formula(expression, catalog).is_valid:
        return False
    if find_coarse_unit_issues(expression, catalog):
        return False
    logic = prepare_formula_logic(expression)
    return logic.issue is None and logic.expression == expression


def _edits(
    node: Expression,
    catalog: GenerationCatalog,
    fields: tuple[str, ...],
    location: str,
    families: tuple[str, ...],
    locations: frozenset[str] | None,
) -> Iterator[tuple[Expression, CandidateChange]]:
    editable = locations is None or location in locations
    if isinstance(node, Name):
        source = catalog.field(node.value)
        if (editable and INTERNAL_FIELD_REPLACEMENT in families
                and source is not None and source.field_type in {"MATRIX", "VECTOR"}):
            for name in fields:
                target = catalog.field(name)
                assert target is not None
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
                    or target.output_kind != source.output_kind
                    or "REGULAR" not in target.scope
                    or any(p.variadic for p in target.parameters)
                ):
                    continue
                arguments = list(node.arguments)
                for assignment in binding.assignments:
                    index = assignment.parameter_index
                    if (
                        index >= len(target.parameters)
                        or target.parameters[index].kind != assignment.parameter.kind
                    ):
                        break
                    argument = arguments[assignment.argument_index]
                    arguments[assignment.argument_index] = CallArgument(
                        argument.value,
                        target.parameters[index].name
                        if argument.name is not None
                        else None,
                    )
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
        if editable and INTERNAL_LAYER_REMOVAL in families and len(inputs) == 1:
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
                families, locations,
            ):
                arguments = list(node.arguments)
                arguments[index] = replace(argument, value=replacement)
                yield replace(node, arguments=tuple(arguments)), change
    elif isinstance(node, Binary):
        for side in ("left", "right"):
            for replacement, change in _edits(
                getattr(node, side), catalog, fields, f"{location}.{side}",
                families, locations,
            ):
                yield replace(node, **{side: replacement}), change
    elif isinstance(node, Prefix):
        for replacement, change in _edits(
            node.operand, catalog, fields, f"{location}.operand", families, locations,
        ):
            yield replace(node, operand=replacement), change
