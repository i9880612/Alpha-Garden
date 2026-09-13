"""Verify archived shared-structure edits for descendant research evidence.

No platform access, scheduling, persistence, or inferred quality ranking.
"""
from __future__ import annotations

from dataclasses import dataclass

from generation.arguments import bind_operator_arguments
from generation.catalog import GenerationCatalog
from generation.formula import (
    Binary, Call, Expression, Literal, Name, Prefix,
    analyze_formula,
)
from generation.logic import prepare_formula_logic
from generation.structure_limits import fits_formula_structure_limits
from generation.unit_validation import find_coarse_unit_issues
from generation.validation import validate_formula


@dataclass(frozen=True, slots=True)
class SharedOccurrence:
    parent_path: str
    reference_path: str
    kind: str
    name: str


@dataclass(frozen=True, slots=True)
class SharedStructure:
    occurrences: tuple[SharedOccurrence, ...]
    # Parent path, reference path, parent value, reference value.
    window_differences: tuple[tuple[str, str, str, str], ...]

def _children(node: Expression, path: str):
    if isinstance(node, Call):
        return tuple((a.value, f"{path}.arguments[{i}]") for i, a in enumerate(node.arguments))
    if isinstance(node, Binary):
        return ((node.left, path + ".left"), (node.right, path + ".right"))
    if isinstance(node, Prefix):
        return ((node.operand, path + ".operand"),)
    return ()


def _walk(node: Expression, path: str = "formula"):
    yield path, node
    for child, child_path in _children(node, path):
        yield from _walk(child, child_path)


def _field(node: Expression, catalog: GenerationCatalog) -> bool:
    return (
        isinstance(node, Name)
        and (field := catalog.field(node.value)) is not None
        and field.field_type in {"MATRIX", "VECTOR"}
    )


def _field_paths(node: Expression, catalog: GenerationCatalog, path="formula"):
    if _field(node, catalog):
        yield path
    if isinstance(node, Call):
        operator = catalog.operator(node.operator)
        assert operator is not None
        binding = bind_operator_arguments(operator.parameters, tuple(a.name for a in node.arguments))
        for arg in binding.assignments:
            if arg.parameter.kind == "expr":
                i = arg.argument_index
                yield from _field_paths(node.arguments[i].value, catalog, f"{path}.arguments[{i}]")
    else:
        for child, child_path in _children(node, path):
            yield from _field_paths(child, catalog, child_path)


def _shape(node: Expression, catalog: GenerationCatalog, *, window: bool = False):
    if isinstance(node, Literal):
        return ("window",) if window else ("literal", node.kind, node.value)
    if isinstance(node, Name):
        return ("name", node.value)
    if isinstance(node, Call):
        operator = catalog.operator(node.operator)
        assert operator is not None  # Public entry validates both input formulas.
        binding = bind_operator_arguments(operator.parameters, tuple(a.name for a in node.arguments))
        kinds = {a.argument_index: a.parameter.kind for a in binding.assignments}
        return ("call", node.operator, tuple(
            (arg.name, _shape(arg.value, catalog, window=kinds[i] == "window"))
            for i, arg in enumerate(node.arguments)
        ))
    return (type(node).__name__, node.operator, tuple(
        _shape(child, catalog) for child, _ in _children(node, "")
    ))


def match_shared_structure(
    parent: Expression, reference: Expression, catalog: GenerationCatalog,
) -> SharedStructure:
    """Greedy largest disjoint subtree matches, ignoring window values only.

    A reference occurrence cannot justify multiple parent occurrences. Matching
    requires an entire field-bearing subtree, not just operator-name overlap.
    Argument ordering, non-window literals, field names and signs stay exact.
    """
    if not all(validate_formula(x, catalog).is_valid for x in (parent, reference)):
        raise ValueError("shared_structure_invalid_formula")
    left, right = tuple(_walk(parent)), tuple(_walk(reference))
    left_fields, right_fields = set(_field_paths(parent, catalog)), set(_field_paths(reference, catalog))
    right_shapes = [(path, node, _shape(node, catalog)) for path, node in right
                    if any(p == path or p.startswith(path + ".") for p in right_fields)]
    possibilities = []
    for path, node in left:
        subtree = tuple(_walk(node, path))
        if not any(p in left_fields for p, _ in subtree):
            continue
        shape = _shape(node, catalog)
        size = sum(isinstance(n, Call) or p in left_fields for p, n in subtree)
        for ref_path, ref_node, ref_shape in right_shapes:
            if shape == ref_shape:
                possibilities.append((-size, path, ref_path, node, ref_node))
    used_left, used_right = set(), set()
    occurrences, windows = [], []
    for _, path, ref_path, node, ref_node in sorted(possibilities, key=lambda x: x[:3]):
        a, b = tuple(_walk(node, path)), tuple(_walk(ref_node, ref_path))
        if used_left.intersection(p for p, _ in a) or used_right.intersection(p for p, _ in b):
            continue
        used_left.update(p for p, _ in a)
        used_right.update(p for p, _ in b)
        for (p, n), (rp, rn) in zip(a, b, strict=True):
            if isinstance(n, Call) or p in left_fields:
                occurrences.append(SharedOccurrence(
                    p, rp, "operator" if isinstance(n, Call) else "field",
                    n.operator if isinstance(n, Call) else n.value,
                ))
            elif isinstance(n, Literal) and n != rn:
                # Equal subtree shapes allow differing literals only at windows.
                windows.append((p, rp, n.value, rn.value))
    return SharedStructure(
        tuple(sorted(occurrences, key=lambda x: next(i for i, (p, _) in enumerate(left) if p == x.parent_path))), tuple(sorted(windows)),
    )


def shared_half_paths(parent, reference, catalog, side: str) -> tuple[str, ...]:
    """Complementary sides in expression order; the odd middle node goes left."""
    if side not in {"left", "right"}:
        raise ValueError("shared_half_side_invalid")
    occurrences = match_shared_structure(parent, reference, catalog).occurrences
    if not occurrences:
        return ()
    midpoint = (len(occurrences) + 1) // 2
    half = occurrences[:midpoint] if side == "left" else occurrences[midpoint:]
    return tuple(x.parent_path for x in half)


def matches_shared_half(parent, reference, candidate, catalog, side: str) -> bool:
    """Recompute positions and legality from facts; labels never grant recovery."""
    if not all(validate_formula(x, catalog).is_valid for x in (parent, reference, candidate)):
        return False
    paths = shared_half_paths(parent, reference, catalog, side)
    before, after = dict(_walk(parent)), dict(_walk(candidate))
    if not paths or before.keys() != after.keys() or candidate in (parent, reference):
        return False
    changed = set()
    field_paths = set(_field_paths(parent, catalog))
    for path, node in before.items():
        target = after[path]
        if type(node) is not type(target):
            return False
        if isinstance(node, Call):
            if node.operator == target.operator and tuple(a.name for a in node.arguments) != tuple(a.name for a in target.arguments):
                return False
            if node.operator != target.operator:
                changed.add(path)
        elif isinstance(node, Name) and node != target:
            if path not in field_paths:
                return False
            source, dest = catalog.field(node.value), catalog.field(target.value)
            if dest is None or not source.dataset_id or source.dataset_id != dest.dataset_id or source.field_type != dest.field_type:
                return False
            changed.add(path)
        elif isinstance(node, (Binary, Prefix)):
            if node.operator != target.operator:
                return False
        elif node != target:
            return False
    facts, original_facts = analyze_formula(candidate), analyze_formula(parent)
    logic = prepare_formula_logic(candidate)
    return (changed == set(paths) and facts.depth <= original_facts.depth
            and facts.complexity <= original_facts.complexity
            and fits_formula_structure_limits(candidate)
            and not find_coarse_unit_issues(candidate, catalog)
            and logic.issue is None and logic.expression == candidate)
