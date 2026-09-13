from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from hashlib import sha256
from itertools import chain
from math import gcd, lcm, prod
from typing import TypeAlias

from generation.catalog import GenerationCatalog, OperatorDefinition
from generation.formula import (
    Binary,
    Call,
    CallArgument,
    Expression,
    Literal,
    Name,
    analyze_formula,
    formula_fingerprint,
    render_formula,
)
from generation.logic import prepare_formula_logic
from generation.structure_limits import fits_formula_structure_limits
from generation.unit_validation import find_coarse_unit_issues
from generation.validation import validate_formula


TEMPORAL_CHANGE_REFRAME = "temporal_change_reframe"
TEMPORAL_PERSISTENCE_REFRAME = "temporal_persistence_reframe"
DISTRIBUTION_STABILIZATION = "distribution_stabilization"
GROUP_RELATIVE_REFRAME = "group_relative_reframe"
COMPLEMENTARY_SIGNAL_REFRAME = "complementary_signal_reframe"
STRUCTURAL_TRANSFORMATION_FAMILIES = (
    TEMPORAL_CHANGE_REFRAME,
    TEMPORAL_PERSISTENCE_REFRAME,
    DISTRIBUTION_STABILIZATION,
    GROUP_RELATIVE_REFRAME,
    COMPLEMENTARY_SIGNAL_REFRAME,
)
MAX_TRANSFORMATION_LEAVES_PER_FAMILY = 64

_CROSS_SECTIONAL_NORMALIZATION = frozenset({"cross_sectional_normalization"})
_CROSS_SECTIONAL_OUTLIER_CONTROL = frozenset({"cross_sectional_outlier_control"})
_TIME_SERIES_CHANGE = frozenset({"time_series_change"})
_TIME_SERIES_SMOOTHING = frozenset({"time_series_smoothing"})
_TIME_SERIES_NORMALIZATION = frozenset({"time_series_normalization"})
_GROUP_RELATIVE_OPERATORS = frozenset({"group_rank", "group_zscore"})


@dataclass(frozen=True, slots=True)
class TemporalReframeSpec:
    family: str
    temporal_operator: str
    outer_normalizer: str
    window: int


@dataclass(frozen=True, slots=True)
class DistributionStabilizationSpec:
    control_operator: str
    outer_normalizer: str
    family: str = field(
        init=False,
        default=DISTRIBUTION_STABILIZATION,
    )


@dataclass(frozen=True, slots=True)
class GroupRelativeReframeSpec:
    input_normalizer: str
    group_operator: str
    group_field: str
    family: str = field(init=False, default=GROUP_RELATIVE_REFRAME)


@dataclass(frozen=True, slots=True)
class ComplementarySignalReframeSpec:
    left_normalizer: str
    temporal_normalizer: str
    outer_normalizer: str
    window: int
    complement_field: str
    family: str = field(init=False, default=COMPLEMENTARY_SIGNAL_REFRAME)


TransformationSpec: TypeAlias = (
    TemporalReframeSpec
    | DistributionStabilizationSpec
    | GroupRelativeReframeSpec
    | ComplementarySignalReframeSpec
)


@dataclass(frozen=True, slots=True)
class TransformationChange:
    location: str
    before: str
    after: str


@dataclass(frozen=True, slots=True)
class TransformationLeaf:
    leaf_id: str
    formula_fingerprint: str
    spec: TransformationSpec
    expression: Expression
    change: TransformationChange

    @property
    def family(self) -> str:
        return self.spec.family


def iter_transformation_leaves(
    expression: Expression,
    catalog: GenerationCatalog,
    *,
    families: tuple[str, ...] = STRUCTURAL_TRANSFORMATION_FAMILIES,
    field_candidates: tuple[str, ...] = (),
    group_candidates: tuple[str, ...] = (),
) -> Iterator[TransformationLeaf]:
    _validate_request(
        catalog,
        families=families,
        field_candidates=field_candidates,
        group_candidates=group_candidates,
    )
    parent_validation = validate_formula(expression, catalog)
    if not parent_validation.is_valid or find_coarse_unit_issues(expression, catalog):
        raise ValueError("structural_transformation_parent_invalid")

    before = render_formula(expression)
    requested_families = set(families)
    used_formula_fingerprints: set[str] = set()
    for family in STRUCTURAL_TRANSFORMATION_FAMILIES:
        if family not in requested_families:
            continue
        candidates = iter(
            _family_candidates(
                expression,
                catalog,
                family=family,
                field_candidates=field_candidates,
                group_candidates=group_candidates,
            )
        )
        first = next(candidates, None)
        # Parameters change identities, not the AST shape inside one family.
        # Reject an over-limit family before traversing a large catalog product.
        if first is None or not fits_formula_structure_limits(first[1]):
            continue
        family_count = 0
        for spec, transformed in chain((first,), candidates):
            prepared = _prepare_leaf(transformed, catalog, before=before)
            if prepared is None:
                continue
            fingerprint = formula_fingerprint(prepared)
            if fingerprint in used_formula_fingerprints:
                continue
            used_formula_fingerprints.add(fingerprint)
            yield TransformationLeaf(
                leaf_id=f"{family}:{fingerprint}",
                formula_fingerprint=fingerprint,
                spec=spec,
                expression=prepared,
                change=TransformationChange(
                    location="formula",
                    before=before,
                    after=render_formula(prepared),
                ),
            )
            family_count += 1
            if family_count == MAX_TRANSFORMATION_LEAVES_PER_FAMILY:
                break


def _family_candidates(
    expression: Expression,
    catalog: GenerationCatalog,
    *,
    family: str,
    field_candidates: tuple[str, ...],
    group_candidates: tuple[str, ...],
) -> Iterator[tuple[TransformationSpec, Expression]]:
    if family == TEMPORAL_CHANGE_REFRAME:
        yield from _temporal_reframes(
            expression,
            catalog,
            family=family,
            roles=_TIME_SERIES_CHANGE,
        )
        return
    if family == TEMPORAL_PERSISTENCE_REFRAME:
        yield from _temporal_reframes(
            expression,
            catalog,
            family=family,
            roles=_TIME_SERIES_SMOOTHING,
        )
        return
    if family == DISTRIBUTION_STABILIZATION:
        yield from _distribution_stabilizations(expression, catalog)
        return
    if family == GROUP_RELATIVE_REFRAME:
        yield from _group_relative_reframes(
            expression,
            catalog,
            group_candidates,
        )
        return
    yield from _complementary_signal_reframes(
        expression,
        catalog,
        field_candidates,
    )


def _temporal_reframes(
    expression: Expression,
    catalog: GenerationCatalog,
    *,
    family: str,
    roles: frozenset[str],
) -> Iterator[tuple[TemporalReframeSpec, Expression]]:
    operators = _operators(catalog, roles=roles, kinds=("expr", "window"))
    normalizers = _operators(
        catalog,
        roles=_CROSS_SECTIONAL_NORMALIZATION,
        kinds=("expr",),
    )
    windows = catalog.window_values()
    identity = f"{family}:{formula_fingerprint(expression)}"
    for operator_index, normalizer_index, window_index in _balanced_product_indexes(
        identity,
        (len(operators), len(normalizers), len(windows)),
    ):
        operator = operators[operator_index]
        normalizer = normalizers[normalizer_index]
        window = windows[window_index]
        spec = TemporalReframeSpec(
            family=family,
            temporal_operator=operator.name,
            outer_normalizer=normalizer.name,
            window=window,
        )
        temporal = Call(
            operator.name,
            (
                CallArgument(expression),
                CallArgument(_integer(window)),
            ),
        )
        yield (
            spec,
            Call(
                normalizer.name,
                (CallArgument(temporal),),
            ),
        )


def _distribution_stabilizations(
    expression: Expression,
    catalog: GenerationCatalog,
) -> Iterator[tuple[DistributionStabilizationSpec, Expression]]:
    controls = _operators(
        catalog,
        roles=_CROSS_SECTIONAL_OUTLIER_CONTROL,
        kinds=("expr",),
    )
    normalizers = _operators(
        catalog,
        roles=_CROSS_SECTIONAL_NORMALIZATION,
        kinds=("expr",),
    )
    identity = f"{DISTRIBUTION_STABILIZATION}:{formula_fingerprint(expression)}"
    for control_index, normalizer_index in _balanced_product_indexes(
        identity,
        (len(controls), len(normalizers)),
    ):
        control = controls[control_index]
        normalizer = normalizers[normalizer_index]
        spec = DistributionStabilizationSpec(
            control_operator=control.name,
            outer_normalizer=normalizer.name,
        )
        controlled = Call(control.name, (CallArgument(expression),))
        yield (
            spec,
            Call(
                normalizer.name,
                (CallArgument(controlled),),
            ),
        )


def _group_relative_reframes(
    expression: Expression,
    catalog: GenerationCatalog,
    requested_groups: tuple[str, ...],
) -> Iterator[tuple[GroupRelativeReframeSpec, Expression]]:
    groups = tuple(
        field.field_id
        for field in (
            tuple(catalog.field(field_id) for field_id in requested_groups)
            if requested_groups
            else catalog.fields
        )
        if field is not None
        and field.field_type == "GROUP"
        and field.coverage is not None
        and field.coverage > 0
    )
    group_operators = tuple(
        operator
        for operator in catalog.operators
        if operator.name in _GROUP_RELATIVE_OPERATORS
        and operator.output_kind == "signal"
        and _matches_signature(operator, ("expr", "group"))
    )
    normalizers = _operators(
        catalog,
        roles=_CROSS_SECTIONAL_NORMALIZATION,
        kinds=("expr",),
    )
    groups = tuple(sorted(groups))
    identity = f"{GROUP_RELATIVE_REFRAME}:{formula_fingerprint(expression)}"
    for (
        normalizer_index,
        group_operator_index,
        group_index,
    ) in _balanced_product_indexes(
        identity,
        (len(normalizers), len(group_operators), len(groups)),
    ):
        normalizer = normalizers[normalizer_index]
        group_operator = group_operators[group_operator_index]
        group = groups[group_index]
        spec = GroupRelativeReframeSpec(
            input_normalizer=normalizer.name,
            group_operator=group_operator.name,
            group_field=group,
        )
        normalized = Call(
            normalizer.name,
            (CallArgument(expression),),
        )
        yield (
            spec,
            Call(
                group_operator.name,
                (
                    CallArgument(normalized),
                    CallArgument(Name(group)),
                ),
            ),
        )


def _complementary_signal_reframes(
    expression: Expression,
    catalog: GenerationCatalog,
    requested_fields: tuple[str, ...],
) -> Iterator[tuple[ComplementarySignalReframeSpec, Expression]]:
    fields = _complementary_fields(expression, catalog, requested_fields)
    cross_normalizers = _operators(
        catalog,
        roles=_CROSS_SECTIONAL_NORMALIZATION,
        kinds=("expr",),
    )
    temporal_normalizers = _operators(
        catalog,
        roles=_TIME_SERIES_NORMALIZATION,
        kinds=("expr", "window"),
    )
    windows = catalog.window_values()
    identity = f"{COMPLEMENTARY_SIGNAL_REFRAME}:{formula_fingerprint(expression)}"
    for (
        left_normalizer_index,
        temporal_normalizer_index,
        outer_normalizer_index,
        window_index,
        field_index,
    ) in _balanced_product_indexes(
        identity,
        (
            len(cross_normalizers),
            len(temporal_normalizers),
            len(cross_normalizers),
            len(windows),
            len(fields),
        ),
    ):
        left_normalizer = cross_normalizers[left_normalizer_index]
        temporal_normalizer = temporal_normalizers[temporal_normalizer_index]
        outer_normalizer = cross_normalizers[outer_normalizer_index]
        window = windows[window_index]
        field_id = fields[field_index]
        spec = ComplementarySignalReframeSpec(
            left_normalizer=left_normalizer.name,
            temporal_normalizer=temporal_normalizer.name,
            outer_normalizer=outer_normalizer.name,
            window=window,
            complement_field=field_id,
        )
        left = Call(
            left_normalizer.name,
            (CallArgument(expression),),
        )
        right = Call(
            temporal_normalizer.name,
            (
                CallArgument(Name(field_id)),
                CallArgument(_integer(window)),
            ),
        )
        yield (
            spec,
            Call(
                outer_normalizer.name,
                (CallArgument(Binary("+", left, right)),),
            ),
        )


def _complementary_fields(
    expression: Expression,
    catalog: GenerationCatalog,
    requested_fields: tuple[str, ...],
) -> tuple[str, ...]:
    referenced_names = set(analyze_formula(expression).referenced_names)
    available = tuple(
        field
        for field in (
            tuple(catalog.field(field_id) for field_id in requested_fields)
            if requested_fields
            else catalog.fields
        )
        if field is not None
        and field.field_id not in referenced_names
        and field.field_type == "MATRIX"
        and field.coverage is not None
        and field.coverage > 0
    )
    referenced_fields = tuple(
        field
        for name in referenced_names
        if (field := catalog.field(name)) is not None
        and field.field_type in {"MATRIX", "VECTOR"}
    )
    relation_pools = (
        tuple(
            field.field_id
            for field in available
            if any(
                _same_nonempty_relation(field.dataset_id, reference.dataset_id)
                for reference in referenced_fields
            )
        ),
        tuple(
            field.field_id
            for field in available
            if any(
                _same_nonempty_relation(
                    field.subcategory,
                    reference.subcategory,
                )
                for reference in referenced_fields
            )
        ),
        tuple(
            field.field_id
            for field in available
            if any(
                _same_nonempty_relation(field.category, reference.category)
                for reference in referenced_fields
            )
        ),
    )
    return next((tuple(sorted(pool)) for pool in relation_pools if pool), ())


def _same_nonempty_relation(left: str | None, right: str | None) -> bool:
    return (
        left is not None
        and right is not None
        and bool(left.strip())
        and bool(right.strip())
        and left == right
    )


def _prepare_leaf(
    expression: Expression,
    catalog: GenerationCatalog,
    *,
    before: str,
) -> Expression | None:
    validation = validate_formula(expression, catalog)
    if not validation.is_valid or find_coarse_unit_issues(expression, catalog):
        return None
    logic = prepare_formula_logic(expression)
    if logic.issue is not None:
        return None
    if not fits_formula_structure_limits(logic.expression):
        return None
    if render_formula(logic.expression) == before:
        return None
    prepared_validation = validate_formula(logic.expression, catalog)
    if not prepared_validation.is_valid or find_coarse_unit_issues(
        logic.expression, catalog
    ):
        return None
    return logic.expression


def _balanced_product_indexes(
    identity: str,
    lengths: tuple[int, ...],
) -> Iterator[tuple[int, ...]]:
    if not lengths or any(length <= 0 for length in lengths):
        return
    total = prod(lengths)
    starts: list[int] = []
    strides: list[int] = []
    for dimension, length in enumerate(lengths):
        dimension_digest = sha256(f"{identity}:{dimension}".encode("utf-8")).digest()
        starts.append(int.from_bytes(dimension_digest[:8], "big") % length)
        strides.append(
            _coprime_stride(
                length,
                int.from_bytes(dimension_digest[8:16], "big"),
            )
        )

    balanced_count = min(
        total,
        lcm(*lengths),
        MAX_TRANSFORMATION_LEAVES_PER_FAMILY,
    )
    balanced: set[tuple[int, ...]] = set()
    for offset in range(balanced_count):
        indexes = tuple(
            (start + offset * stride) % length
            for start, stride, length in zip(
                starts,
                strides,
                lengths,
                strict=True,
            )
        )
        balanced.add(indexes)
        yield indexes

    for indexes in _permuted_product_indexes(identity, lengths):
        if indexes not in balanced:
            yield indexes


def _permuted_product_indexes(
    identity: str,
    lengths: tuple[int, ...],
) -> Iterator[tuple[int, ...]]:
    total = prod(lengths)
    digest = sha256(identity.encode("utf-8")).digest()
    start = int.from_bytes(digest[:8], "big") % total
    stride = _coprime_stride(total, int.from_bytes(digest[8:16], "big"))
    for offset in range(total):
        flat_index = (start + offset * stride) % total
        indexes: list[int] = []
        for length in reversed(lengths):
            flat_index, index = divmod(flat_index, length)
            indexes.append(index)
        yield tuple(reversed(indexes))


def _coprime_stride(total: int, seed: int) -> int:
    if total == 1:
        return 1
    lower_bound = max(1, total // 2)
    stride = lower_bound + seed % (total - lower_bound)
    while gcd(stride, total) != 1:
        stride += 1
        if stride == total:
            stride = lower_bound
    return stride


def _operators(
    catalog: GenerationCatalog,
    *,
    roles: frozenset[str],
    kinds: tuple[str, ...],
) -> tuple[OperatorDefinition, ...]:
    return tuple(
        operator
        for operator in catalog.operators
        if operator.output_kind == "signal"
        and operator.has_any_role(roles)
        and _matches_signature(operator, kinds)
    )


def _matches_signature(
    operator: OperatorDefinition,
    kinds: tuple[str, ...],
) -> bool:
    if len(operator.parameters) < len(kinds):
        return False
    required = operator.parameters[: len(kinds)]
    remaining = operator.parameters[len(kinds) :]
    return all(
        parameter.kind == kind and not parameter.optional and not parameter.variadic
        for parameter, kind in zip(required, kinds)
    ) and all(parameter.optional for parameter in remaining)


def _validate_request(
    catalog: GenerationCatalog,
    *,
    families: tuple[str, ...],
    field_candidates: tuple[str, ...],
    group_candidates: tuple[str, ...],
) -> None:
    if (
        not isinstance(families, tuple)
        or not families
        or any(
            not isinstance(family, str)
            or family not in STRUCTURAL_TRANSFORMATION_FAMILIES
            for family in families
        )
        or len(set(families)) != len(families)
    ):
        raise ValueError("structural_transformation_families_invalid")
    for candidates, expected_type, error in (
        (
            field_candidates,
            "MATRIX",
            "structural_transformation_field_candidates_invalid",
        ),
        (
            group_candidates,
            "GROUP",
            "structural_transformation_group_candidates_invalid",
        ),
    ):
        if not isinstance(candidates, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in candidates
        ):
            raise ValueError(error)
        if len(set(candidates)) != len(candidates):
            raise ValueError(error)
        for field_id in candidates:
            field = catalog.field(field_id)
            if field is None or field.field_type != expected_type:
                raise ValueError(f"{error}:{field_id}")


def _integer(value: int) -> Literal:
    return Literal(str(value), "integer")
