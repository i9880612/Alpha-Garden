from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256


def explore_internal_fields(
    fields: tuple[str, ...], *, rotation_key: str
) -> tuple[str, ...]:
    """Bound local enumeration, rotating uniformly; this is not a quality score."""
    return tuple(
        sorted(
            set(fields),
            key=lambda name: sha256(f"{rotation_key}|{name}".encode()).digest(),
        )[:64]
    )


SELECTION_POLICY = "family_unique_diversity_coverage"


@dataclass(frozen=True, slots=True)
class DiversityDimension:
    name: str
    values: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SelectionCandidate:
    candidate_id: str
    family_root_candidate_id: str
    diversity: tuple[DiversityDimension, ...] = ()
    rejection_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SelectedCandidate:
    candidate_id: str
    family_root_candidate_id: str
    representative_reason: str


@dataclass(frozen=True, slots=True)
class RejectedCandidate:
    candidate_id: str
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CandidateSelection:
    requested_count: int
    selected: tuple[SelectedCandidate, ...]
    rejected: tuple[RejectedCandidate, ...]

    @property
    def missing_count(self) -> int:
        return self.requested_count - len(self.selected)


def select_candidates(
    candidates: tuple[SelectionCandidate, ...],
    *,
    requested_count: int,
    family_representatives: tuple[tuple[str, str], ...] = (),
) -> CandidateSelection:
    if (
        isinstance(requested_count, bool)
        or not isinstance(requested_count, int)
        or requested_count <= 0
    ):
        raise ValueError("selection_requested_count_invalid")
    if not isinstance(candidates, tuple):
        raise ValueError("selection_candidates_invalid")

    candidate_by_id: dict[str, SelectionCandidate] = {}
    families: dict[str, list[SelectionCandidate]] = {}
    rejected: list[RejectedCandidate] = []
    for candidate in candidates:
        if not isinstance(candidate, SelectionCandidate):
            raise ValueError("selection_candidate_invalid")
        _require_text(candidate.candidate_id, "selection_candidate_id_missing")
        _require_text(
            candidate.family_root_candidate_id,
            "selection_family_root_missing",
        )
        if candidate.candidate_id in candidate_by_id:
            raise ValueError("selection_candidate_duplicate")
        _validate_diversity(candidate.diversity)
        _validate_rejection_reasons(candidate.rejection_reasons)
        candidate_by_id[candidate.candidate_id] = candidate
        if candidate.rejection_reasons:
            rejected.append(
                RejectedCandidate(
                    candidate_id=candidate.candidate_id,
                    reasons=candidate.rejection_reasons,
                )
            )
            continue
        families.setdefault(candidate.family_root_candidate_id, []).append(candidate)

    representatives = _validate_representatives(
        family_representatives,
        candidate_by_id=candidate_by_id,
        families=families,
    )
    family_choices: list[tuple[SelectedCandidate, SelectionCandidate]] = []
    for family_root_candidate_id, members in families.items():
        representative_id = representatives.get(family_root_candidate_id)
        if representative_id is None:
            if len(members) != 1:
                raise ValueError("selection_family_representative_required")
            representative = members[0]
            reason = "only_family_member"
        else:
            representative = candidate_by_id[representative_id]
            reason = "explicit_family_representative"
        family_choices.append(
            (
                SelectedCandidate(
                    candidate_id=representative.candidate_id,
                    family_root_candidate_id=family_root_candidate_id,
                    representative_reason=reason,
                ),
                representative,
            )
        )

    return CandidateSelection(
        requested_count=requested_count,
        selected=_select_diverse(family_choices, requested_count=requested_count),
        rejected=tuple(sorted(rejected, key=lambda item: item.candidate_id)),
    )


def _select_diverse(
    choices: list[tuple[SelectedCandidate, SelectionCandidate]],
    *,
    requested_count: int,
) -> tuple[SelectedCandidate, ...]:
    remaining = list(choices)
    coverage_counts: dict[str, Counter[str]] = {}
    balanced_values = _balanced_dimension_values(
        [source for _, source in choices]
    )
    selected: list[SelectedCandidate] = []
    while remaining and len(selected) < requested_count:
        ranked = sorted(
            remaining,
            key=lambda choice: (
                *_balance_spread_after(
                    choice[1].diversity,
                    coverage_counts,
                    balanced_values,
                ),
                -_coverage_gain(choice[1].diversity, coverage_counts),
                choice[0].candidate_id,
            ),
        )
        selected_item, source = ranked[0]
        selected.append(selected_item)
        for dimension in source.diversity:
            coverage_counts.setdefault(dimension.name, Counter()).update(
                dimension.values
            )
        remaining.remove(ranked[0])
    return tuple(selected)


def _balanced_dimension_values(
    candidates: list[SelectionCandidate],
) -> dict[str, frozenset[str]]:
    if not candidates:
        return {}
    dimensions_by_candidate = [
        {dimension.name: dimension for dimension in candidate.diversity}
        for candidate in candidates
    ]
    common_names = set(dimensions_by_candidate[0])
    for dimensions in dimensions_by_candidate[1:]:
        common_names.intersection_update(dimensions)

    balanced: dict[str, frozenset[str]] = {}
    for name in sorted(common_names):
        dimensions = [item[name] for item in dimensions_by_candidate]
        if any(len(dimension.values) != 1 for dimension in dimensions):
            continue
        values = frozenset(
            dimension.values[0] for dimension in dimensions
        )
        if len(values) > 1:
            balanced[name] = values
    return balanced


def _balance_spread_after(
    dimensions: tuple[DiversityDimension, ...],
    coverage_counts: dict[str, Counter[str]],
    balanced_values: dict[str, frozenset[str]],
) -> tuple[int, int]:
    candidate_values = {
        dimension.name: frozenset(dimension.values)
        for dimension in dimensions
    }
    spreads: list[int] = []
    for name, values in balanced_values.items():
        counts = coverage_counts.get(name, Counter())
        selected_values = candidate_values.get(name, frozenset())
        projected = tuple(
            counts[value] + int(value in selected_values)
            for value in values
        )
        spreads.append(max(projected) - min(projected))
    return max(spreads, default=0), sum(spreads)


def _coverage_gain(
    dimensions: tuple[DiversityDimension, ...],
    coverage_counts: dict[str, Counter[str]],
) -> Fraction:
    gains = (
        sum(
            Fraction(
                1,
                coverage_counts.get(dimension.name, Counter())[value] + 1,
            )
            for value in dimension.values
        )
        / len(dimension.values)
        for dimension in dimensions
    )
    return sum(gains, Fraction())


def _validate_diversity(dimensions: tuple[DiversityDimension, ...]) -> None:
    if not isinstance(dimensions, tuple):
        raise ValueError("selection_diversity_invalid")
    names: set[str] = set()
    for dimension in dimensions:
        if not isinstance(dimension, DiversityDimension):
            raise ValueError("selection_diversity_dimension_invalid")
        _require_text(dimension.name, "selection_diversity_name_missing")
        if dimension.name in names:
            raise ValueError("selection_diversity_dimension_duplicate")
        names.add(dimension.name)
        if (
            not isinstance(dimension.values, tuple)
            or not dimension.values
            or any(
                not isinstance(value, str) or not value.strip()
                for value in dimension.values
            )
            or len(set(dimension.values)) != len(dimension.values)
        ):
            raise ValueError("selection_diversity_values_invalid")


def _validate_rejection_reasons(reasons: tuple[str, ...]) -> None:
    if (
        not isinstance(reasons, tuple)
        or any(not isinstance(reason, str) or not reason.strip() for reason in reasons)
        or len(set(reasons)) != len(reasons)
    ):
        raise ValueError("selection_rejection_reasons_invalid")


def _validate_representatives(
    family_representatives: tuple[tuple[str, str], ...],
    *,
    candidate_by_id: dict[str, SelectionCandidate],
    families: dict[str, list[SelectionCandidate]],
) -> dict[str, str]:
    if not isinstance(family_representatives, tuple):
        raise ValueError("selection_family_representatives_invalid")
    representatives: dict[str, str] = {}
    for item in family_representatives:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError("selection_family_representative_invalid")
        family_root_candidate_id, candidate_id = item
        _require_text(
            family_root_candidate_id,
            "selection_family_root_missing",
        )
        _require_text(candidate_id, "selection_candidate_id_missing")
        if family_root_candidate_id in representatives:
            raise ValueError("selection_family_representative_duplicate")
        candidate = candidate_by_id.get(candidate_id)
        if (
            family_root_candidate_id not in families
            or candidate is None
            or candidate.family_root_candidate_id != family_root_candidate_id
        ):
            raise ValueError("selection_family_representative_mismatch")
        representatives[family_root_candidate_id] = candidate_id
    return representatives


def _require_text(value: object, error: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(error)
