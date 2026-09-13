from __future__ import annotations

import sqlite3
from collections import Counter
from dataclasses import dataclass

from generation.catalog import GenerationCatalog
from generation.candidate import FormulaCandidate, exploration_candidate
from generation.exploration import (
    ExplorationFailure,
    ExplorationResult,
    explore_formula,
)
from generation.formula import Expression, formula_fingerprint
from generation.parser import FormulaSyntaxError, parse_formula
from persistence.backtests import list_backtest_formulas


@dataclass(frozen=True, slots=True)
class ExplorationCandidate:
    candidate: FormulaCandidate
    seed: int

    @property
    def expression(self) -> Expression:
        return self.candidate.expression


@dataclass(frozen=True, slots=True)
class ExclusionCount:
    code: str
    count: int


@dataclass(frozen=True, slots=True)
class BatchShortfall:
    missing_count: int


@dataclass(frozen=True, slots=True)
class ExplorationBatch:
    candidates: tuple[ExplorationCandidate, ...]
    attempted_seed_count: int
    exclusions: tuple[ExclusionCount, ...]
    shortfall: BatchShortfall | None


def generate_unsubmitted_exploration_batch(
    connection: sqlite3.Connection,
    catalog: GenerationCatalog,
    *,
    target_count: int,
    seeds: tuple[int, ...],
    field_candidates: tuple[str, ...],
    group_candidates: tuple[str, ...] = (),
    operator_candidates: tuple[str, ...] = (),
) -> ExplorationBatch:
    """Generate in memory and exclude only formulas reserved for real backtests."""
    return generate_exploration_batch(
        catalog,
        target_count=target_count,
        seeds=seeds,
        field_candidates=field_candidates,
        group_candidates=group_candidates,
        operator_candidates=operator_candidates,
        existing_formulas=list_backtest_formulas(connection),
    )


def generate_exploration_batch(
    catalog: GenerationCatalog,
    *,
    target_count: int,
    seeds: tuple[int, ...],
    field_candidates: tuple[str, ...],
    group_candidates: tuple[str, ...] = (),
    operator_candidates: tuple[str, ...] = (),
    existing_formulas: tuple[str, ...] = (),
) -> ExplorationBatch:
    _validate_batch_request(
        target_count=target_count,
        seeds=seeds,
        existing_formulas=existing_formulas,
    )
    if target_count == 0:
        return ExplorationBatch(
            candidates=(),
            attempted_seed_count=0,
            exclusions=(),
            shortfall=None,
        )

    existing_fingerprints = _existing_fingerprints(existing_formulas)
    batch_fingerprints: set[str] = set()
    candidates: list[ExplorationCandidate] = []
    exclusions: Counter[str] = Counter()
    attempted_seed_count = 0

    for seed in seeds:
        if len(candidates) == target_count:
            break
        attempted_seed_count += 1
        outcome = explore_formula(
            catalog,
            seed=seed,
            field_candidates=field_candidates,
            group_candidates=group_candidates,
            operator_candidates=operator_candidates,
        )
        if isinstance(outcome, ExplorationFailure):
            exclusions[f"exploration_failure:{outcome.code}"] += 1
            continue
        assert isinstance(outcome, ExplorationResult)
        fingerprint = formula_fingerprint(outcome.expression)
        if fingerprint in existing_fingerprints:
            exclusions["backtest_formula_duplicate"] += 1
            continue
        if fingerprint in batch_fingerprints:
            exclusions["batch_formula_duplicate"] += 1
            continue
        batch_fingerprints.add(fingerprint)
        candidates.append(
            ExplorationCandidate(
                candidate=exploration_candidate(outcome.expression),
                seed=seed,
            )
        )

    exclusion_counts = tuple(
        ExclusionCount(code=code, count=count)
        for code, count in sorted(exclusions.items())
    )
    shortfall = None
    if len(candidates) < target_count:
        shortfall = BatchShortfall(
            missing_count=target_count - len(candidates),
        )
    return ExplorationBatch(
        candidates=tuple(candidates),
        attempted_seed_count=attempted_seed_count,
        exclusions=exclusion_counts,
        shortfall=shortfall,
    )


def _validate_batch_request(
    *,
    target_count: int,
    seeds: tuple[int, ...],
    existing_formulas: tuple[str, ...],
) -> None:
    if (
        isinstance(target_count, bool)
        or not isinstance(target_count, int)
        or target_count < 0
    ):
        raise ValueError("batch_target_count_invalid")
    if not isinstance(seeds, tuple) or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds
    ):
        raise ValueError("batch_seeds_invalid")
    if len(set(seeds)) != len(seeds):
        raise ValueError("batch_seeds_duplicate")
    _validate_existing_formulas(existing_formulas)


def _validate_existing_formulas(existing_formulas: tuple[str, ...]) -> None:
    if not isinstance(existing_formulas, tuple) or any(
        not isinstance(formula, str) for formula in existing_formulas
    ):
        raise ValueError("batch_existing_formulas_invalid")


def _existing_fingerprints(formulas: tuple[str, ...]) -> frozenset[str]:
    fingerprints: set[str] = set()
    for index, formula in enumerate(formulas):
        try:
            fingerprints.add(parse_formula(formula).fingerprint)
        except FormulaSyntaxError as exc:
            raise ValueError(
                f"batch_existing_formula_invalid:{index}:{exc.code}"
            ) from exc
    return frozenset(fingerprints)
