"""Load the idea variant pool for one cycle without letting profile problems stop a run."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from execution.progress import phase
from generation.catalog import GenerationCatalog
from generation.field_profiles import (
    CatalogFieldInfo,
    FieldProfileSet,
    FieldRelatives,
    load_field_profiles,
    reconcile_field_profiles,
)
from generation.ideas import (
    COMBO_MIN_ABS_SHARPE,
    COMBO_TEMPLATE_ID,
    ComboParent,
    IdeaVariant,
    IdeaVariantPool,
    build_idea_variant_pool,
    plan_combo_variants,
)
from generation.parser import FormulaSyntaxError, parse_formula
from persistence.catalog import FieldCatalogContext, list_field_catalog
from persistence.pnl import list_pnl_series

FIELD_PROFILE_FILE_NAME = "field-profiles.json"
# Hand-reviewed and rule-profiled fields both feed templates; within each template the
# reviewed variants are drawn first (see plan_idea_batch), so rule-tier fields extend the
# pool instead of diluting it. Unclassified fields stay with random exploration.
ENABLED_TIERS = frozenset({"reviewed", "rule"})


def field_profiles_path(database_path: str | Path) -> Path:
    return Path(database_path).parent / FIELD_PROFILE_FILE_NAME


def load_idea_pool(
    connection: sqlite3.Connection,
    database_path: str | Path,
    catalog: GenerationCatalog,
    context: FieldCatalogContext,
    *,
    cycle_number: int,
) -> IdeaVariantPool | None:
    """None means "no templates this cycle": random exploration takes the whole share."""
    loaded = load_enabled_field_profiles(
        connection, database_path, context, cycle_number=cycle_number,
        unavailable_note="本轮模板名额由随机探索补齐",
    )
    if loaded is None:
        return None
    profiles, enabled = loaded
    return build_idea_variant_pool(catalog, profiles, enabled_field_ids=enabled)


def load_enabled_field_profiles(
    connection: sqlite3.Connection,
    database_path: str | Path,
    context: FieldCatalogContext,
    *,
    cycle_number: int,
    unavailable_note: str,
) -> tuple[FieldProfileSet, frozenset[str]] | None:
    """Profiles reconciled with today's catalog, plus the fields enabled for use."""
    try:
        profiles = load_field_profiles(field_profiles_path(database_path))
    except (OSError, ValueError) as exc:
        phase(cycle_number, 1, f"字段画像不可用（{exc}），{unavailable_note}")
        return None
    reconciled = reconcile_field_profiles(profiles, tuple(
        CatalogFieldInfo(record.field_id, record.dataset_id, record.field_type,
                         record.coverage, record.description)
        for record in list_field_catalog(connection, context)
    ))
    if reconciled.missing_field_ids or reconciled.added_field_ids:
        phase(cycle_number, 1,
              f"字段画像与平台目录核对：停用 {len(reconciled.missing_field_ids)} 个已消失字段，"
              f"{len(reconciled.added_field_ids)} 个新字段已按规则画像")
    enabled = frozenset(
        profile.field_id for profile in reconciled.profiles.template_fields()
        if profile.tier in ENABLED_TIERS
    )
    return reconciled.profiles, enabled


def no_field_relatives(field_id: str) -> tuple[str, ...]:
    """Without profiles a mutation swaps no fields rather than random ones."""
    return ()


def field_relatives_from(loaded: tuple[FieldProfileSet, frozenset[str]] | None) -> FieldRelatives | None:
    return FieldRelatives(*loaded) if loaded is not None else None


def load_combo_variants(
    connection: sqlite3.Connection,
    catalog: GenerationCatalog,
    *,
    account_scope: str,
) -> tuple[IdeaVariant, ...]:
    """Combos from this account's completed single-template backtests with daily PnL."""
    rows = connection.execute(
        """SELECT i.template_id, i.family, i.field_ids_json, i.hypothesis,
                  t.formula, t.platform_alpha_id, r.sharpe
           FROM formula_ideas i
           JOIN backtest_tasks t ON t.task_id = i.task_id
           JOIN backtest_results r ON r.task_id = i.task_id
           WHERE t.account_scope = ? AND t.status = 'completed'
             AND t.platform_alpha_id IS NOT NULL AND i.template_id != ?
             AND abs(r.sharpe) >= ?
             -- A parent whose book piles into a few stocks passes that onto every combo:
             -- all 15 failed combos of 2026-10-22 carried such an option parent.
             AND NOT EXISTS (SELECT 1 FROM backtest_checks c
                             WHERE c.task_id = i.task_id AND c.check_name = 'CONCENTRATED_WEIGHT'
                               AND c.status = 'FAIL')""",
        (account_scope, COMBO_TEMPLATE_ID, COMBO_MIN_ABS_SHARPE),
    ).fetchall()
    if len(rows) < 2:
        return ()
    series = {
        record.platform_alpha_id: _daily_changes(record.points)
        for record in list_pnl_series(
            connection,
            platform_alpha_ids=frozenset(row[5] for row in rows),
            account_scope=account_scope,
        )
        if record.points
    }
    parents = []
    for template_id, family, field_ids_json, hypothesis, formula, alpha_id, sharpe in rows:
        daily = series.get(alpha_id)
        if not daily:
            continue
        try:
            expression = parse_formula(formula).expression
        except FormulaSyntaxError:
            continue
        parents.append(ComboParent(
            variant=IdeaVariant(
                template_id=template_id, family=family, hypothesis=hypothesis,
                field_ids=tuple(json.loads(field_ids_json)), window=None, expression=expression,
            ),
            sharpe=float(sharpe),
            daily_pnl=daily,
        ))
    return plan_combo_variants(parents, catalog)


def _daily_changes(points: tuple[tuple[str, float], ...]) -> dict[str, float]:
    """Platform PnL points are cumulative; correlation needs day-to-day changes."""
    return {points[i][0]: points[i][1] - points[i - 1][1] for i in range(1, len(points))}
