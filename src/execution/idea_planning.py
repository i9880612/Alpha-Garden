"""Load the idea variant pool for one cycle without letting profile problems stop a run."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from execution.progress import phase
from generation.catalog import GenerationCatalog
from generation.field_profiles import (
    CatalogFieldInfo,
    load_field_profiles,
    reconcile_field_profiles,
)
from generation.ideas import IdeaVariantPool, build_idea_variant_pool
from persistence.catalog import FieldCatalogContext, list_field_catalog

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
    try:
        profiles = load_field_profiles(field_profiles_path(database_path))
    except (OSError, ValueError) as exc:
        phase(cycle_number, 1, f"字段画像不可用（{exc}），本轮模板名额由随机探索补齐")
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
    return build_idea_variant_pool(catalog, reconciled.profiles, enabled_field_ids=enabled)
