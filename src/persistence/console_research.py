"""Scoped, read-only projections for catalog browsing and research inspection."""
from __future__ import annotations

import json
from dataclasses import asdict

from persistence.catalog import get_platform_catalog_sync


def catalog_page(connection, account, *, kind, search, category, dataset, page, page_size):
    sync = get_platform_catalog_sync(connection)
    available = sync is not None and sync.account_scope == account
    result = {"items": [], "total": 0, "page": page, "page_size": page_size,
              "available": available, "context": asdict(sync.context) if available else None,
              "synced_at": sync.synced_at if available else None, "categories": [], "datasets": []}
    if not available:
        return result
    fields = kind == "fields"
    table, identity = ("platform_fields", "field_id") if fields else ("platform_operators", "operator_name")
    base, parameters = "1=1", []
    if fields:
        base = "instrument_type=? AND region=? AND universe=? AND delay=?"
        parameters = list(asdict(sync.context).values())
    result["categories"] = [row[0] for row in connection.execute(
        f"SELECT DISTINCT category FROM {table} WHERE {base} AND category IS NOT NULL ORDER BY category", parameters)]
    if fields:
        result["datasets"] = [dict(row) for row in connection.execute(
            f"SELECT DISTINCT dataset_id, dataset_name FROM {table} WHERE {base} AND dataset_id IS NOT NULL ORDER BY dataset_id", parameters)]
    where = base
    if search:
        where += f" AND instr(lower({identity} || ' ' || COALESCE(description,'')),?)>0"
        parameters.append(search.lower())
    if category:
        where += " AND category=?"
        parameters.append(category)
    if fields and dataset:
        where += " AND dataset_id=?"
        parameters.append(dataset)
    result["total"] = connection.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}", parameters).fetchone()[0]
    columns = ("field_id, dataset_id, dataset_name, category, subcategory, field_type, coverage, description" if fields else
               "operator_name, category, definition, description, documentation, level, scope_json, parameters_json")
    rows = [dict(row) for row in connection.execute(
        f"SELECT {columns} FROM {table} WHERE {where} ORDER BY {identity} LIMIT ? OFFSET ?",
        (*parameters, page_size, (page - 1) * page_size))]
    if not fields:
        for row in rows:
            row["scope"] = json.loads(row.pop("scope_json"))
            row["parameters"] = json.loads(row.pop("parameters_json"))
            row["roles"] = [r[0] for r in connection.execute(
                "SELECT role FROM generation_operator_roles WHERE operator_name=? ORDER BY role", (row["operator_name"],))]
    result["items"] = rows
    return result


def formula_research(connection, account, task_id):
    task = connection.execute("""SELECT formula, settings_json FROM backtest_tasks
        WHERE account_scope=? AND task_id=?""", (account, task_id)).fetchone()
    if task is None:
        raise ValueError("console_formula_not_found")
    settings = json.loads(task["settings_json"])
    # Only research settings are exposed, never arbitrary platform response payloads.
    public_settings = {key: value for key, value in settings.items() if key in {
        "instrumentType", "region", "universe", "delay", "decay", "neutralization", "truncation",
        "pasteurization", "unitHandling", "nanHandling", "language", "visualization", "testPeriod"}}
    mutation = connection.execute("""SELECT m.action,m.location,m.before,m.after
        FROM backtest_mutations m JOIN backtest_tasks p ON p.task_id=m.parent_task_id
        WHERE m.child_task_id=? AND p.account_scope=?""", (task_id, account)).fetchone()
    seed = connection.execute("SELECT promoted_at FROM signal_seeds WHERE root_task_id=?", (task_id,)).fetchone()
    pnl = connection.execute("""SELECT p.observed_at,p.points_json FROM platform_pnl_series p
        JOIN backtest_tasks t ON t.platform_alpha_id=p.platform_alpha_id AND t.account_scope=p.account_scope
        WHERE t.task_id=? AND t.account_scope=?""", (task_id, account)).fetchone()
    return {"expression": task["formula"], "settings": public_settings,
            "mutation": dict(mutation) if mutation else None,
            "seed_promoted_at": seed[0] if seed else None,
            "pnl": {"observed_at": pnl[0], "points": json.loads(pnl[1]) if pnl[1] else None} if pnl else None}


def formula_children_page(connection, account, task_id, *, page, page_size):
    if connection.execute("SELECT 1 FROM backtest_tasks WHERE account_scope=? AND task_id=?",
                          (account, task_id)).fetchone() is None:
        raise ValueError("console_formula_not_found")
    source = """FROM backtest_mutations m JOIN backtest_tasks t ON t.task_id=m.child_task_id
        WHERE m.parent_task_id=? AND t.account_scope=?"""
    total = connection.execute(f"SELECT COUNT(*) {source}", (task_id, account)).fetchone()[0]
    children = tuple(row[0] for row in connection.execute(f"""SELECT t.task_id {source}
        ORDER BY COALESCE(t.finished_at,t.created_at) DESC,t.task_id LIMIT ? OFFSET ?""",
        (task_id, account, page_size, (page - 1) * page_size)))
    return children, total


def seed_memberships(connection, account):
    return [dict(row) for row in connection.execute("""
        SELECT s.root_task_id,s.promoted_at,t.platform_alpha_id AS root_alpha_id
        FROM signal_seeds s JOIN backtest_tasks t ON t.task_id=s.root_task_id
        WHERE t.account_scope=? ORDER BY s.promoted_at DESC,s.root_task_id
    """, (account,))]
