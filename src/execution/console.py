"""Project observations for the web adapter, using the research engine's decisions."""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

from execution.qualified_candidates import load_submittable_qualified_candidates, load_submission_opportunity_ids
from execution.console_settings import read_settings
from execution.launch import run_resume_rejection
from execution.seeds import load_signal_frontiers
from execution.submission_queue import list_submission_candidates
from generation.direction import DIRECTION_REVERSAL
from generation.self_correlation import SELF_CORRELATION_REPAIR_FAMILIES
from learning.frontiers import PARENT_ATTEMPT_BUDGET, count_parent_attempts
from persistence.backtests import get_backtest_task, list_backtest_mutations
from persistence.console_research import catalog_page, formula_children_page, formula_research, seed_memberships
from generation.formula import analyze_formula
from generation.parser import parse_formula
from persistence.runs import get_automated_run
from persistence.console import (archive_facts, attempted_task_ids, completed_dates, latest_cycle_backtest_facts, failed_backtest_checks, formula_facts, formula_page,
                                 read_console_database, recent_backtest_facts, run_facts,
                                 run_backtest_counts, submitted_facts, submission_facts)
from submission.formal import assess_formal_check_payload
from worldquant.config import load_worldquant_connection_settings


@dataclass(frozen=True)
class ConsolePaths:
    database: Path
    settings: Path
    run_config: Path
    environment: Path


class ConsoleReader:
    def __init__(self, paths: ConsolePaths):
        self.paths = paths

    @property
    def account_scope(self):
        return load_worldquant_connection_settings(self.paths.environment).account_scope

    def settings(self):
        return read_settings(self.paths)

    def require_selected_submission(self, task_id, source="optimization"):
        from execution.submission_queue import require_selected_submission
        with read_console_database(self.paths.database) as connection:
            require_selected_submission(connection, account_scope=self.account_scope, task_id=task_id, source=source)

    def _facts(self, connection, account, *, task_ids=None, include_check_details=False):
        rows = formula_facts(connection, account, task_ids=task_ids)
        mutations = list_backtest_mutations(connection, parent_task_ids=frozenset(row["task_id"] for row in rows))
        counts = count_parent_attempts(mutations, attempted_task_ids(
            connection, account, tuple(m.child_task_id for m in mutations)))
        submitted_ids = {row["alpha_id"] for row in submitted_facts(connection, account)}
        for row in rows:
            action = row.pop("action")
            row["source"] = _source(action)
            row["attempts_used"] = counts[row["task_id"]]
            row["attempts_remaining"] = max(0, PARENT_ATTEMPT_BUDGET - row["attempts_used"])
            row["detail_available"] = True
            assessment = _assess_check(row)
            if include_check_details:
                row["check_details"] = _full_check_details(row, assessment)
            for key in ("check_payload", "formal_check_at", "formal_check_payload"):
                row.pop(key)
            row["full_check"] = assessment.state
            row["failed_checks"] = list(assessment.failed_checks)
            row["submitted"] = row["alpha_id"] in submitted_ids or row["submission_status"] == "submitted"
        return rows

    def formulas(self, *, category="all", search="", grade="", page=1, page_size=25, run_id="", execution="all", source=""):
        if execution not in {"all", "started", "finished", "inflight", "planned"}:
            raise ValueError("console_execution_filter_invalid")
        if source not in {"", "exploration", "sc", "mutation", "reversal"} or source and category != "optimization":
            raise ValueError("console_source_filter_invalid")
        account = self.account_scope
        with read_console_database(self.paths.database) as connection:
            if run_id:
                self._require_run(connection, account, run_id)
            if category == "all":
                task_ids, total, counts = formula_page(connection, account, run_id=run_id,
                    execution=execution, search=search, grade=grade, page=page, page_size=page_size)
                rows = self._facts(connection, account, task_ids=task_ids)
                checks = failed_backtest_checks(connection, account, task_ids)
                for row in rows:
                    row["backtest_failed_checks"] = checks.get(row["task_id"], [])
                return {"items": rows, "total": total, "page": page, "page_size": page_size,
                        "unstarted": counts.get("planned", 0), "inflight": counts.get("inflight", 0)}
            rows = self._facts(connection, account)
            if category == "optimization":
                ids = set(load_signal_frontiers(connection, optimization_only=True).active_branch_task_ids)
                rows = [row for row in rows if row["task_id"] in ids]
                submittable = set(load_submission_opportunity_ids(
                    connection, account_scope=account, include_optimization=True))
                for row in rows:
                    row["can_submit"] = row["task_id"] in submittable
            elif category == "archive":
                ids = {s.task.task_id for s in load_submittable_qualified_candidates(connection, account_scope=account)}
                rows = [row for row in rows if row["task_id"] in ids]
                for row in rows:
                    row["can_submit"] = True
            elif category == "candidates":
                ids = {item.task_id for item in list_submission_candidates(connection, account_scope=account,
                    source="queue" if grade in {"", "SPECTACULAR"} else "qualified_archive",
                    grade=grade if grade not in {"", "SPECTACULAR"} else None)}
                rows = [row for row in rows if row["task_id"] in ids]
            elif category == "submitted":
                by_alpha = {}
                for row in rows:
                    by_alpha.setdefault(row["alpha_id"], row)
                rows = [{**by_alpha.get(item["alpha_id"], {}), **item,
                         "detail_available": item["alpha_id"] in by_alpha,
                         "task_id": by_alpha.get(item["alpha_id"], {}).get("task_id", item["alpha_id"])}
                        for item in submitted_facts(connection, account)]
            elif category != "all":
                raise ValueError("console_category_invalid")
            if run_id:
                rows = [row for row in rows if row.get("run_id") == run_id]
            groups = {"finished": [], "inflight": [], "planned": []}
            for row in rows:
                group = ("finished" if row.get("status") in {"completed", "failed"} else
                         "planned" if row.get("status") == "created" and row.get("submission_started_at") is None else
                         "inflight")
                groups[group].append(row)
            if execution == "started":
                rows = [row for row in rows if row.get("submission_started_at") is not None]
            elif execution != "all":
                rows = groups[execution]
            if search:
                rows = [row for row in rows if search.lower() in (row.get("alpha_id") or row["task_id"]).lower()]
            if grade:
                rows = [row for row in rows if (row.get("grade") or "UNKNOWN") == grade]
            if source:
                rows = [row for row in rows if row.get("source") == source]
            result = _page(rows, page, page_size)
            result["unstarted"] = len(groups["planned"])
            result["inflight"] = len(groups["inflight"])
            checks = failed_backtest_checks(connection, account, tuple(row["task_id"] for row in result["items"]))
            for row in result["items"]:
                row["backtest_failed_checks"] = checks.get(row["task_id"], [])
            return result

    def runs(self, *, page=1, page_size=25):
        account = self.account_scope
        with read_console_database(self.paths.database) as connection:
            result = _page(run_facts(connection, account), page, page_size)
            rows, counts = result["items"], run_backtest_counts(connection, account)
            for row in rows:
                row.update(self._resume_fields(connection, row["run_id"]))
        for row in rows:
            row.update(counts.get(row["run_id"], {"planned": 0, "finished": 0, "completed": 0}))
            if row["stop_reason"] == "user_paused":
                row["status"] = "paused"
        return result

    def catalog(self, *, kind, search="", category="", dataset="", page=1, page_size=10):
        if kind not in {"fields", "operators"}:
            raise ValueError("console_catalog_kind_invalid")
        with read_console_database(self.paths.database) as connection:
            return catalog_page(connection, self.account_scope, kind=kind, search=search,
                                category=category, dataset=dataset, page=page, page_size=page_size)

    def formula_detail(self, task_id):
        account = self.account_scope
        with read_console_database(self.paths.database) as connection:
            detail = formula_research(connection, account, task_id)
            snapshot = get_backtest_task(connection, task_id)
            row = self._facts(connection, account, task_ids=(task_id,), include_check_details=True)[0]
            row["backtest_failed_checks"] = failed_backtest_checks(connection, account, (task_id,)).get(task_id, [])
            detail["summary"] = row
            detail["result"] = asdict(snapshot.result) if snapshot.result is not None else None
            detail["checks"] = row.pop("check_details") or {
                "source": "backtest", "observed_at": snapshot.task.finished_at, "error": None,
                "items": [asdict(check) for check in snapshot.result.checks] if snapshot.result else [],
            }
            detail["yearly"] = [asdict(item) for item in snapshot.yearly_stats] if snapshot.yearly_stats is not None else None
        facts = analyze_formula(parse_formula(detail["expression"]).expression)
        detail["references"] = {"names": sorted(set(facts.referenced_names)), "operators": sorted(set(facts.operator_names))}
        return detail

    def formula_children(self, task_id, *, page=1, page_size=10):
        account = self.account_scope
        with read_console_database(self.paths.database) as connection:
            task_ids, total = formula_children_page(connection, account, task_id, page=page, page_size=page_size)
            items = self._facts(connection, account, task_ids=task_ids)
        return {"items": items, "total": total, "page": page, "page_size": page_size}

    def seeds(self, *, mode="roots", search="", page=1, page_size=10):
        if mode not in {"roots", "normal"}:
            raise ValueError("console_seed_mode_invalid")
        account = self.account_scope
        with read_console_database(self.paths.database) as connection:
            memberships = seed_memberships(connection, account)
            roots = {row["root_task_id"]: row for row in memberships}
            if mode == "roots":
                entries = [{**row, "task_id": row["root_task_id"]} for row in memberships]
            else:
                frontiers = load_signal_frontiers(connection)
                entries = [{**roots[branch.root_task_id], "task_id": branch.task_id,
                            "remaining_attempts": branch.remaining_attempts}
                           for frontier in frontiers.records for branch in frontier.branches
                           if branch.account_scope == account and branch.root_task_id in roots]
            ids = tuple(dict.fromkeys(entry["task_id"] for entry in entries))
            facts = {row["task_id"]: row for row in self._facts(connection, account, task_ids=ids)}
            rows = [{**facts[entry["task_id"]], **entry} for entry in entries]
            for row in rows:
                if "remaining_attempts" in row:
                    row["attempts_remaining"] = row.pop("remaining_attempts")
            if search:
                rows = [row for row in rows if search.lower() in
                        f"{row['alpha_id'] or row['task_id']} {row['root_alpha_id'] or row['root_task_id']}".lower()]
            result = _page(rows, page, page_size)
            result["root_count"] = len(roots)
            return result

    def require_run(self, run_id):
        with read_console_database(self.paths.database) as connection:
            row = self._require_run(connection, self.account_scope, run_id)
            return {**row, **self._resume_fields(connection, run_id)}

    @staticmethod
    def _resume_fields(connection, run_id):
        rejection = run_resume_rejection(connection, get_automated_run(connection, run_id))
        return {"can_resume": rejection is None, "resume_blocked_reason": rejection}

    @staticmethod
    def _require_run(connection, account, run_id):
        run = next((row for row in run_facts(connection, account) if row["run_id"] == run_id), None)
        if run is None:
            raise ValueError("console_run_not_found")
        return run

    def submissions(self, *, page=1, page_size=25):
        with read_console_database(self.paths.database) as connection:
            return _page(submission_facts(connection, self.account_scope), page, page_size)

    def dashboard_activity(self):
        account = self.account_scope
        now = datetime.now().astimezone()
        dates = [(now.date() - timedelta(days=14-i)).isoformat() for i in range(15)]
        since = (now - timedelta(days=14)).replace(hour=0, minute=0, second=0, microsecond=0)
        with read_console_database(self.paths.database) as connection:
            finished = completed_dates(connection, account, since=since.isoformat())
            submitted = submitted_facts(connection, account)
            archive = archive_facts(connection, account)
            archive_ids = {s.task.task_id for s in load_submittable_qualified_candidates(
                connection, account_scope=account)}
            queue = list_submission_candidates(connection, account_scope=account)
        def daily(values):
            counts = Counter(datetime.fromisoformat(value).astimezone().date().isoformat() for value in values if value)
            return [counts[day] for day in dates]
        trend = daily(finished)
        submitted_trend = daily(row["date_submitted"] for row in submitted)
        return {
            "observed_at": now.isoformat(), "dates": dates, "trend": trend,
            "weekly_backtests": sum(trend[-7:]), "weekly_submissions": sum(submitted_trend[-7:]),
            "metrics": [
                {"value": trend[-1], "spark": trend[-7:]},
                {"value": len(queue), "spark": daily(item.enqueued_at for item in queue)[-7:]},
                {"value": len(archive_ids), "spark": daily(row["archived_at"] for row in archive if row["task_id"] in archive_ids)[-7:]},
                {"value": len(submitted), "spark": submitted_trend[-7:]},
            ],
        }

    def dashboard_grades(self):
        with read_console_database(self.paths.database) as connection:
            counts = Counter(row["grade"] or "UNKNOWN" for row in submitted_facts(connection, self.account_scope))
        return {"grades": [{"label": grade, "value": counts[grade]} for grade in
                         ("SPECTACULAR", "EXCELLENT", "GOOD", "AVERAGE", "INFERIOR", "UNKNOWN") if counts[grade]]}

    def dashboard_research(self):
        account = self.account_scope
        with read_console_database(self.paths.database) as connection:
            frontiers = load_signal_frontiers(connection, optimization_only=True)
            branches = [branch for frontier in frontiers.records for branch in frontier.branches
                        if branch.account_scope == account]
            manual = list_submission_candidates(connection, account_scope=account, source="qualified_archive")
        return {"observed_at": datetime.now().astimezone().isoformat(),
                "optimization_parents": len(branches),
                "optimization_attempts": sum(branch.remaining_attempts for branch in branches),
                "manual_candidates": len(manual)}

    def dashboard_progress(self):
        account = self.account_scope
        with read_console_database(self.paths.database) as connection:
            runs = run_facts(connection, account, limit=1)
            latest = runs[0] if runs else None
            # current_cycle counts settlements, not the latest persisted batch.
            cycle = latest_cycle_backtest_facts(connection, account, latest["run_id"]) if latest else []
        planned = Counter(_source(row["action"]) for row in cycle)
        completed = Counter(_source(row["action"]) for row in cycle if row["status"] in {"completed", "failed"})
        return {"run": latest, "cycle_number": cycle[0]["cycle_number"] if cycle else None, "progress": [
            {"source": source, "completed": completed[source], "planned": planned[source]}
            for source in ("exploration", "sc", "mutation", "reversal")]}

    def dashboard_recent(self):
        with read_console_database(self.paths.database) as connection:
            return {"items": recent_backtest_facts(connection, self.account_scope)}


def _source(action):
    return ("exploration" if action is None else "reversal" if action == DIRECTION_REVERSAL
            else "sc" if action in SELF_CORRELATION_REPAIR_FAMILIES else "mutation")


def _assess_check(row):
    """Choose the newest recorded formal check, as in the formula list."""
    at, payload = row["check_at"], row["check_payload"]
    formal_at = row["formal_check_at"]
    if formal_at and (not at or datetime.fromisoformat(formal_at) > datetime.fromisoformat(at)):
        row["check_at"], payload, row["check_error"] = formal_at, row["formal_check_payload"], None
    row["check_payload"] = payload
    return assess_formal_check_payload(json.loads(payload) if payload else None)


def _full_check_details(row, assessment):
    if not row["check_at"]:
        return None
    payload = json.loads(row["check_payload"]) if row["check_payload"] else None
    # Expose only assessed check fields, never the full payload or an older pass
    # after a newer pending/error observation.
    checks = {check["name"].strip(): check for check in payload["is"]["checks"]} if assessment.statuses else {}
    def number(value):
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None
    return {"source": "full", "observed_at": row["check_at"], "error": row["check_error"],
            "items": [{"name": name, "status": status,
                       "actual": number(checks[name].get("value")), "threshold": number(checks[name].get("limit"))}
                      for name, status in assessment.statuses]}


def _page(rows, page, page_size):
    return {"items": rows[(page-1)*page_size:page*page_size], "total": len(rows), "page": page, "page_size": page_size}
