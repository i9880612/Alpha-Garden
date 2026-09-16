"""Read-only diagnostic projections; research and submission retain their decisions."""
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timedelta
import hashlib
import json
import math

from evaluation.correlation import CORRELATION_CUTOFF, MIN_CORRELATION_INTERVALS, correlation_check, submitted_sharpe
from execution.console import _assess_check, _full_check_details, _source
from learning.pnl import PnlCorrelations
from learning.seed_correlation import assess_seed_correlation
from persistence.backtests import list_completed_backtests
from persistence.console import read_console_database
from persistence.console_analysis import quality_checks, quality_facts
from persistence.pnl import list_pnl_series
from persistence.submissions import list_platform_submitted_alphas
from worldquant.backtests import STANDARD_NON_SC_CHECK_NAMES


def finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def check_state(check):
    if check is None:
        return "missing"
    return {"PASS": "passed", "FAIL": "failed"}.get(check["status"].upper(), "pending")


class QualityDiagnosis:
    def __init__(self, reader):
        self.reader = reader

    def _scope(self, connection, *, days="30", mode="all", run_id="", source="", configuration=""):
        if days not in {"7", "15", "30", "all"} or mode not in {"all", "normal", "optimization", "unassigned"}:
            raise ValueError("console_analysis_filter_invalid")
        if source not in {"", "exploration", "mutation", "sc", "reversal"} or len(run_id) > 128:
            raise ValueError("console_analysis_filter_invalid")
        if configuration and (len(configuration) != 64 or any(c not in "0123456789abcdef" for c in configuration)):
            raise ValueError("console_analysis_filter_invalid")
        now = datetime.now().astimezone()
        tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        since = None if days == "all" else (tomorrow - timedelta(days=int(days))).isoformat()
        rows = quality_facts(connection, self.reader.account_scope, since=since, until=tomorrow.isoformat())
        runs = sorted({r["run_id"] for r in rows if r["run_id"]})
        configurations = {}
        selected = []
        for row in rows:
            row["source"] = _source(row.pop("action"))
            row_mode = "unassigned" if row["optimization_only"] is None else "optimization" if row["optimization_only"] else "normal"
            if ((mode != "all" and row_mode != mode) or (run_id and row["run_id"] != run_id)
                    or (source and row["source"] != source)):
                continue
            settings = json.loads(row["settings_json"])
            key = hashlib.sha256(json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            row["configuration"] = key
            if key not in configurations:
                configurations[key] = {"id": key, "count": 0, "settings": settings}
            configurations[key]["count"] += 1
            selected.append(row)
        options = sorted(configurations.values(), key=lambda c: (-c["count"], c["id"]))
        current = configuration or (options[0]["id"] if options else "")
        return [r for r in selected if r["configuration"] == current], {
            "configurations": options, "configuration": current, "runs": runs,
            "since": since, "until": tomorrow.isoformat(),
        }

    def quality(self, **filters):
        with read_console_database(self.reader.paths.database) as connection:
            rows, scope = self._scope(connection, **filters)
            completed = [r for r in rows if r["status"] == "completed"]
            checks = quality_checks(connection, self.reader.account_scope, [r["task_id"] for r in completed])
        records, thresholds = [], set()
        threshold_count = 0
        reasons = {name: Counter() for name in STANDARD_NON_SC_CHECK_NAMES}
        platform = Counter()
        for row in completed:
            # Latest formal observation wins, including pending/error; never revive an older pass.
            assessment = _assess_check(row)
            formal = _full_check_details(row, assessment)
            effective = formal["items"] if formal is not None else checks[row["task_id"]]
            by_name = {c["name"]: c for c in effective}
            sc = "error" if formal and formal["error"] else check_state(by_name.get("SELF_CORRELATION"))
            platform[sc] += 1
            for name, counts in reasons.items():
                counts[check_state(by_name.get(name))] += 1
            sharpe, fitness = by_name.get("LOW_SHARPE"), by_name.get("LOW_FITNESS")
            states = (check_state(sharpe), check_state(fitness))
            group = {("passed", "passed"): "both", ("passed", "failed"): "fitness",
                     ("failed", "passed"): "sharpe", ("failed", "failed"): "neither"}.get(states, "unknown")
            if sharpe and fitness and finite(sharpe["threshold"]) and finite(fitness["threshold"]):
                thresholds.add((sharpe["threshold"], fitness["threshold"]))
                threshold_count += 1
            records.append({k: row[k] for k in ("task_id", "alpha_id", "grade", "source", "sharpe", "fitness", "turnover", "finished_at")} | {
                "metric_group": group, "platform_state": sc,
                "failed_checks": [name for name, c in by_name.items() if check_state(c) == "failed"],
            })
        failure_rows = [{"name": name, "failed": counts["failed"], "passed": counts["passed"],
                         "unresolved": counts["pending"] + counts["missing"]}
                        for name, counts in reasons.items()]
        return {**scope, "records": records, "total": len(rows), "completed": len(completed),
                "inflight": sum(r["status"] in {"created", "pending", "submission_unknown"} for r in rows),
                "errors": sum(r["status"] == "failed" for r in rows),
                "analyzable": sum(finite(r["sharpe"]) and finite(r["fitness"]) for r in completed),
                "thresholds": list(next(iter(thresholds))) if len(thresholds) == 1 and threshold_count == len(completed) else None,
                "reasons": sorted(failure_rows, key=lambda r: (-r["failed"], r["name"])),
                "platform": dict(platform)}

    def correlation(self, **filters):
        with read_console_database(self.reader.paths.database) as connection:
            rows, scope = self._scope(connection, **filters)
            ids = frozenset(r["task_id"] for r in rows if r["status"] == "completed")
            snapshots = list_completed_backtests(connection, task_ids=ids)
            references = list_platform_submitted_alphas(connection, account_scope=self.reader.account_scope)
            alphas = frozenset(s.task.platform_alpha_id for s in snapshots if s.task.platform_alpha_id)
            series = list_pnl_series(connection, account_scope=self.reader.account_scope,
                                    platform_alpha_ids=alphas | frozenset(r.platform_alpha_id for r in references))
        correlations = PnlCorrelations(series)
        records = []
        for snapshot in snapshots:
            assessment = assess_seed_correlation(snapshot, references, series, correlations=correlations)
            evidence = asdict(assessment)
            high_pairs, improved_pairs = 0, 0
            if assessment.state != "submitted":
                for ref in references:
                    value = correlations.correlation((snapshot.task.account_scope, snapshot.task.platform_alpha_id),
                        (ref.account_scope, ref.platform_alpha_id), minimum_intervals=MIN_CORRELATION_INTERVALS)
                    if value is not None and value >= CORRELATION_CUTOFF:
                        high_pairs += 1
                        improved_pairs += correlation_check(value, snapshot.result.sharpe, submitted_sharpe(ref.raw_payload)) == "passed"
            # No submitted references is not evidence of low correlation.
            evidence["state"] = "no_references" if not references else assessment.state
            evidence.update(task_id=snapshot.task.task_id, high_pairs=high_pairs, improved_pairs=improved_pairs)
            records.append(evidence)
        return {"configuration": scope["configuration"], "records": records,
                "counts": dict(Counter(r["state"] for r in records)), "reference_count": len(references),
                "cutoff": CORRELATION_CUTOFF, "minimum_intervals": MIN_CORRELATION_INTERVALS,
                "improvement_passed": sum(r["state"] == "passed" and r["high_pairs"] > 0 for r in records)}
