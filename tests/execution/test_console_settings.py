import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from execution.console import ConsoleReader
from execution.console_jobs import ConsoleJobs
from execution.process_lock import exclusive_run_process
from execution.run_config import load_automated_run_limits
from execution.runs import prepare_automated_run, retire_previous_automated_runs
from persistence.console import read_console_database
from persistence.runs import get_automated_run
from selection.settings import load_backtest_settings_policy
from tests.execution.console_fixture import make_console_reader


class ConsoleSettingsTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        reader = make_console_reader(self.folder.name)
        policy, run_config = (Path(self.folder.name) / name for name in ("policy.json", "run.json"))
        policy.write_bytes(reader.paths.settings.read_bytes())
        run_config.write_bytes(reader.paths.run_config.read_bytes())
        self.reader = ConsoleReader(replace(reader.paths, settings=policy, run_config=run_config))
        self.jobs = ConsoleJobs(self.reader)
        self.addCleanup(self.jobs.close)

    def request(self, section):
        current = self.reader.settings()
        return {"section": section, "revision": current["revisions"][section],
                "value": current["policy" if section == "backtest" else "run_config"]}

    def test_saved_defaults_are_consumed_by_new_plans_and_do_not_change_existing_plan(self):
        paths = self.reader.paths
        limits = load_automated_run_limits(paths.run_config, cycles=2)
        previous = prepare_automated_run(paths.database, paths.settings, account_scope=self.reader.account_scope,
                                        limits=limits, created_at="2026-09-01T00:00:00+00:00")
        database_before = paths.database.read_bytes()
        run_before = paths.run_config.read_bytes()
        request = self.request("backtest")
        request["value"]["decay"] = 8
        request["value"]["neutralization"]["byFieldCategory"]["Model"] = "SECTOR"
        with patch("worldquant.client.WorldQuantClient") as client:
            result = self.jobs.save_settings(request)
            client.assert_not_called()
        self.assertEqual(result["revision"], self.reader.settings()["revisions"]["backtest"])
        self.assertEqual(paths.run_config.read_bytes(), run_before)
        self.assertEqual(load_backtest_settings_policy(paths.settings).decay, 8)
        policy_before = paths.settings.read_bytes()
        request = self.request("run")
        request["value"].update(generationCount=600, backtestCount=120, explorationPercent=40,
                                selfCorrelationPercent=20, maxPendingSeconds=900)
        self.jobs.save_settings(request)
        self.assertEqual(paths.settings.read_bytes(), policy_before)
        self.assertEqual(paths.database.read_bytes(), database_before)
        with read_console_database(paths.database) as connection:
            self.assertEqual(get_automated_run(connection, previous.run_id), previous)
        current_limits = load_automated_run_limits(paths.run_config, cycles=3)
        retire_previous_automated_runs(paths.database, account_scope=self.reader.account_scope,
                                      observed_at="2026-09-02T00:00:00+00:00")
        new = prepare_automated_run(paths.database, paths.settings, account_scope=self.reader.account_scope,
                                   limits=current_limits, created_at="2026-09-02T00:00:00+00:00")
        self.assertEqual(json.loads(new.settings_policy_json)["decay"], 8)
        self.assertEqual(new.backtest_count, 120)
        self.assertEqual(new.max_backtests, 360)
        self.assertEqual(new.max_pending_seconds, 900)
        self.assertFalse(new.automatic_submissions_enabled)

    def test_invalid_settings_never_change_either_file(self):
        originals = [path.read_bytes() for path in (self.reader.paths.settings, self.reader.paths.run_config)]
        cases = [
            ("run", lambda v: v.update(explorationPercent=50)),
            ("run", lambda v: v.update(backtestCount=501)),
            ("run", lambda v: v.update(directionValidationPercent=41)),
            ("run", lambda v: v.update(maxPendingSeconds=0)),
            ("run", lambda v: v.update(generationCount=True)),
            ("run", lambda v: v.update(automatic_submissions_enabled=True)),
            ("backtest", lambda v: v["truncation"].update(tailRisk=0.5)),
            ("backtest", lambda v: v["neutralization"].update(default="INVALID")),
            ("backtest", lambda v: v.update(decay=-1)),
            ("backtest", lambda v: v.update(visualization="true")),
            ("backtest", lambda v: v.update(decay=None)),
        ]
        for section, change in cases:
            request = self.request(section)
            change(request["value"])
            with self.subTest(section=section, value=request["value"]), self.assertRaisesRegex(ValueError, "console_settings_invalid"):
                self.jobs.save_settings(request)
            self.assertEqual(originals, [path.read_bytes() for path in (self.reader.paths.settings, self.reader.paths.run_config)])

    def test_conflict_read_only_busy_and_failed_file_replacement_preserve_saved_values(self):
        request = self.request("run")
        changed = self.request("run")
        changed["value"]["backtestCount"] = 110
        self.jobs.save_settings(changed)
        before = self.reader.paths.run_config.read_bytes()
        with self.assertRaisesRegex(ValueError, "console_settings_conflict"):
            self.jobs.save_settings(request)
        request = self.request("run")
        request["value"]["backtestCount"] = 120
        self.jobs.read_only = True
        with self.assertRaisesRegex(ValueError, "console_read_only"):
            self.jobs.save_settings(request)
        self.jobs.read_only = False
        with exclusive_run_process(self.reader.paths.database), self.assertRaisesRegex(ValueError, "console_settings_busy"):
            self.jobs.save_settings(request)
        with patch("execution.console_settings.os.replace", side_effect=PermissionError("locked")):
            with self.assertRaises(PermissionError):
                self.jobs.save_settings(request)
        self.assertEqual(self.reader.paths.run_config.read_bytes(), before)
        self.assertEqual(list(Path(self.folder.name).glob(".settings-*")), [])
