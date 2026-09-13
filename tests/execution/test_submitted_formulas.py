import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from execution.submitted_formulas import export_submitted_formulas, refresh_submitted_formulas
from persistence.database import open_database
from persistence.submissions import initialize_submission_schema, record_platform_submitted_alphas
from tests.persistence import test_submissions as submission_fixture


class SubmittedFormulasTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.database = Path(directory.name) / "database.sqlite3"
        self.record = submission_fixture.PlatformSubmittedAlphaPersistenceTests._record(raw_extra={
            "author": "not-for-the-report", "is": {"sharpe": 1.5, "fitness": 1.1, "turnover": 0.03},
            "settings": {"region": "USA", "delay": 1},
        })
        with open_database(self.database) as connection:
            initialize_submission_schema(connection)
            record_platform_submitted_alphas(connection, (self.record,))

    def test_document_contains_only_identity_time_and_brief_metrics(self):
        other = replace(self.record, account_scope="another-account")
        with open_database(self.database) as connection:
            record_platform_submitted_alphas(connection, (other,))
        before = self.database.read_bytes()
        path = export_submitted_formulas(self.database)
        text = path.read_text(encoding="utf-8")
        self.assertIn("共 **2** 条", text)
        self.assertEqual(text.count("| alpha-1 |"), 2)
        self.assertIn("2026-09-01 14:35:02", text)
        self.assertIn("| 1.50 | 1.10 | 3.0% |", text)
        self.assertNotIn(self.record.formula, text)
        self.assertNotIn(self.record.account_scope, text)
        self.assertNotIn(other.account_scope, text)
        self.assertNotIn("USA", text)
        self.assertNotIn("settings", text)
        self.assertNotIn("```", text)
        self.assertNotIn("not-for-the-report", text)
        self.assertEqual(self.database.read_bytes(), before)
        self.assertEqual(export_submitted_formulas(self.database).read_text(encoding="utf-8"), text)

    def test_missing_metrics_remain_unknown(self):
        raw = dict(self.record.raw_payload)
        raw.pop("is")
        raw.pop("settings")
        with open_database(self.database) as connection:
            record_platform_submitted_alphas(connection, (replace(self.record, raw_payload=raw),))
        text = export_submitted_formulas(self.database).read_text(encoding="utf-8")
        self.assertIn("| 未记录 | 未记录 | 未记录 |", text)

    def test_failed_replacement_preserves_prior_document_and_committed_facts(self):
        path = export_submitted_formulas(self.database)
        original = path.read_bytes()
        before = self.database.read_bytes()
        with patch.object(Path, "replace", side_effect=PermissionError("locked")):
            with self.assertLogs("execution.progress", level="WARNING") as messages:
                refresh_submitted_formulas(self.database)
        self.assertIn("export-submitted", messages.output[0])
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(self.database.read_bytes(), before)
        self.assertEqual(list(path.parent.glob(".submitted-formulas-*.tmp")), [])

    def test_missing_database_is_not_created(self):
        missing = self.database.with_name("missing.sqlite3")
        with self.assertRaises(sqlite3.OperationalError):
            export_submitted_formulas(missing)
        self.assertFalse(missing.exists())
