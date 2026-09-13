import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from alpha_garden.console import run_progress_console
from execution.submission_progress import SubmissionProgress


class SubmissionProgressTests(unittest.TestCase):
    def test_missing_database_is_not_created_by_logging(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.sqlite3"
            with redirect_stdout(io.StringIO()), run_progress_console():
                SubmissionProgress(path, "account", frozenset({"task"})).observe("task")
            self.assertFalse(path.exists())
