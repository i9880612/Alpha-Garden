from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from persistence.database import open_database


class OpenDatabaseTests(unittest.TestCase):
    def test_read_transaction_longer_than_default_wait_does_not_abort_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "database.sqlite3"
            with open_database(path) as connection:
                connection.execute("CREATE TABLE samples (value TEXT)")
            reader = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
            reader.execute("BEGIN")
            reader.execute("SELECT * FROM samples").fetchall()

            def write_once():
                with open_database(path) as connection:
                    connection.execute("INSERT INTO samples VALUES ('saved')")

            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(write_once)
                try:
                    with self.assertRaises(TimeoutError):
                        future.result(timeout=6)
                finally:
                    reader.close()
                future.result(timeout=5)
            with open_database(path) as connection:
                self.assertEqual(connection.execute("SELECT value FROM samples").fetchall()[0][0], "saved")
                self.assertEqual(connection.execute("SELECT count(*) FROM samples").fetchone()[0], 1)

    def test_analysis_on_materialized_rows_does_not_hold_a_read_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "database.sqlite3"
            with open_database(path) as connection:
                connection.execute("CREATE TABLE samples (value TEXT)")
                connection.execute("INSERT INTO samples VALUES ('before')")
            reader = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
            try:
                reader.execute("BEGIN")
                facts = reader.execute("SELECT value FROM samples").fetchall()
            finally:
                reader.close()
            # Potentially slow analysis uses detached facts, not a live cursor
            # or a connection whose explicit read transaction is still open.
            with open_database(path) as connection:
                connection.execute("PRAGMA busy_timeout=50")
                connection.execute("INSERT INTO samples VALUES ('after')")
            self.assertEqual(facts, [("before",)])

    def test_exhausted_lock_wait_rolls_back_without_replaying_transaction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "database.sqlite3"
            with open_database(path) as connection:
                connection.execute("CREATE TABLE samples (value TEXT)")
            reader = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
            reader.execute("BEGIN")
            reader.execute("SELECT * FROM samples").fetchall()
            executions = []
            try:
                with self.assertRaises(sqlite3.OperationalError) as raised:
                    with open_database(path) as connection:
                        connection.execute("PRAGMA busy_timeout=50")
                        executions.append("executed")
                        connection.execute("INSERT INTO samples VALUES ('discarded')")
                self.assertEqual(raised.exception.sqlite_errorcode, sqlite3.SQLITE_BUSY)
            finally:
                reader.close()
            with open_database(path) as connection:
                self.assertEqual(connection.execute("SELECT count(*) FROM samples").fetchone()[0], 0)
            self.assertEqual(executions, ["executed"])

    def test_commits_successful_transaction_and_returns_named_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "database.sqlite3"

            with open_database(database_path) as connection:
                connection.execute("CREATE TABLE samples (id INTEGER PRIMARY KEY, value TEXT)")
                connection.execute("INSERT INTO samples (value) VALUES (?)", ("saved",))

            with open_database(database_path) as connection:
                row = connection.execute("SELECT value FROM samples").fetchone()

            self.assertIsNotNone(row)
            self.assertEqual(row["value"], "saved")

    def test_rolls_back_failed_transaction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "database.sqlite3"
            with open_database(database_path) as connection:
                connection.execute("CREATE TABLE samples (value TEXT)")

            with self.assertRaisesRegex(RuntimeError, "stop"):
                with open_database(database_path) as connection:
                    connection.execute("INSERT INTO samples (value) VALUES (?)", ("discarded",))
                    raise RuntimeError("stop")

            with open_database(database_path) as connection:
                count = connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0]

            self.assertEqual(count, 0)

    def test_enables_foreign_key_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "database.sqlite3"

            with self.assertRaises(sqlite3.IntegrityError):
                with open_database(database_path) as connection:
                    connection.execute("CREATE TABLE parents (id INTEGER PRIMARY KEY)")
                    connection.execute(
                        "CREATE TABLE children (parent_id INTEGER REFERENCES parents(id))"
                    )
                    connection.execute("INSERT INTO children (parent_id) VALUES (1)")


if __name__ == "__main__":
    unittest.main()
