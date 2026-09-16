import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from unittest.mock import patch

from alpha_garden.web_events import ConsoleEvents
from execution.console_jobs import ConsoleJobs
from persistence.database import open_database
from tests.execution.console_fixture import make_console_reader


class ConsoleEventsTests(unittest.TestCase):
    def test_filtered_analysis_cache_retains_only_latest_scope_and_refreshes_after_commit(self):
        calls = []
        def read():
            calls.append(1)
            return len(calls)
        group = "/api/analysis/quality"
        self.assertEqual(self.events.read_current(group + "a", "database", read, cache=True, cache_group=group), 1)
        self.assertEqual(self.events.read_current(group + "a", "database", read, cache=True, cache_group=group), 1)
        self.assertEqual(self.events.read_current(group + "b", "database", read, cache=True, cache_group=group), 2)
        self.assertEqual(self.events.read_current(group + "a", "database", read, cache=True, cache_group=group), 3)
        with open_database(self.reader.paths.database) as c:
            c.execute("INSERT INTO event_input VALUES (1)")
        self.assertEqual(self.events.read_current(group + "a", "database", read, cache=True, cache_group=group), 4)

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.reader = make_console_reader(self.folder.name)
        with open_database(self.reader.paths.database) as connection:
            connection.execute("CREATE TABLE event_input(value INTEGER)")
        self.jobs = ConsoleJobs(self.reader, read_only=True)
        self.addCleanup(self.jobs.close)
        self.events = ConsoleEvents(self.reader.paths, self.jobs)
        self.addCleanup(self.events.close)

    def test_idle_observation_is_read_only_and_commits_reach_every_subscriber(self):
        first = self.events.wait()
        before = self.reader.paths.database.read_bytes()
        self.assertEqual(self.events.wait(first, timeout=0.02), first)
        self.assertEqual(self.reader.paths.database.read_bytes(), before)
        with open_database(self.reader.paths.database) as writer:
            writer.execute("INSERT INTO event_input VALUES (1)")
            self.assertEqual(self.events.wait(first, timeout=0.02), first)
        changed = self.events.wait(first, timeout=3)
        self.assertGreater(changed["database"], first["database"])
        self.assertEqual(changed["jobs"], first["jobs"])
        self.assertEqual(changed["session"], first["session"])
        self.assertEqual(self.events.wait(first, timeout=0), changed)
        self.assertEqual(self.events.wait(), changed)

    def test_configuration_change_refreshes_session_and_data(self):
        first = self.events.wait()
        self.reader.paths.environment.write_text("WQB_ACCOUNT_SCOPE=changed\n", encoding="utf-8")
        changed = self.events.wait(first, timeout=3)
        self.assertGreater(changed["session"], first["session"])
        self.assertGreater(changed["database"], first["database"])

    def test_missing_database_is_not_created_and_recovery_notifies_readers(self):
        self.events.close()
        self.reader.paths.database.unlink()
        self.events = ConsoleEvents(self.reader.paths, self.jobs)
        self.addCleanup(self.events.close)
        first = self.events.wait()
        self.assertFalse(self.reader.paths.database.exists())
        with open_database(self.reader.paths.database) as writer:
            writer.execute("CREATE TABLE event_input(value INTEGER)")
        changed = self.events.wait(first, timeout=3)
        self.assertGreater(changed["database"], first["database"])
        self.assertGreater(changed["session"], first["session"])

    def test_closing_releases_waiters(self):
        self.events.wait()
        self.events.close()
        self.assertIsNone(self.events.wait())

    def test_concurrent_reads_share_work_without_caching_completed_results(self):
        started, release, joined = threading.Event(), threading.Event(), threading.Event()
        calls = []
        def read():
            calls.append(1)
            started.set()
            release.wait(3)
            return {"value": len(calls)}
        def follow():
            joined.set()
            return self.events.read_current("research", "database", read)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.events.read_current, "research", "database", read)
            try:
                self.assertTrue(started.wait(1))
                second = pool.submit(follow)
                self.assertTrue(joined.wait(1))
                with self.assertRaises(TimeoutError):
                    second.result(timeout=0.05)
            finally:
                release.set()
            self.assertEqual(first.result(1), {"value": 1})
            self.assertEqual(second.result(1), {"value": 1})
        self.assertEqual(self.events.read_current("research", "database", read), {"value": 2})

    def test_new_database_revision_does_not_join_an_older_read(self):
        first_version = self.events.wait()
        started, release = threading.Event(), threading.Event()
        def old_read():
            started.set()
            release.wait(4)
            return {"value": 0}
        with ThreadPoolExecutor(max_workers=2) as pool:
            old = pool.submit(self.events.read_current, "research", "database", old_read)
            try:
                self.assertTrue(started.wait(1))
                with open_database(self.reader.paths.database) as writer:
                    writer.execute("INSERT INTO event_input VALUES (1)")
                changed = self.events.wait(first_version, timeout=3)
                self.assertGreater(changed["database"], first_version["database"])
                current = pool.submit(self.events.read_current, "research", "database", lambda: {"value": 1})
                self.assertEqual(current.result(1), {"value": 1})
                self.assertFalse(old.done())
            finally:
                release.set()
            self.assertEqual(old.result(1), {"value": 0})

    def test_cached_display_checks_commits_synchronously_including_wal(self):
        self.events.close()
        # No observer tick is available to invalidate these reads for them.
        with patch.object(ConsoleEvents, "_watch"):
            self.events = ConsoleEvents(self.reader.paths, self.jobs)
        self.addCleanup(self.events.close)
        calls = []
        def read():
            calls.append(1)
            from persistence.console import read_console_database
            with read_console_database(self.reader.paths.database) as connection:
                return connection.execute("SELECT COUNT(*) FROM event_input").fetchone()[0]
        with open_database(self.reader.paths.database) as writer:
            writer.execute("PRAGMA journal_mode=WAL")
            self.assertEqual(self.events.read_current("activity", "database", read, cache=True), 0)
            self.assertEqual(self.events.read_current("activity", "database", read, cache=True), 0)
            self.assertEqual(len(calls), 1)
            writer.execute("INSERT INTO event_input VALUES (1)")
            self.assertEqual(self.events.read_current("activity", "database", read, cache=True), 0)
            writer.commit()
            self.assertEqual(self.events.read_current("activity", "database", read, cache=True), 1)
            self.assertEqual(len(calls), 2)
        self.events.close()

    def test_cached_display_invalidates_configuration_and_does_not_cache_errors(self):
        calls = []
        def read():
            calls.append(1)
            return self.reader.paths.environment.read_text(encoding="utf-8")
        before = self.events.read_current("research", "database", read, cache=True)
        self.assertEqual(self.events.read_current("research", "database", read, cache=True), before)
        self.reader.paths.environment.write_text("WQB_ACCOUNT_SCOPE=other\n", encoding="utf-8")
        self.assertEqual(self.events.read_current("research", "database", read, cache=True), "WQB_ACCOUNT_SCOPE=other\n")
        self.assertEqual(len(calls), 2)
        def failing():
            raise ValueError("unavailable")
        with self.assertRaisesRegex(ValueError, "unavailable"):
            self.events.read_current("activity", "database", failing, cache=True)
        self.assertEqual(self.events.read_current("activity", "database", lambda: 3, cache=True), 3)

    def test_old_computation_cannot_populate_cache_after_a_commit(self):
        started, release = threading.Event(), threading.Event()
        def old_read():
            started.set()
            release.wait(3)
            return 0
        with ThreadPoolExecutor(max_workers=1) as pool:
            old = pool.submit(self.events.read_current, "research", "database", old_read, cache=True)
            try:
                self.assertTrue(started.wait(1))
                with open_database(self.reader.paths.database) as writer:
                    writer.execute("INSERT INTO event_input VALUES (1)")
                self.assertEqual(self.events.read_current("research", "database", lambda: 1, cache=True), 1)
            finally:
                release.set()
            self.assertEqual(old.result(1), 0)
        self.assertEqual(self.events.read_current("research", "database", lambda: 2, cache=True), 1)

    def test_cached_display_cannot_hide_database_loss(self):
        self.assertEqual(self.events.read_current("research", "database", lambda: 1, cache=True), 1)
        with patch.object(type(self.reader.paths.database), "stat", side_effect=FileNotFoundError):
            def missing():
                raise FileNotFoundError
            with self.assertRaises(FileNotFoundError):
                self.events.read_current("research", "database", missing, cache=True)
        self.assertEqual(self.events.read_current("research", "database", lambda: 2, cache=True), 2)

    def test_stream_sends_revisions_and_reconnect_starts_at_current_revision(self):
        stream = self.events.stream()
        first = next(stream)
        self.assertEqual(set(first), {"database", "jobs", "session"})
        with open_database(self.reader.paths.database) as writer:
            writer.execute("INSERT INTO event_input VALUES (1)")
        changed = next(stream)
        self.assertGreater(changed["database"], first["database"])
        stream.close()
        reconnected = self.events.stream()
        self.assertEqual(next(reconnected), changed)
        reconnected.close()
