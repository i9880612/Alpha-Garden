from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


DATABASE_LOCK_TIMEOUT_SECONDS = 30.0


@contextmanager
def open_database(path: str | Path) -> Iterator[sqlite3.Connection]:
    """Open one transaction; wait for brief lock contention without replaying work."""
    # Native busy handling waits on the blocked SQL/commit, never re-enters the
    # caller (which may already have sent an external request). Keep it bounded.
    connection = sqlite3.connect(Path(path), timeout=DATABASE_LOCK_TIMEOUT_SECONDS)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")

    try:
        with connection:
            yield connection
    finally:
        connection.close()
