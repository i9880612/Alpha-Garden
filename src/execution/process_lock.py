from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def exclusive_run_process(database_path: str | Path):
    """One driver per database; the OS releases ownership even after a crash.

    The empty sidecar is only a lock target, not a PID or business-state store.
    Keep it in place: unlinking it could give two processes different locks.
    """
    path = Path(database_path).resolve()
    with path.with_name(path.name + ".run.lock").open("a+b") as handle:
        handle.seek(0)
        try:
            _lock(handle, release=False)
        except (BlockingIOError, PermissionError) as exc:
            raise ValueError("已有命令正在使用此数据库，请先停止该进程。") from exc
        try:
            yield
        finally:
            _lock(handle, release=True)


def _lock(handle, *, release: bool) -> None:
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK if release else msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle, fcntl.LOCK_UN if release else fcntl.LOCK_EX | fcntl.LOCK_NB)
