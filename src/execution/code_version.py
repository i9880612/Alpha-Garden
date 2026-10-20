"""Detect that the source on disk changed since this process imported it.

The web console runs research in a thread of its long-lived process, so after a code
update it would keep running the version it imported at startup, silently. Hashing file
contents (not mtimes, which editors and sync tools rewrite) keeps the check honest.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]


def source_fingerprint(root: Path = SOURCE_ROOT) -> str:
    digest = sha256()
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
