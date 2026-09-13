from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FormalCheckObservation:
    payload: object
    retry_after_seconds: float | None


@dataclass(frozen=True, slots=True)
class FormalSubmissionObservation:
    status_code: int
    payload: object | None
    retry_after_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class AlphaDetailObservation:
    payload: Mapping[str, object]
