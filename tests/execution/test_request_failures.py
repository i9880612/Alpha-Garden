from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from execution.request_failures import (
    request_outage_tolerance_seconds,
    request_retry_delay_seconds,
)
from execution.run_config import load_automated_run_limits


def test_retry_delay_doubles_from_five_seconds_and_caps_at_five_minutes():
    assert [request_retry_delay_seconds(n) for n in range(8)] == [
        5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 300.0, 300.0]
    assert request_retry_delay_seconds(10_000) == 300.0


def test_failure_limit_maps_to_a_bounded_outage_length():
    assert request_outage_tolerance_seconds(1) == 0
    assert request_outage_tolerance_seconds(3) == 15.0
    # 29 waits before the 30th consecutive fault stops a finite run: about two hours.
    assert request_outage_tolerance_seconds(30) == 315.0 + 23 * 300.0


def test_default_run_config_survives_a_long_network_drop():
    root = Path(__file__).resolve().parents[2] / "config"
    limits = load_automated_run_limits(root / "run.default.json", cycles=1)
    assert request_outage_tolerance_seconds(limits.max_request_failures) >= 2 * 60 * 60


def test_formal_check_rate_limit_waits_like_the_pnl_limit():
    from execution.request_failures import is_slow_rate_limit
    from worldquant.client import WorldQuantRequestError

    def error(code, status):
        return WorldQuantRequestError(code, status_code=status, retryable=True, outcome_unknown=False)

    assert is_slow_rate_limit(error("worldquant_formal_check_http_error", 429))
    assert not is_slow_rate_limit(error("worldquant_formal_check_http_error", 503))
    assert not is_slow_rate_limit(error("worldquant_submission_http_error", 429))
