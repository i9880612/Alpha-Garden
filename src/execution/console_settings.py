"""Read and explicitly save the same defaults consumed by new research runs."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from execution.process_lock import exclusive_run_process
from execution.run_config import automated_run_limits_from_config
from selection.settings import BacktestSettingsPolicy


def read_settings(paths):
    # Values and revisions must describe the same bytes, even during a save.
    policy_bytes = paths.settings.read_bytes()
    run_bytes = paths.run_config.read_bytes()
    policy = json.loads(policy_bytes)
    run_config = json.loads(run_bytes)
    backtest = BacktestSettingsPolicy.from_config_dict(policy)
    limits = automated_run_limits_from_config(run_config, cycles=1)
    return {"policy": policy, "backtest": asdict(backtest), "limits": asdict(limits),
            "run_config": run_config,
            "revisions": {"backtest": _revision(policy_bytes), "run": _revision(run_bytes)}}


def save_settings(paths, request):
    if (set(request) != {"section", "revision", "value"}
            or request["section"] not in ("backtest", "run")
            or not isinstance(request["revision"], str)):
        raise ValueError("console_settings_invalid")
    value = request["value"]
    try:
        if request["section"] == "backtest":
            value = BacktestSettingsPolicy.from_config_dict(value).as_config_dict()
        else:
            automated_run_limits_from_config(value, cycles=1)
        encoded = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("console_settings_invalid") from exc
    path = paths.settings if request["section"] == "backtest" else paths.run_config
    try:
        # Coordinate with CLI research as well as web operations. No database write.
        with exclusive_run_process(paths.database):
            if _revision(path.read_bytes()) != request["revision"]:
                raise ValueError("console_settings_conflict")
            _replace_file(path, encoded)
    except ValueError as exc:
        if str(exc) == "已有命令正在使用此数据库，请先停止该进程。":
            raise ValueError("console_settings_busy") from exc
        raise
    return {"section": request["section"], "value": value, "revision": _revision(encoded)}


def _revision(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _replace_file(path: Path, value: bytes):
    # A single section owns a single file: failed writes leave its previous bytes intact.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".settings-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
