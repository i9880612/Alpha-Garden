from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from worldquant.client import WorldQuantCredentials


_ENVIRONMENT_KEYS = {
    "WQB_ACCOUNT_SCOPE",
    "WQB_BASE_URL",
    "WQB_EMAIL",
    "WQB_PASSWORD",
    "WQB_SESSION_TOKEN",
}


@dataclass(frozen=True, slots=True)
class WorldQuantConnectionSettings:
    account_scope: str
    base_url: str
    credentials: WorldQuantCredentials


def load_worldquant_connection_settings(
    environment_path: str | Path,
) -> WorldQuantConnectionSettings:
    values = _read_environment_file(environment_path)
    account_scope = values.get("WQB_ACCOUNT_SCOPE", "").strip()
    if not account_scope:
        raise ValueError("worldquant_account_scope_missing")
    base_url = values.get("WQB_BASE_URL", "").strip()
    if not base_url:
        raise ValueError("worldquant_base_url_missing")

    token = values.get("WQB_SESSION_TOKEN", "").strip()
    email = values.get("WQB_EMAIL", "").strip()
    password = values.get("WQB_PASSWORD", "")
    credentials = WorldQuantCredentials(
        username=email or None,
        password=password or None,
        bearer_token=token or None,
    )
    return WorldQuantConnectionSettings(
        account_scope=account_scope,
        base_url=base_url,
        credentials=credentials,
    )


def _read_environment_file(environment_path: str | Path) -> dict[str, str]:
    path = Path(environment_path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ValueError("worldquant_environment_file_missing") from exc
    except (OSError, UnicodeError) as exc:
        raise ValueError("worldquant_environment_file_invalid") from exc

    values: dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError("worldquant_environment_line_invalid")
        raw_name, raw_value = line.split("=", 1)
        name = raw_name.strip()
        if name not in _ENVIRONMENT_KEYS:
            continue
        if name in values:
            raise ValueError("worldquant_environment_key_duplicate")
        values[name] = raw_value.strip()
    return values
