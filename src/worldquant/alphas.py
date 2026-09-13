from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from worldquant.backtests import WorldQuantProtocolError


@dataclass(frozen=True, slots=True)
class UserAlphaRecord:
    platform_alpha_id: str
    alpha_type: str
    status: str
    formula: str | None
    settings: Mapping[str, object] | None
    created_at: datetime
    hidden: bool


@dataclass(frozen=True, slots=True)
class UserAlphaPage:
    total_count: int
    records: tuple[UserAlphaRecord, ...]
    has_next: bool


def parse_user_alpha_page(payload: object) -> UserAlphaPage:
    if not isinstance(payload, Mapping) or set(payload) != {
        "count",
        "next",
        "previous",
        "results",
    }:
        raise WorldQuantProtocolError("worldquant_user_alphas_page_invalid")
    total_count = payload["count"]
    if (
        isinstance(total_count, bool)
        or not isinstance(total_count, int)
        or total_count < 0
    ):
        raise WorldQuantProtocolError("worldquant_user_alphas_count_invalid")
    next_page = payload["next"]
    previous_page = payload["previous"]
    for value in (next_page, previous_page):
        if value is not None and (
            not isinstance(value, str) or not value.strip()
        ):
            raise WorldQuantProtocolError("worldquant_user_alphas_pagination_invalid")
    results = payload["results"]
    if not isinstance(results, list) or len(results) > total_count:
        raise WorldQuantProtocolError("worldquant_user_alphas_results_invalid")

    records: list[UserAlphaRecord] = []
    identities: set[str] = set()
    for value in results:
        if not isinstance(value, Mapping):
            raise WorldQuantProtocolError("worldquant_user_alpha_invalid")
        platform_alpha_id = _required_text(
            value.get("id"),
            "worldquant_user_alpha_id_invalid",
        )
        if platform_alpha_id in identities:
            raise WorldQuantProtocolError("worldquant_user_alpha_duplicate")
        identities.add(platform_alpha_id)
        alpha_type = _required_text(
            value.get("type"),
            "worldquant_user_alpha_type_invalid",
        )
        status = _required_text(
            value.get("status"),
            "worldquant_user_alpha_status_invalid",
        )
        created_at = _aware_timestamp(value.get("dateCreated"))
        hidden = value.get("hidden")
        if not isinstance(hidden, bool):
            raise WorldQuantProtocolError("worldquant_user_alpha_hidden_invalid")

        formula: str | None = None
        settings: Mapping[str, object] | None = None
        if alpha_type == "REGULAR":
            regular = value.get("regular")
            raw_settings = value.get("settings")
            if not isinstance(regular, Mapping):
                raise WorldQuantProtocolError("worldquant_user_alpha_formula_invalid")
            formula = _required_text(
                regular.get("code"),
                "worldquant_user_alpha_formula_invalid",
            )
            if not isinstance(raw_settings, Mapping):
                raise WorldQuantProtocolError("worldquant_user_alpha_settings_invalid")
            settings = dict(raw_settings)

        records.append(
            UserAlphaRecord(
                platform_alpha_id=platform_alpha_id,
                alpha_type=alpha_type,
                status=status,
                formula=formula,
                settings=settings,
                created_at=created_at,
                hidden=hidden,
            )
        )
    return UserAlphaPage(
        total_count=total_count,
        records=tuple(records),
        has_next=next_page is not None,
    )


def _required_text(value: object, error: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise WorldQuantProtocolError(error)
    return value


def _aware_timestamp(value: object) -> datetime:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise WorldQuantProtocolError("worldquant_user_alpha_created_at_invalid")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise WorldQuantProtocolError(
            "worldquant_user_alpha_created_at_invalid"
        ) from exc
    if parsed.utcoffset() is None:
        raise WorldQuantProtocolError("worldquant_user_alpha_created_at_invalid")
    return parsed
