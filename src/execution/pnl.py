from __future__ import annotations

import logging
from datetime import datetime, timedelta

from execution.progress import LOGGER_NAME
from persistence.database import open_database
from persistence.pnl import PnlSeriesRecord, pnl_capture_states, save_pnl_series
from worldquant.backtests import WorldQuantProtocolError
from worldquant.client import WorldQuantRequestError
from worldquant.pnl import PnlObservation


PNL_CAPTURE_ACTIONS = frozenset({"pnl_captured", "pnl_pending"})


class PnlAccountRequestError(WorldQuantRequestError):
    """An account fault during a curve read, never a simulation failure."""


def capture_pnl_series(
    database_path, client, *, account_scope: str, platform_alpha_id: str, observed_at: str,
) -> PnlObservation | None:
    """Read one due curve; None means cached or still in its saved cooldown."""
    now = datetime.fromisoformat(observed_at)
    with open_database(database_path) as connection:
        state = pnl_capture_states(connection, account_scope=account_scope,
            platform_alpha_ids=frozenset((platform_alpha_id,))).get(platform_alpha_id)
    if state is not None and (state[0] or (state[1] is not None and datetime.fromisoformat(state[1]) > now)):
        return None
    logger = logging.getLogger(LOGGER_NAME)
    logger.info("[PnL采集] %s 开始获取每日曲线", platform_alpha_id)
    error_code = None
    try:
        observation = client.fetch_pnl(platform_alpha_id=platform_alpha_id)
    except (WorldQuantRequestError, WorldQuantProtocolError) as exc:
        if isinstance(exc, WorldQuantRequestError) and (
            exc.status_code in {401, 403, 429} or exc.code.startswith("worldquant_authentication_")
        ):
            logger.warning("[PnL采集] %s 请求异常（%s），交由账号级故障处理", platform_alpha_id, exc.code)
            raise PnlAccountRequestError(
                exc.code, status_code=exc.status_code, retryable=exc.retryable,
                outcome_unknown=exc.outcome_unknown, retry_after_seconds=exc.retry_after_seconds,
                transport_error_type=exc.transport_error_type,
            ) from exc
        observation = PnlObservation(None, getattr(exc, "retry_after_seconds", None))
        error_code = exc.code
    retry = max(600, observation.retry_after_seconds or 0)
    retry_at = (now + timedelta(seconds=retry)).isoformat() if observation.points is None else None
    with open_database(database_path) as connection:
        save_pnl_series(connection, PnlSeriesRecord(
            account_scope, platform_alpha_id, observed_at, observation.points, retry_at))
    if observation.points is None:
        logger.warning("[PnL采集] %s %s，证据待定，%g 秒后可重试", platform_alpha_id,
            f"读取失败（{error_code}）" if error_code else "暂未就绪", retry)
    else:
        logger.info("[PnL采集] %s 每日曲线已保存（%s 个点）", platform_alpha_id, len(observation.points))
    return observation
