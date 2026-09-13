from __future__ import annotations

import logging
import sys
import shutil
import unicodedata
from contextlib import contextmanager

from execution.progress import LOGGER_NAME


class _ProgressHandler(logging.StreamHandler):
    def __init__(self, stream):
        super().__init__(stream)
        self.live_width = 0

    def format(self, record):
        rendered = super().format(record)
        if self.stream.isatty():
            for badge, color in (("[通过]", "32"), ("[未通过]", "31")):
                if record.getMessage().startswith(badge):
                    return rendered.replace(badge, f"\x1b[{color}m{badge}\x1b[0m", 1)
        return rendered

    def emit(self, record):
        try:
            transient = getattr(record, "transient", False)
            if transient:
                if not self.stream.isatty():
                    return
                message = self.format(record)
                limit = max(1, shutil.get_terminal_size().columns - 1)
                text, width = "", 0
                for character in message:
                    size = (
                        0
                        if unicodedata.combining(character)
                        else (
                            2
                            if unicodedata.east_asian_width(character) in {"W", "F"}
                            else 1
                        )
                    )
                    if width + size > limit:
                        break
                    text += character
                    width += size
                self.stream.write(
                    "\r" + text + " " * max(0, self.live_width - width) + "\r"
                )
                self.live_width = width
                self.flush()
                return
            self.clear_live()
            super().emit(record)
        except Exception:
            self.handleError(record)

    def clear_live(self):
        if self.live_width:
            self.stream.write("\r" + " " * self.live_width + "\r")
            self.live_width = 0
            self.flush()

    def handleError(self, record) -> None:
        # Output failures cannot terminate research or change business facts.
        self.setLevel(logging.CRITICAL + 1)


@contextmanager
def run_progress_console():
    """Enable immediate console progress only for this CLI invocation."""
    logger = logging.getLogger(LOGGER_NAME)
    previous = logger.level, logger.propagate, logger.disabled
    handler = _ProgressHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.disabled = False
    try:
        yield
    finally:
        try:
            handler.clear_live()
        except (OSError, ValueError):
            pass
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(previous[0])
        logger.propagate, logger.disabled = previous[1:]
