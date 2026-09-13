import io
import logging
import unittest
from unittest.mock import patch
import os
from contextlib import redirect_stdout

from alpha_garden.console import run_progress_console
from execution.progress import LOGGER_NAME, phase, live_progress


class ProgressConsoleTests(unittest.TestCase):
    def test_live_progress_overwrites_one_terminal_line_and_clips_width(self):
        class Terminal(io.StringIO):
            def isatty(self):
                return True
        output = Terminal()
        with redirect_stdout(output), patch("alpha_garden.console.shutil.get_terminal_size", return_value=os.terminal_size((40, 20))), run_progress_console():
            live_progress("等待" * 100)
            live_progress("完成 1/20")
            self.assertNotIn("\n", output.getvalue())
            self.assertIn("\r", output.getvalue())
            phase(1, 3, "最终结果")
        self.assertEqual(output.getvalue().count("\n"), 1)
        self.assertNotIn("等待" * 20, output.getvalue())

    def test_redirected_output_omits_transient_updates(self):
        output = io.StringIO()
        with redirect_stdout(output), run_progress_console():
            live_progress("轮询中")
            phase(1, 3, "最终结果")
        self.assertNotIn("轮询中", output.getvalue())
        self.assertEqual(len(output.getvalue().splitlines()), 1)

    def test_console_flushes_and_restores_logging_after_interruption(self):
        class Stream(io.StringIO):
            flushes = 0

            def flush(self):
                self.flushes += 1
                super().flush()

        logger = logging.getLogger(LOGGER_NAME)
        before = (logger.level, logger.disabled, logger.propagate, tuple(logger.handlers))
        output = Stream()
        with redirect_stdout(output):
            with self.assertRaises(KeyboardInterrupt), run_progress_console():
                phase(1, 1, "公式生成中...")
                self.assertIn("阶段[1] 公式生成中", output.getvalue())
                self.assertGreater(output.flushes, 0)
                raise KeyboardInterrupt
        self.assertEqual((logger.level, logger.disabled, logger.propagate, tuple(logger.handlers)), before)

    def test_output_failure_does_not_escape_into_execution(self):
        class BrokenStream(io.StringIO):
            def write(self, message):
                raise BrokenPipeError("closed by consumer")

        with redirect_stdout(BrokenStream()), run_progress_console():
            phase(1, 3, "回测已接收")
            phase(1, 3, "继续推进")
