import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from execution.process_lock import exclusive_run_process


class RunProcessLockTests(unittest.TestCase):
    def test_process_death_releases_lock_without_deleting_state_or_pid_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "research.sqlite3"
            source = str(Path(__file__).resolve().parents[2] / "src")
            code = (
                f"import sys,time;sys.path.insert(0,{source!r});"
                "from execution.process_lock import exclusive_run_process\n"
                f"with exclusive_run_process({str(path)!r}):\n"
                " print('locked',flush=True)\n time.sleep(30)\n"
            )
            process = subprocess.Popen([sys.executable, "-I", "-c", code],
                                       stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(process.stdout.readline().strip(), "locked")
                with self.assertRaisesRegex(ValueError, "已有命令"):
                    with exclusive_run_process(path):
                        self.fail("two drivers acquired the same lock")
            finally:
                process.terminate()
                process.wait(timeout=5)
                process.stdout.close()
            with exclusive_run_process(path):
                self.assertFalse(path.exists())
