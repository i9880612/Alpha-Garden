from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from worldquant.config import load_worldquant_connection_settings


class WorldQuantConnectionSettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.environment_path = Path(self.temporary_directory.name) / ".env"

    def test_loads_basic_credentials_without_exposing_them(self) -> None:
        self.environment_path.write_text(
            "\n".join(
                (
                    "WQB_ACCOUNT_SCOPE=group-account",
                    "WQB_BASE_URL=https://api.worldquantbrain.com",
                    "WQB_EMAIL=user@example.com",
                    "WQB_PASSWORD=secret",
                    "WQB_SESSION_TOKEN=",
                )
            ),
            encoding="utf-8",
        )

        settings = load_worldquant_connection_settings(self.environment_path)

        self.assertEqual(settings.base_url, "https://api.worldquantbrain.com")
        self.assertEqual(settings.account_scope, "group-account")
        self.assertEqual(settings.credentials.username, "user@example.com")
        self.assertEqual(settings.credentials.password, "secret")
        self.assertNotIn("secret", repr(settings))

    def test_loads_bearer_token_as_the_only_credential(self) -> None:
        self.environment_path.write_text(
            "\n".join(
                (
                    "WQB_ACCOUNT_SCOPE=group-account",
                    "WQB_BASE_URL=https://api.worldquantbrain.com/",
                    "WQB_EMAIL=",
                    "WQB_PASSWORD=",
                    "WQB_SESSION_TOKEN=token-value",
                )
            ),
            encoding="utf-8",
        )

        settings = load_worldquant_connection_settings(self.environment_path)

        self.assertEqual(settings.credentials.bearer_token, "token-value")
        self.assertNotIn("token-value", repr(settings))

    def test_ignores_unrelated_settings_in_the_shared_file(self) -> None:
        self.environment_path.write_text(
            "\n".join(
                (
                    "WQB_EXECUTION_MODE=explicit",
                    "WQB_ACCOUNT_SCOPE=group-account",
                    "WQB_BASE_URL=https://api.worldquantbrain.com",
                    "WQB_EMAIL=user@example.com",
                    "WQB_PASSWORD=secret",
                    "WQB_SESSION_TOKEN=",
                    "WQB_SUPPRESS_RUN_LOGGING=1",
                )
            ),
            encoding="utf-8",
        )

        settings = load_worldquant_connection_settings(self.environment_path)

        self.assertEqual(settings.credentials.username, "user@example.com")

    def test_rejects_duplicate_owned_keys(self) -> None:
        self.environment_path.write_text(
            "WQB_ACCOUNT_SCOPE=group-account\n"
            "WQB_BASE_URL=https://api.worldquantbrain.com\n"
            "WQB_BASE_URL=https://example.com\n"
            "WQB_SESSION_TOKEN=token",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            ValueError,
            "worldquant_environment_key_duplicate",
        ):
            load_worldquant_connection_settings(self.environment_path)

    def test_requires_explicit_non_sensitive_account_scope(self) -> None:
        self.environment_path.write_text(
            "WQB_BASE_URL=https://api.worldquantbrain.com\n"
            "WQB_SESSION_TOKEN=token",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "worldquant_account_scope_missing"):
            load_worldquant_connection_settings(self.environment_path)


if __name__ == "__main__":
    unittest.main()
