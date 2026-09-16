"""Synthetic local console data; never copies a user's database or credentials."""
from pathlib import Path
from execution.console import ConsolePaths, ConsoleReader
from persistence.database import open_database
from persistence.schema import initialize_database_schema
from tests.execution.catalog_fixture import initialize_test_generation_catalog

ROOT = Path(__file__).resolve().parents[2]

def make_console_reader(folder):
    folder = Path(folder)
    database = folder / "console.sqlite3"
    environment = folder / "test.env"
    environment.write_text("WQB_ACCOUNT_SCOPE=group-account\nWQB_BASE_URL=https://example.invalid\nWQB_SESSION_TOKEN=synthetic-console-secret\n", encoding="utf-8")
    with open_database(database) as connection:
        initialize_database_schema(connection)
        initialize_test_generation_catalog(connection)
    return ConsoleReader(ConsolePaths(database, ROOT / "config/backtest.default.json",
                                     ROOT / "config/run.default.json", environment))
