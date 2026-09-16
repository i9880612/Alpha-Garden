import tempfile
import unittest

from persistence.console import read_console_database
from persistence.console_research import catalog_page
from tests.execution.console_fixture import make_console_reader


class CatalogReadTests(unittest.TestCase):
    def test_pagination_filters_and_account_scope_use_synced_context(self):
        with tempfile.TemporaryDirectory() as folder:
            reader = make_console_reader(folder)
            before = reader.paths.database.read_bytes()
            with read_console_database(reader.paths.database) as connection:
                def read(**overrides):
                    args = dict(kind="fields", search="", category="", dataset="", page=1, page_size=2)
                    args.update(overrides)
                    return catalog_page(connection, "group-account", **args)
                first, second = read(), read(page=2)
                self.assertEqual(first["total"], 4)
                self.assertEqual(len(second["items"]), 2)
                self.assertFalse({r["field_id"] for r in first["items"]} & {r["field_id"] for r in second["items"]})
                self.assertEqual(read(search="CLOSE", category="sample", dataset="dataset")["items"][0]["field_id"], "close")
                self.assertEqual(read(search="%' OR 1=1 --")["total"], 0)
                self.assertEqual(read(dataset="missing")["items"], [])
                operator = read(kind="operators", search="rank")["items"][0]
                self.assertEqual(operator["definition"], "rank(x)")
                self.assertEqual(operator["parameters"], [{"name": "x", "kind": "expr"}])
                self.assertEqual(operator["roles"], ["cross_sectional_normalization"])
                foreign = catalog_page(connection, "other-account", kind="operators", search="", category="", dataset="", page=1, page_size=10)
                self.assertFalse(foreign["available"])
                self.assertIsNone(foreign["context"])
                self.assertEqual(foreign["items"], [])
                self.assertEqual(foreign["categories"], [])
            self.assertEqual(before, reader.paths.database.read_bytes())
