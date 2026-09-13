from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from generation.parser import FormulaSyntaxError, parse_formula


class FormulaParserTests(unittest.TestCase):
    def test_normalizes_representative_formulas(self) -> None:
        cases = {
            " ( a + b ) / ( 1 + ( c - d ) ) ": "(a+b)/(1+(c-d))",
            "1 + rank(close)": "1+rank(close)",
            "rank(1 + ts_rank(close, 5))": "rank(1+ts_rank(close,5))",
            "rank(close) + 1e-3": "rank(close)+1e-3",
            "1e-3 - rank(close)": "1e-3-rank(close)",
            "normalize(close, useStd=true, limit=0.5)": (
                "normalize(close,useStd=true,limit=0.5)"
            ),
            "quantile(ts_rank(volume,20), driver=gaussian, sigma=1.0)": (
                "quantile(ts_rank(volume,20),driver=gaussian,sigma=1.0)"
            ),
            "trade_when(close > ts_mean(close,20), rank(volume), -1)": (
                "trade_when(close>ts_mean(close,20),rank(volume),-1)"
            ),
            "if_else(close>=open, ts_delta(close,5), -ts_delta(open,5))": (
                "if_else(close>=open,ts_delta(close,5),-ts_delta(open,5))"
            ),
            'group_neutralize(close, group="sector level")': (
                'group_neutralize(close,group="sector level")'
            ),
        }

        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(parse_formula(source).normalized, expected)

    def test_parse_render_parse_roundtrip_preserves_structure(self) -> None:
        formulas = (
            "a-(b-c)",
            "(a+b)/(1+(c-d))",
            "normalize(close,useStd=true,limit=0.5)",
            "trade_when(close>ts_mean(close,20),rank(volume),-1)",
        )

        for formula in formulas:
            with self.subTest(formula=formula):
                parsed = parse_formula(formula)
                reparsed = parse_formula(parsed.normalized)
                self.assertEqual(reparsed.expression, parsed.expression)

    def test_literals_are_not_reported_as_referenced_names(self) -> None:
        parsed = parse_formula(
            'normalize(close,useStd=true,limit=1e-3,label="Market Value")'
        )

        self.assertEqual(parsed.facts.referenced_names, ("close",))
        self.assertEqual(parsed.facts.operator_names, ("normalize",))

    def test_equivalent_whitespace_has_the_same_fingerprint(self) -> None:
        compact = parse_formula("rank(ts_delta(close,5))")
        spaced = parse_formula(" rank( ts_delta( close , 5 ) ) ")

        self.assertEqual(compact.normalized, spaced.normalized)
        self.assertEqual(compact.fingerprint, spaced.fingerprint)

    def test_reports_specific_syntax_errors(self) -> None:
        cases = {
            "": "empty_formula",
            "a+": "missing_expression",
            "(a+b": "unclosed_parenthesis",
            "rank(,a)": "empty_argument",
            "rank(a,)": "empty_argument",
            "a$b": "unexpected_character",
            "normalize(close,limit=1,limit=2)": "duplicate_named_argument",
            'group_neutralize(close,group="sector)': "unterminated_string",
        }

        for formula, expected_code in cases.items():
            with self.subTest(formula=formula):
                with self.assertRaises(FormulaSyntaxError) as raised:
                    parse_formula(formula)
                self.assertEqual(raised.exception.code, expected_code)
                self.assertGreaterEqual(raised.exception.position, 0)


if __name__ == "__main__":
    unittest.main()
