from __future__ import annotations

from dataclasses import dataclass
import re

from generation.formula import (
    Binary,
    Call,
    CallArgument,
    Expression,
    FormulaFacts,
    Literal,
    Name,
    Prefix,
    analyze_formula,
    formula_fingerprint,
    render_formula,
)


@dataclass(frozen=True, slots=True)
class ParsedFormula:
    expression: Expression
    normalized: str
    fingerprint: str
    facts: FormulaFacts


class FormulaSyntaxError(ValueError):
    def __init__(self, code: str, position: int) -> None:
        self.code = code
        self.position = position
        super().__init__(f"{code} at position {position}")


@dataclass(frozen=True, slots=True)
class _Token:
    kind: str
    value: str
    position: int


_NUMBER = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_BINARY_PRECEDENCE = {
    ">": 10,
    ">=": 10,
    "<": 10,
    "<=": 10,
    "==": 10,
    "!=": 10,
    "+": 20,
    "-": 20,
    "*": 30,
    "/": 30,
}


def parse_formula(formula: str) -> ParsedFormula:
    tokens = _tokenize(formula)
    parser = _Parser(tokens)
    expression = parser.parse()
    normalized = render_formula(expression)
    return ParsedFormula(
        expression=expression,
        normalized=normalized,
        fingerprint=formula_fingerprint(expression),
        facts=analyze_formula(expression),
    )


class _Parser:
    def __init__(self, tokens: tuple[_Token, ...]) -> None:
        self._tokens = tokens
        self._index = 0

    def parse(self) -> Expression:
        if self._current.kind == "EOF":
            raise FormulaSyntaxError("empty_formula", 0)
        expression = self._parse_expression()
        if self._current.kind != "EOF":
            raise FormulaSyntaxError("unexpected_token", self._current.position)
        return expression

    @property
    def _current(self) -> _Token:
        return self._tokens[self._index]

    def _peek(self, offset: int = 1) -> _Token:
        index = min(self._index + offset, len(self._tokens) - 1)
        return self._tokens[index]

    def _advance(self) -> _Token:
        token = self._current
        self._index += 1
        return token

    def _parse_expression(self, minimum_precedence: int = 0) -> Expression:
        left = self._parse_prefix()
        while self._current.kind == "OPERATOR":
            precedence = _BINARY_PRECEDENCE[self._current.value]
            if precedence < minimum_precedence:
                break
            operator = self._advance().value
            if self._current.kind in {"EOF", "COMMA", "RPAREN"}:
                raise FormulaSyntaxError("missing_expression", self._current.position)
            right = self._parse_expression(precedence + 1)
            left = Binary(operator, left, right)
        return left

    def _parse_prefix(self) -> Expression:
        if self._current.kind == "OPERATOR" and self._current.value in {"+", "-"}:
            operator = self._advance()
            if self._current.kind in {"EOF", "COMMA", "RPAREN"}:
                raise FormulaSyntaxError("missing_expression", self._current.position)
            return Prefix(operator.value, self._parse_prefix())
        return self._parse_primary()

    def _parse_primary(self) -> Expression:
        token = self._current
        if token.kind == "NUMBER":
            self._advance()
            kind = "float" if any(marker in token.value for marker in ".eE") else "integer"
            return Literal(token.value, kind)
        if token.kind == "STRING":
            self._advance()
            return Literal(token.value, "string")
        if token.kind == "NAME":
            self._advance()
            if token.value.lower() in {"true", "false"}:
                return Literal(token.value.lower(), "boolean")
            if self._current.kind == "LPAREN":
                return self._parse_call(token)
            return Name(token.value)
        if token.kind == "LPAREN":
            opening = self._advance()
            if self._current.kind == "RPAREN":
                raise FormulaSyntaxError("empty_group", self._current.position)
            expression = self._parse_expression()
            if self._current.kind != "RPAREN":
                raise FormulaSyntaxError("unclosed_parenthesis", opening.position)
            self._advance()
            return expression
        raise FormulaSyntaxError("missing_expression", token.position)

    def _parse_call(self, operator: _Token) -> Call:
        self._advance()
        arguments: list[CallArgument] = []
        argument_names: set[str] = set()
        if self._current.kind == "RPAREN":
            self._advance()
            return Call(operator.value, ())

        while True:
            if self._current.kind in {"COMMA", "RPAREN", "EOF"}:
                raise FormulaSyntaxError("empty_argument", self._current.position)

            argument_name: str | None = None
            if self._current.kind == "NAME" and self._peek().kind == "EQUALS":
                name_token = self._advance()
                self._advance()
                argument_name = name_token.value
                if argument_name in argument_names:
                    raise FormulaSyntaxError(
                        "duplicate_named_argument",
                        name_token.position,
                    )
                argument_names.add(argument_name)
                if self._current.kind in {"COMMA", "RPAREN", "EOF"}:
                    raise FormulaSyntaxError("empty_argument", self._current.position)

            arguments.append(
                CallArgument(
                    value=self._parse_expression(),
                    name=argument_name,
                )
            )
            if self._current.kind == "RPAREN":
                self._advance()
                break
            if self._current.kind != "COMMA":
                raise FormulaSyntaxError("unclosed_parenthesis", operator.position)
            self._advance()
            if self._current.kind == "RPAREN":
                raise FormulaSyntaxError("empty_argument", self._current.position)

        return Call(operator.value, tuple(arguments))


def _tokenize(formula: str) -> tuple[_Token, ...]:
    tokens: list[_Token] = []
    index = 0
    while index < len(formula):
        character = formula[index]
        if character.isspace():
            index += 1
            continue
        if character in {"'", '"'}:
            token, index = _read_string(formula, index)
            tokens.append(token)
            continue

        number = _NUMBER.match(formula, index)
        if number is not None:
            tokens.append(_Token("NUMBER", number.group(0), index))
            index = number.end()
            continue

        name = _NAME.match(formula, index)
        if name is not None:
            tokens.append(_Token("NAME", name.group(0), index))
            index = name.end()
            continue

        two_characters = formula[index : index + 2]
        if two_characters in {">=", "<=", "==", "!="}:
            tokens.append(_Token("OPERATOR", two_characters, index))
            index += 2
            continue
        if character in "+-*/><":
            tokens.append(_Token("OPERATOR", character, index))
            index += 1
            continue
        if character == "(":
            tokens.append(_Token("LPAREN", character, index))
            index += 1
            continue
        if character == ")":
            tokens.append(_Token("RPAREN", character, index))
            index += 1
            continue
        if character == ",":
            tokens.append(_Token("COMMA", character, index))
            index += 1
            continue
        if character == "=":
            tokens.append(_Token("EQUALS", character, index))
            index += 1
            continue
        raise FormulaSyntaxError("unexpected_character", index)

    tokens.append(_Token("EOF", "", len(formula)))
    return tuple(tokens)


def _read_string(formula: str, start: int) -> tuple[_Token, int]:
    quote = formula[start]
    index = start + 1
    while index < len(formula):
        if formula[index] == "\\":
            index += 2
            continue
        if formula[index] == quote:
            end = index + 1
            return _Token("STRING", formula[start:end], start), end
        index += 1
    raise FormulaSyntaxError("unterminated_string", start)
