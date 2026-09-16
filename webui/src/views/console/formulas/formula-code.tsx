import { Typography } from "antd";
import type { ConsoleTheme } from "@/layouts/console/theme";

export default function FormulaCode({ expression, theme }: { expression: string; theme: ConsoleTheme }) {
  const colors = theme === "dark"
    ? { function: "#82aaff", field: "#c3e88d", number: "#f78c6c", string: "#ecc48d", operator: "#c792ea" }
    : { function: "#2457a7", field: "#376c32", number: "#a34816", string: "#805715", operator: "#7b3ca6" };
  const tokens = [...expression.matchAll(/"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|[A-Za-z_][\w]*|[+*/^%=!<>?:&|~-]+|\s+|./g)];
  return <Typography.Paragraph className="ag-formula-expression" copyable={{ text: expression }} style={{ margin: 0 }}>
    <code className="ag-formula-code">{tokens.map((match, index) => {
      const value = match[0];
      const kind = /^["']/.test(value) ? "string" : /^(?:\d|\.\d)/.test(value) ? "number" : /^[A-Za-z_]/.test(value)
        ? /^\s*\(/.test(expression.slice(match.index + value.length)) ? "function" : "field"
        : /^[+*/^%=!<>?:&|~-]/.test(value) ? "operator" : null;
      return <span key={index} style={kind ? { color: colors[kind] } : undefined}>{value}</span>;
    })}</code>
  </Typography.Paragraph>;
}
