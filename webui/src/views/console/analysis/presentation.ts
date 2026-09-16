export const groupStyles: Record<string, { dark: string; light: string }> = {
  both: { dark: "#34d399", light: "#047857" },
  fitness: { dark: "#fbbf24", light: "#a16207" },
  sharpe: { dark: "#60a5fa", light: "#2563eb" },
  neither: { dark: "#fb7185", light: "#be123c" },
  unknown: { dark: "#cbd5e1", light: "#64748b" },
};
export const groupName = (key: string, zh: boolean) => ({ both: ["两项均过线", "Both passed"], fitness: ["仅 Fitness 未过线", "Fitness failed only"], sharpe: ["仅夏普未过线", "Sharpe failed only"], neither: ["两项均未过线", "Both failed"], unknown: ["检查待定 / 缺失", "Unresolved checks"] })[key]?.[zh ? 0 : 1] || key;
export const stateColors: Record<string, string> = { passed: "#80bba8", failed: "#bb8298", pending: "#c9b46e", missing: "#818c9f", error: "#d6956a", submitted: "#868fdb", no_references: "#818c9f" };
export const stateName = (key: string, zh: boolean) => ({ passed: ["通过", "Passed"], failed: ["未通过", "Failed"], pending: ["待定", "Pending"], missing: ["未取得", "Unavailable"], error: ["请求异常", "Request error"], submitted: ["已提交", "Submitted"], no_references: ["无比较对象", "No references"] })[key]?.[zh ? 0 : 1] || key;
export const quantile = (values: number[], p: number) => {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b), position = (sorted.length - 1) * p, lower = Math.floor(position);
  return sorted[lower] + (sorted[Math.ceil(position)] - sorted[lower]) * (position - lower);
};

export const turnoverBins = Array.from({ length: 9 }, (_, index) => ({ key: String(index), min: index / 10, max: index === 8 ? Infinity : (index + 1) / 10, label: index === 8 ? "≥80%" : `${index * 10}–${(index + 1) * 10}%` }));
