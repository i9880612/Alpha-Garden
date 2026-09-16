export const grades = ["SPECTACULAR", "EXCELLENT", "GOOD", "AVERAGE", "INFERIOR"];
export const gradeColors: Record<string, string> = { SPECTACULAR: "#ef6468", EXCELLENT: "#2f9e65", GOOD: "#a1dc88", AVERAGE: "#5b9bea", INFERIOR: "#a0a5af", UNKNOWN: "#777b89" };
export const gradeTagColors: Record<string, string> = { SPECTACULAR: "red", EXCELLENT: "green", GOOD: "lime", AVERAGE: "blue", INFERIOR: "default", UNKNOWN: "default" };
export const runStatusTagColors: Record<string, string> = { created: "default", running: "green", stopping: "default", paused: "default", failed: "red", completed: "green" };
export const progressGradients = [["#4354f7", "#949aff"], ["#bf8730", "#f1d45c"], ["#723bdc", "#ba91f1"], ["#168d87", "#74c9c3"]];
export const gradeName = (grade?: string | null) => grade ? grade[0] + grade.slice(1).toLowerCase() : "—";
export const metric = (value?: number | null, percent = false) => value == null ? "—" : percent ? `${(value * 100).toFixed(1)}%` : value.toFixed(2);
export const timestamp = (value?: string | null) => value ? new Date(value).toLocaleString() : "—";
export const logMessage = (message: string) => message
  .replace(/^(结果列：.*)；(通过\/未通过\/待定.*)$/, "$1\n说明：$2")
  .replace(
  /\bS\s*(-?\d+\.\d+) F\s*(-?\d+\.\d+) T\s*(-?\d+\.\d+%)$/,
  (_, sharpe: string, fitness: string, turnover: string) =>
    `Sharpe ${sharpe.padStart(6)}  Fitness ${fitness.padStart(6)}  Turnover ${turnover.padStart(6)}`,
);
export const sourceName = (source: string | undefined, zh: boolean) => ({ exploration: ["探索", "Exploration"], sc: ["SC 治理", "SC repair"], mutation: ["变异", "Mutation"], reversal: ["反转", "Reversal"] })[source || ""]?.[zh ? 0 : 1] || "—";
export function mutationDescription({ action, before, after }: { action: string; before: string; after: string }, zh: boolean) {
  const beforeOperator = /^\s*([A-Za-z_][\w]*)\s*\(/.exec(before)?.[1];
  const afterOperator = /^\s*([A-Za-z_][\w]*)\s*\(/.exec(after)?.[1];
  if (action === "internal_operator_replacement" && beforeOperator && afterOperator && beforeOperator !== afterOperator) {
    return zh ? `将这段公式中的 ${beforeOperator} 算子替换为 ${afterOperator}。` : `Replace ${beforeOperator} with ${afterOperator} in this part of the formula.`;
  }
  if (action === "internal_field_replacement") {
    return zh ? `将数据字段 ${before} 替换为 ${after}。` : `Replace the data field ${before} with ${after}.`;
  }
  if (action === "internal_layer_removal" && beforeOperator) {
    return zh ? `移除这段公式外层的 ${beforeOperator} 算子，保留内部表达式。` : `Remove the outer ${beforeOperator} operator from this part of the formula and keep its inner expression.`;
  }
  if (action === "single_window_mutation") {
    return zh ? `将时间窗口从 ${before} 调整为 ${after}。` : `Change the time window from ${before} to ${after}.`;
  }
  const names: Record<string, [string, string]> = {
    temporal_change_reframe: ["提取信号的时序变化", "Extract changes in the signal over time"],
    temporal_persistence_reframe: ["平滑信号并提取持续性", "Smooth the signal to capture persistence"],
    distribution_stabilization: ["控制极端值并稳定信号分布", "Control outliers and stabilize the signal distribution"],
    group_relative_reframe: ["转换为组内相对信号", "Express the signal relative to its group"],
    complementary_signal_reframe: ["引入互补信号并重新归一化", "Add a complementary signal and renormalize"],
    internal_operator_replacement: ["替换公式内部算子", "Replace an operator within the formula"],
    internal_layer_removal: ["移除一层算子以简化公式", "Remove an operator layer to simplify the formula"],
    direction_reversal: ["反转信号方向", "Reverse the signal direction"],
    self_correlation_conflict_reference_residual: ["针对冲突参考信号进行残差化", "Residualize against the conflicting reference signal"],
    self_correlation_shared_field_replacement: ["替换共享字段以降低自相关性", "Replace a shared field to reduce self-correlation"],
    self_correlation_shared_field_operator_replacement: ["替换共享字段及算子以降低自相关性", "Replace a shared field and operator to reduce self-correlation"],
    self_correlation_shared_left_replacement: ["替换共享左侧表达式以降低自相关性", "Replace the shared left expression to reduce self-correlation"],
    self_correlation_shared_right_replacement: ["替换共享右侧表达式以降低自相关性", "Replace the shared right expression to reduce self-correlation"],
    self_correlation_industry_rank: ["采用行业内排序以降低自相关性", "Apply within-industry ranking to reduce self-correlation"],
    self_correlation_time_smoothing: ["采用时序平滑以降低自相关性", "Apply time-series smoothing to reduce self-correlation"],
    self_correlation_industry_neutralization: ["采用行业中性化以降低自相关性", "Apply industry neutralization to reduce self-correlation"],
  };
  return names[action]?.[zh ? 0 : 1] || action.replaceAll("_", " ");
}
export function checkFailureName(name: string, zh: boolean) {
  const names: Record<string, [string, string]> = {
    SELF_CORRELATION: ["自相关性未通过", "Self-correlation failed"],
    LOW_SHARPE: ["夏普不足", "Low Sharpe"],
    LOW_FITNESS: ["适应度不足", "Low fitness"],
    LOW_SUB_UNIVERSE_SHARPE: ["子股票池夏普不足", "Low sub-universe Sharpe"],
    CONCENTRATED_WEIGHT: ["权重过度集中", "Concentrated weight"],
    HIGH_TURNOVER: ["换手率过高", "High turnover"],
    LOW_TURNOVER: ["换手率过低", "Low turnover"],
    MATCHES_COMPETITION: ["不符合竞赛要求", "Competition requirements not met"],
  };
  return names[name]?.[zh ? 0 : 1] || (zh ? `${name} 未通过` : `${name} failed`);
}
export function statusName(status: string | undefined, zh: boolean) {
  if (!status) return "—";
  const names: Record<string, string> = { created: "已创建", running: "运行中", completed: "已完成", failed: "失败", pending: "待定", passed: "通过", paused: "已暂停", stopping: "暂停中", needs_attention: "需处理", submitted: "已提交", ineligible: "不符合提交条件", check_pending: "等待复检", detail_pending: "读取详情", ready: "待提交", submitting: "提交中", confirmation_pending: "等待确认", submission_unknown: "结果未知" };
  return zh ? names[status] || status : status.replaceAll("_", " ");
}

export const chartPalette = (theme: "light" | "dark") => ({
  text: theme === "dark" ? "#a5a6b0" : "#646778",
  grid: theme === "dark" ? "rgba(255,255,255,.08)" : "rgba(0,0,0,.08)",
  background: theme === "dark" ? "#101014" : "#ffffff",
});
