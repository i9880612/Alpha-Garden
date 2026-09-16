import type { EChartsCoreOption } from "echarts/core";
import { Chart } from "@/components/research-chart";
import { chartPalette } from "@/api/presentation";
import type { DiagnosticRecord, QualityDiagnosis } from "@/api/analysis";
import type { ConsoleTheme } from "@/layouts/console/theme";

import { groupStyles, groupName, turnoverBins } from "./presentation";

const escape = (text: string) => text.replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);

export function Scatter({ data, theme, zh, hiddenGroups, onOpen }: { data: QualityDiagnosis; theme: ConsoleTheme; zh: boolean; hiddenGroups: Record<string, boolean>; onOpen: (id: string) => void }) {
  const palette = chartPalette(theme);
  const option: EChartsCoreOption = {
    animation: true, animationDuration: 650, animationDurationUpdate: 250,
    grid: { left: 16, right: 32, top: 30, bottom: 32, containLabel: true },
    tooltip: { trigger: "item", confine: true, backgroundColor: palette.background, borderColor: palette.grid, textStyle: { color: palette.text },
      formatter: (event: { seriesName: string; data: { name: string; value: number[] } }) => `${escape(event.data.name)}<br/>${escape(event.seriesName)}<br/>Sharpe ${event.data.value[0].toFixed(2)}<br/>Fitness ${event.data.value[1].toFixed(2)}` },
    xAxis: { type: "value", name: "Sharpe", nameLocation: "middle", nameGap: 28, scale: true, axisLabel: { color: palette.text }, nameTextStyle: { color: palette.text }, splitLine: { lineStyle: { color: palette.grid, type: "dashed" } } },
    yAxis: { type: "value", name: "Fitness", scale: true, axisLabel: { color: palette.text }, nameTextStyle: { color: palette.text }, splitLine: { lineStyle: { color: palette.grid, type: "dashed" } } },
    series: Object.entries(groupStyles).map(([group, style], index) => ({ id: group, name: groupName(group, zh), type: "scatter", symbolSize: 7,
      symbol: "circle",
      itemStyle: { color: style[theme], opacity: .95 },
      emphasis: { scale: 1.6, focus: "series" }, blur: { itemStyle: { opacity: .15 } },
      data: hiddenGroups[group] ? [] : data.records.filter(r => r.metric_group === group && r.sharpe != null && r.fitness != null).map(r => ({ name: r.alpha_id || r.task_id, taskId: r.task_id, value: [r.sharpe, r.fitness] })),
      ...(index === 0 && data.thresholds ? { markLine: { silent: true, symbol: "none", lineStyle: { color: "#999bcb", type: "dashed" }, label: { color: palette.text, position: "insideEndTop" }, data: [{ xAxis: data.thresholds[0], name: "Sharpe" }, { yAxis: data.thresholds[1], name: "Fitness" }] } } : {}),
    })),
  };
  return <Chart className="ag-diagnosis-chart" label={zh ? "夏普与 Fitness 散点图，点击数据点查看详情" : "Sharpe and Fitness scatter plot; click a point for details"} option={option} onClick={event => {
    const point = event.data as { taskId?: string }; if (point?.taskId) onOpen(point.taskId);
  }} />;
}

export function Turnover({ rows, theme, zh, onSelect }: { rows: DiagnosticRecord[]; theme: ConsoleTheme; zh: boolean; onSelect: (key: string) => void }) {
  const palette = chartPalette(theme);
  const option: EChartsCoreOption = { animation: true, animationDuration: 650,
    grid: { left: 8, right: 12, top: 22, bottom: 8, containLabel: true },
    tooltip: { trigger: "axis", confine: true, backgroundColor: palette.background, textStyle: { color: palette.text }, borderColor: palette.grid, axisPointer: { type: "shadow" } },
    xAxis: { type: "category", data: turnoverBins.map(b => b.label), axisLabel: { color: palette.text, fontSize: 11, rotate: 30, interval: 0 }, axisTick: { show: false } },
    yAxis: { type: "value", minInterval: 1, axisLabel: { color: palette.text }, splitLine: { lineStyle: { color: palette.grid, type: "dashed" } } },
    series: [{ id: "turnover", type: "bar", name: zh ? "公式数" : "Formulas", barMaxWidth: 42,
      data: turnoverBins.map(b => rows.filter(r => r.turnover != null && r.turnover >= b.min && r.turnover < b.max).length),
      itemStyle: { color: "#7875cf", borderRadius: [2, 2, 0, 0] }, label: { show: true, position: "top", color: palette.text, formatter: (p: { value: number }) => p.value || "" } }],
  };
  return <Chart className="ag-diagnosis-chart" label={zh ? "换手率分布，最后一组包含全部 80% 及以上结果" : "Turnover distribution, including all values at or above 80%"} option={option} onClick={event => onSelect(turnoverBins[event.dataIndex].key)} />;
}
