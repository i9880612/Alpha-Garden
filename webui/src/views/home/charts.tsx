import * as echarts from "echarts/core";
import type { ConsoleTheme } from "@/layouts/console/theme";
import type { Progress } from "@/api/console";
import { gradeColors, gradeName, progressGradients } from "@/api/presentation";
import { chartPalette as palette } from "@/api/presentation";
import { Chart } from "@/components/research-chart";

export function Sparkline({ values, label }: { values: number[]; label: string }) {
  if (!values.length) return <div className="ag-kpi-spark" role="img" aria-label={label} />;
  return (
    <Chart
      className="ag-kpi-spark"
      label={`${label}: ${values.join(" → ")}`}
      option={{
        animation: true,
        grid: { top: 8, bottom: 4, left: 4, right: 6 },
        xAxis: { type: "category", show: false, boundaryGap: false, data: values.map((_, i) => i) },
        yAxis: { type: "value", show: false, scale: true },
        series: [
          {
            type: "line",
            data: values,
            symbol: "none",
            lineStyle: { color: "#4b5aff", width: 2.2, shadowBlur: 8, shadowColor: "#4354f755" },
            markPoint: {
              symbol: "circle",
              symbolSize: 5,
              silent: true,
              data: values.length ? [{ coord: [values.length - 1, values.at(-1)] }] : [],
              itemStyle: { color: "#6672ff" },
              label: { show: false },
            },
          },
        ],
      }}
    />
  );
}

export function TrendChart({
  theme,
  label,
  days,
  values,
}: {
  theme: ConsoleTheme;
  label: string;
  days: string[];
  values: number[];
}) {
  const colors = palette(theme);
  if (!values.length) return <div className="ag-trend-chart" role="img" aria-label={label} />;
  return (
    <Chart
      className="ag-trend-chart"
      label={`${label}: ${values.join(", ")}`}
      option={{
        animation: true,
        grid: { top: 16, bottom: 8, left: 8, right: 24, containLabel: true },
        tooltip: {
          trigger: "axis",
          confine: true,
          backgroundColor: colors.background,
          borderColor: colors.grid,
          textStyle: { color: colors.text },
        },
        xAxis: {
          type: "category",
          data: days,
          boundaryGap: false,
          axisTick: { show: false },
          axisLine: { lineStyle: { color: colors.grid } },
          axisLabel: { color: colors.text, margin: 14, showMinLabel: true, showMaxLabel: true, hideOverlap: true },
        },
        yAxis: {
          type: "value",
          min: 0,
          axisLabel: { color: colors.text },
          splitLine: { lineStyle: { color: colors.grid, type: "dashed" } },
        },
        series: [
          {
            name: label,
            type: "line",
            data: values,
            symbol: "circle",
            symbolSize: 7,
            itemStyle: { color: colors.background, borderColor: "#5464ff", borderWidth: 3 },
            lineStyle: { color: "#4b5aff", width: 3, shadowColor: "#4354f733", shadowBlur: 10 },
            areaStyle: {
              color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                { offset: 0, color: "#4b5aff25" },
                { offset: 1, color: "#4b5aff00" },
              ]),
            },
          },
        ],
      }}
    />
  );
}

export function GradeChart({
  theme,
  hidden,
  label,
  values,
}: {
  theme: ConsoleTheme;
  hidden: string[];
  label: string;
  values: { label: string; value: number }[];
}) {
  const colors = palette(theme);
  const grades = values.filter((item) => !hidden.includes(item.label));
  if (!values.length) return <div className="ag-donut-chart" role="img" aria-label={label} />;
  return (
    <Chart
      className="ag-donut-chart"
      label={`${label}: ${grades.map((item) => `${item.label} ${item.value}`).join(", ")}`}
      option={{
        animation: true,
        tooltip: {
          trigger: "item",
          confine: true,
          backgroundColor: colors.background,
          borderColor: colors.grid,
          textStyle: { color: colors.text },
          formatter: `${label}<br/>{b}: {c} ({d}%)`,
        },
        series: [
          {
            type: "pie",
            radius: ["62%", "84%"],
            center: ["50%", "50%"],
            label: { show: false },
            emphasis: { scaleSize: 3 },
            data: grades.map((item) => ({
              name: gradeName(item.label),
              value: item.value,
              itemStyle: { color: gradeColors[item.label] },
            })),
          },
        ],
      }}
    />
  );
}

export function ProgressChart({
  theme,
  labels,
  label,
  values,
}: {
  theme: ConsoleTheme;
  labels: string[];
  label: string;
  values: Progress[];
}) {
  const colors = palette(theme);
  if (!values.length) return <div className="ag-health-chart" role="img" aria-label={label} />;
  return (
    <Chart
      className="ag-health-chart"
      label={`${label}: ${labels.map((name, i) => `${name} ${values[i].completed}/${values[i].planned}`).join(", ")}`}
      option={{
        animation: true,
        grid: { top: 4, bottom: 4, left: 0, right: 8, containLabel: true },
        xAxis: { type: "value", min: 0, max: 100, show: false },
        yAxis: [
          {
            type: "category",
            data: labels,
            inverse: true,
            axisLine: { show: false },
            axisTick: { show: false },
            axisLabel: { color: colors.text, fontSize: 13, margin: 16 },
          },
          {
            type: "category",
            position: "right",
            data: values.map((item) => `${item.completed}/${item.planned}`),
            inverse: true,
            axisLine: { show: false },
            axisTick: { show: false },
            axisLabel: { color: colors.text, fontSize: 12, margin: 16 },
          },
        ],
        series: [
          {
            type: "bar",
            barWidth: 7,
            showBackground: true,
            backgroundStyle: { color: colors.grid, borderRadius: 5 },
            data: values.map((item, i) => ({
              value: item.planned ? (item.completed / item.planned) * 100 : 0,
              itemStyle: {
                borderRadius: 5,
                color: new echarts.graphic.LinearGradient(
                  0,
                  0,
                  1,
                  0,
                  progressGradients[i].map((color, offset) => ({ offset, color })),
                ),
              },
            })),
          },
        ],
      }}
    />
  );
}
