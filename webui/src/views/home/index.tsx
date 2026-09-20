import {
  BarChartOutlined,
  DatabaseOutlined,
  ExperimentOutlined,
  LineChartOutlined,
  PieChartOutlined,
  RightOutlined,
  SendOutlined,
  TrophyOutlined,
} from "@ant-design/icons";
import { Tag, Tooltip } from "antd";
import { useState } from "react";
import { Link, useOutletContext } from "react-router-dom";
import { consoleMessages, type ConsoleLocale } from "@/layouts/console/locale";
import type { ConsoleTheme } from "@/layouts/console/theme";
import { GradeChart, ProgressChart, Sparkline, TrendChart } from "./charts";
import { ModuleIcon } from "./module-icon";
import { useApi } from "@/api/use-api";
import { useConsole } from "@/api/use-console";
import type {
  DashboardActivity,
  DashboardProgress,
  DashboardResearch,
  RecentBacktests,
  SubmittedGrades,
} from "@/api/console";
import {
  gradeColors,
  gradeTagColors,
  gradeName,
  metric,
  sourceName,
  timestamp,
} from "@/api/presentation";
import "./index.css";

const moduleIcons = [
  <ModuleIcon kind="run" />,
  <ModuleIcon kind="library" />,
  <ModuleIcon kind="submit" />,
  <ModuleIcon kind="settings" />,
];
const metricIcons = [
  <BarChartOutlined />,
  <SendOutlined />,
  <DatabaseOutlined />,
  <TrophyOutlined />,
];

const HomePage = () => {
  const { theme, locale } = useOutletContext<{ theme: ConsoleTheme; locale: ConsoleLocale }>();
  const { home: m } = consoleMessages[locale];
  const zh = locale === "zh-CN";
  const { events } = useConsole();
  const activity = useApi<DashboardActivity>("/api/dashboard/activity");
  const submitted = useApi<SubmittedGrades>("/api/dashboard/submitted-grades");
  const research = useApi<DashboardResearch>("/api/dashboard/research");
  const batch = useApi<DashboardProgress>("/api/dashboard/progress");
  const recent = useApi<RecentBacktests>("/api/dashboard/recent");
  const grades = submitted.data?.grades || [];
  const progress = batch.data?.progress || [];
  const localLabel = zh ? "本地记录" : "Local records";
  const statusLabel = (loading: boolean) =>
    events.failed
      ? zh
        ? "实时连接不可用"
        : "Live updates unavailable"
      : !events.ready || loading
        ? zh
          ? "正在加载…"
          : "Loading…"
        : zh
          ? "数据不可用"
          : "Unavailable";
  const [hiddenGrades, setHiddenGrades] = useState<string[]>([]);
  const visibleTotal = grades
    .filter((item) => !hiddenGrades.includes(item.label))
    .reduce((sum, item) => sum + item.value, 0);
  const completed = progress.reduce((sum, item) => sum + item.completed, 0);
  const planned = progress.reduce((sum, item) => sum + item.planned, 0);
  const toggleGrade = (grade: string) =>
    setHiddenGrades((current) =>
      current.includes(grade) ? current.filter((item) => item !== grade) : [...current, grade],
    );
  const researchItems = [
    {
      label: zh ? "剩余优化次数" : "Attempts remaining",
      value: research.data?.optimization_attempts,
      tone: "blue",
    },
    {
      label: zh ? "可手动提交" : "Manual candidates",
      value: research.data?.manual_candidates,
      tone: "yellow",
    },
    {
      label: zh ? "近 7 天回测" : "Backtests · 7 days",
      value: activity.data?.weekly_backtests,
      tone: "cyan",
    },
    {
      label: zh ? "近 7 天提交" : "Submitted · 7 days",
      value: activity.data?.weekly_submissions,
      tone: "purple",
    },
  ];
  const gradeScope = zh
    ? "按当前账号已提交记录中的平台评级统计；缺失评级单独列为未知。"
    : "Platform grades from this account's submitted records; missing grades are shown as unknown.";
  return (
    <section className={`ag-home ag-home--${theme}`}>
      <div className="ag-dashboard">
        <section
          className="ag-top-kpis"
          aria-label={locale === "zh-CN" ? "研究概览" : "Research overview"}
        >
          {m.metrics.map((title, i) => (
            <article className="ag-card ag-kpi-card" key={title}>
              <div className="ag-kpi-content">
                <div className="ag-kpi-title">
                  <span className={`ag-kpi-icon is-${m.modules[i].tone}`}>{metricIcons[i]}</span>
                  {title}
                </div>
                <div className="ag-kpi-value">
                  <span>{activity.data?.metrics[i].value ?? "—"}</span>
                  <small>{m.countUnit}</small>
                </div>
                <div className="ag-kpi-meta">
                  {activity.data ? localLabel : statusLabel(activity.loading)}
                </div>
              </div>
              <Sparkline
                values={activity.data?.metrics[i].spark || []}
                label={`${title} · ${zh ? "近 7 天记录日期分布" : "Record dates over 7 days"}`}
              />
            </article>
          ))}
        </section>
        <section className="ag-main-grid">
          <article className="ag-card ag-trend-card">
            <div className="ag-card-title-row">
              <h2 className="ag-card-title">
                <LineChartOutlined className="is-blue" />
                {m.trend}
              </h2>
              <span className="ag-card-select">
                {activity.data ? m.period : statusLabel(activity.loading)}
              </span>
            </div>
            <TrendChart
              theme={theme}
              label={m.trend}
              days={activity.data?.dates.map((day) => day.slice(5)) || []}
              values={activity.data?.trend || []}
            />
          </article>
          <article className="ag-card ag-donut-card">
            <div className="ag-card-title-row">
              <Tooltip title={gradeScope} trigger={["hover", "focus"]}>
                <h2 className="ag-card-title" tabIndex={0}>
                  <PieChartOutlined className="is-cyan" />
                  {m.distribution}
                </h2>
              </Tooltip>
              <span className="ag-sample-label">{localLabel}</span>
            </div>
            <div className="ag-donut-body">
              <div className="ag-donut-shell">
                <GradeChart
                  theme={theme}
                  hidden={hiddenGrades}
                  label={m.distribution}
                  values={grades}
                />
                <div className="ag-donut-center">
                  <span className="ag-donut-value">{submitted.data ? visibleTotal : "—"}</span>
                  <span className="ag-donut-label">{zh ? "已提交公式" : "Submitted"}</span>
                </div>
              </div>
              <div className="ag-legend">
                {grades.map((item) => (
                  <button
                    key={item.label}
                    type="button"
                    className={`ag-legend-item${hiddenGrades.includes(item.label) ? " is-inactive" : ""}`}
                    aria-pressed={!hiddenGrades.includes(item.label)}
                    onClick={() => toggleGrade(item.label)}
                  >
                    <span
                      className="ag-legend-dot"
                      style={{
                        color: gradeColors[item.label],
                        background: gradeColors[item.label],
                      }}
                    />
                    <span className="ag-legend-label">{gradeName(item.label)}</span>
                    <span className="ag-legend-value">{item.value}</span>
                  </button>
                ))}
                {!grades.length && (
                  <span className="ag-empty-grades">
                    {submitted.data
                      ? zh
                        ? "暂无已提交公式"
                        : "No submitted Alphas"
                      : statusLabel(submitted.loading)}
                  </span>
                )}
              </div>
            </div>
          </article>
          <article className="ag-card ag-research-card">
            <div className="ag-card-title-row">
              <h2 className="ag-card-title">
                <ExperimentOutlined className="is-yellow" />
                {m.research}
              </h2>
              <Link
                className="ag-research-link"
                to="/formulas"
                aria-label={zh ? "查看公式库" : "Open formula library"}
              >
                <RightOutlined />
              </Link>
            </div>
            <div className="ag-research-body">
              <div className="ag-research-focus">
                <span>{zh ? "可优化父代" : "Parents to optimize"}</span>
                <strong>{research.data?.optimization_parents ?? "—"}</strong>
              </div>
              <div className="ag-research-grid">
                {researchItems.map((item) => (
                  <div key={item.label}>
                    <span>{item.label}</span>
                    <strong className={`is-${item.tone}`}>
                      {item.value?.toLocaleString() ?? "—"}
                    </strong>
                  </div>
                ))}
              </div>
            </div>
            <p className="ag-research-caption">
              {research.data
                ? `${zh ? "更新于" : "Updated"} ${timestamp(research.data.observed_at)}`
                : statusLabel(research.loading)}
            </p>
          </article>
          <article className="ag-card ag-health-card">
            <div className="ag-card-title-row">
              <h2 className="ag-card-title">
                <BarChartOutlined className="is-purple" />
                {m.currentRun}
              </h2>
              <span className="ag-progress-summary">
                {batch.data?.run
                  ? batch.data.cycle_number != null
                    ? `${zh ? "最近 · 第" : "Latest · Cycle "}${batch.data.cycle_number}${zh ? "轮" : ""}`
                    : zh
                      ? "尚未生成批次"
                      : "No batch planned"
                  : batch.data
                    ? zh
                      ? "暂无运行"
                      : "No runs"
                    : statusLabel(batch.loading)}{" "}
                <strong>
                  {batch.data ? completed : "—"}
                  <span> / {batch.data ? planned : "—"}</span>
                </strong>
              </span>
            </div>
            <ProgressChart
              theme={theme}
              labels={progress.map((item) => sourceName(item.source, zh))}
              label={m.currentRun}
              values={progress}
            />
          </article>
          <article className="ag-card ag-queue-card">
            <div className="ag-card-title-row">
              <h2 className="ag-card-title">
                <ExperimentOutlined className="is-cyan" />
                {m.recent}
              </h2>
              <span className="ag-sample-label">{localLabel}</span>
            </div>
            <div className="ag-preview-results">
              {!recent.data && (
                <span className="ag-empty-grades">{statusLabel(recent.loading)}</span>
              )}
              {recent.data?.items.length === 0 && (
                <span className="ag-empty-grades">
                  {zh ? "暂无回测结果" : "No backtest results"}
                </span>
              )}
              {recent.data?.items.map((item) => (
                <div key={item.task_id} className="ag-preview-result">
                  <TrophyOutlined style={{ color: gradeColors[item.grade || "UNKNOWN"] }} />
                  <span className="ag-result-id">{item.alpha_id}</span>
                  <Tooltip
                    title={<code className="ag-formula-tooltip">{item.formula}</code>}
                    trigger={["hover", "focus"]}
                    styles={{ root: { maxWidth: "min(640px, calc(100vw - 32px))" } }}
                  >
                    <span className="ag-result-formula" tabIndex={0}>
                      {item.formula}
                    </span>
                  </Tooltip>
                  <Tag className="ag-grade" color={gradeTagColors[item.grade || "UNKNOWN"]}>
                    {gradeName(item.grade)}
                  </Tag>
                  <span className="ag-result-sharpe">
                    S <strong>{metric(item.sharpe)}</strong>
                  </span>
                </div>
              ))}
            </div>
          </article>
        </section>
        <section
          className="ag-bottom-grid"
          aria-label={locale === "zh-CN" ? "功能入口" : "Quick links"}
        >
          {m.modules.map((item, i) => (
            <Link
              to={item.path}
              key={item.path}
              className={`ag-card ag-module-card is-${item.tone}`}
            >
              <div className="ag-module-top">
                <span className="ag-module-icon">{moduleIcons[i]}</span>
                <div className="ag-module-title">{item.title}</div>
                <RightOutlined className="ag-module-arrow" />
              </div>
              <div className="ag-module-desc">{item.desc}</div>
            </Link>
          ))}
        </section>
      </div>
    </section>
  );
};
export default HomePage;
