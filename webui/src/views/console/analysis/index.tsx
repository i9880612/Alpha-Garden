import { BarChartOutlined, CloseOutlined, InfoCircleOutlined, SafetyCertificateOutlined, TableOutlined } from "@ant-design/icons";
import { Alert, Button, Empty, Modal, Segmented, Select, Skeleton, Table, Tag, Tooltip, Typography } from "antd";
import type { ColumnsType } from "antd/es/table";
import { useRef, useState } from "react";
import { useOutletContext, useSearchParams } from "react-router-dom";
import { useApi } from "@/api/use-api";
import { errorMessage } from "@/api/console";
import type { DiagnosticCorrelation, DiagnosticRecord, QualityDiagnosis } from "@/api/analysis";
import { checkFailureName, gradeName, gradeTagColors, metric, sourceName, timestamp } from "@/api/presentation";
import type { ConsoleLocale } from "@/layouts/console/locale";
import type { ConsoleTheme } from "@/layouts/console/theme";
import ModulePage from "../components/module-page";
import { formulaRowProps } from "../components/formula-row";
import FormulaDetails from "../formulas/formula-details";
import { Scatter, Turnover } from "./charts";
import { groupStyles, groupName, quantile, stateColors, stateName, turnoverBins } from "./presentation";
import "./index.css";

type Selection = { kind: "all" | "metric" | "check" | "platform" | "local" | "turnover"; value: string; problemOnly?: boolean };

function selectionTitle(selection: Selection, zh: boolean) {
  switch (selection.kind) {
    case "all": return zh ? "全部已完成公式" : "All completed formulas";
    case "metric": return groupName(selection.value, zh);
    case "check": return checkFailureName(selection.value, zh);
    case "platform": return `${zh ? "平台自相关" : "Platform correlation"} · ${stateName(selection.value, zh)}`;
    case "local": return `${zh ? "本地相关性" : "Local correlation"} · ${stateName(selection.value, zh)}`;
    case "turnover": return `${zh ? "换手率" : "Turnover"} ${turnoverBins.find(b => b.key === selection.value)!.label}${selection.problemOnly ? ` · ${groupName("fitness", zh)}` : ""}`;
  }
}

function States({ counts, zh, onSelect }: { counts: Record<string, number>; zh: boolean; onSelect: (state: string) => void }) {
  const entries = Object.keys(stateColors).filter(key => counts[key] > 0), total = Object.values(counts).reduce((a, b) => a + b, 0);
  if (!total) return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={zh ? "暂无已完成结果" : "No completed results"} />;
  return <>
    <div className="ag-diagnosis-state-bar" aria-hidden="true">{entries.map(key => <span key={key} style={{ flex: counts[key], background: stateColors[key] }} />)}</div>
    <div className="ag-diagnosis-state-legend">{entries.map(key => <button key={key} onClick={() => onSelect(key)}><i style={{ background: stateColors[key] }} />{stateName(key, zh)} <strong>{counts[key]}</strong></button>)}</div>
  </>;
}

function settingLabel(settings: Record<string, unknown>) {
  return [settings.region, settings.universe, settings.delay != null && `D${settings.delay}`, settings.decay != null && `Decay ${settings.decay}`, settings.neutralization, settings.truncation != null && `Trunc ${settings.truncation}`].filter(Boolean).join(" · ") || JSON.stringify(settings);
}

export default function AnalysisPage() {
  const { locale, theme } = useOutletContext<{ locale: ConsoleLocale; theme: ConsoleTheme }>();
  const zh = locale === "zh-CN";
  const [params, setParams] = useSearchParams();
  const [help, setHelp] = useState(false), [problemOnly, setProblemOnly] = useState(false);
  const [hiddenGroups, setHiddenGroups] = useState<Record<string, boolean>>({});
  const [selection, setSelection] = useState<Selection>(), [taskId, setTaskId] = useState<string>();
  const [page, setPage] = useState(1), [pageSize, setPageSize] = useState(10);
  const detailRef = useRef<HTMLDivElement>(null);
  const filters = new URLSearchParams({ days: params.get("days") || "30", mode: params.get("mode") || "all", source: params.get("source") || "", run_id: params.get("run_id") || "", configuration: params.get("configuration") || "" });
  const quality = useApi<QualityDiagnosis>(`/api/analysis/quality?${filters}`), data = quality.data;
  const localFilters = new URLSearchParams(filters);
  if (data) localFilters.set("configuration", data.configuration);
  const local = useApi<DiagnosticCorrelation>(data?.completed ? `/api/analysis/correlation?${localFilters}` : null);
  const correlation = local.data?.configuration === data?.configuration ? local.data : null;
  const evidence = new Map(correlation?.records.map(r => [r.task_id, r]));
  const currentSetting = data?.configurations.find(c => c.id === data.configuration);
  const change = (key: string, value: string) => {
    const next = new URLSearchParams(params); next.set(key, value);
    if (key !== "configuration") next.delete("configuration");
    setParams(next, { replace: true }); setSelection(undefined); setPage(1);
  };
  const choose = (next: Selection) => {
    setSelection(next); setPage(1);
    requestAnimationFrame(() => detailRef.current?.scrollIntoView({ block: "start", behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth" }));
  };
  const rows = data?.records || [], turnoverRows = problemOnly ? rows.filter(r => r.metric_group === "fitness") : rows;
  const turnovers = turnoverRows.flatMap(r => r.turnover == null ? [] : [r.turnover]);
  const problemCount = rows.filter(r => r.metric_group === "fitness").length;
  const selectedRows = rows.filter(r => {
    if (!selection || selection.kind === "all") return true;
    if (selection.kind === "metric") return r.metric_group === selection.value;
    if (selection.kind === "check") return r.failed_checks.includes(selection.value);
    if (selection.kind === "platform") return r.platform_state === selection.value;
    if (selection.kind === "local") return evidence.get(r.task_id)?.state === selection.value;
    const bin = turnoverBins.find(b => b.key === selection.value)!;
    return (!selection.problemOnly || r.metric_group === "fitness") && r.turnover != null && r.turnover >= bin.min && r.turnover < bin.max;
  });
  const selectTurnover = (value: string) => choose({ kind: "turnover", value, problemOnly });
  const columns: ColumnsType<DiagnosticRecord> = [
    { title: "Alpha ID", dataIndex: "alpha_id", ellipsis: true, render: value => value || "—" },
    { title: zh ? "评级" : "Grade", dataIndex: "grade", render: value => <Tag color={gradeTagColors[value || ""]}>{gradeName(value)}</Tag> },
    { title: zh ? "来源" : "Source", dataIndex: "source", render: value => sourceName(value, zh), responsive: ["lg"] },
    { title: "Sharpe", dataIndex: "sharpe", render: value => metric(value) },
    { title: "Fitness", dataIndex: "fitness", render: value => metric(value) },
    { title: "Turnover", dataIndex: "turnover", render: value => metric(value, true) },
    ...(selection?.kind === "local" ? [
      { title: zh ? "最高相关值" : "Max correlation", key: "correlation", render: (_: unknown, r: DiagnosticRecord) => evidence.get(r.task_id)?.maximum?.toFixed(4) ?? "—" },
      { title: zh ? "比较覆盖" : "Coverage", key: "coverage", render: (_: unknown, r: DiagnosticRecord) => { const e = evidence.get(r.task_id); return e ? <Tooltip title={`${zh ? "最高相关参考" : "Highest-correlation reference"}: ${e.reference_id || "—"}; ${zh ? "高相关且提升达标" : "High-correlation pairs meeting improvement"}: ${e.improved_pairs}/${e.high_pairs}`}>{e.compared} / {e.required}</Tooltip> : "—"; } },
    ] : [{ title: zh ? "结束时间" : "Finished", dataIndex: "finished_at", render: (value: string) => timestamp(value), responsive: ["xl" as const] }]),
  ];
  return <ModulePage><div className={`ag-diagnosis ag-diagnosis--${theme}`}>
    <header className="ag-diagnosis-header"><div><Typography.Title level={3}>{zh ? "质量诊断" : "Quality diagnosis"}</Typography.Title><Typography.Text type="secondary">{zh ? "定位质量瓶颈，查看对应公式与检查证据。" : "Find quality bottlenecks and inspect the underlying formulas."}</Typography.Text></div><Button icon={<InfoCircleOutlined />} onClick={() => setHelp(true)}>{zh ? "口径说明" : "How it works"}</Button></header>
    <div className="ag-diagnosis-filters">
      <label>{zh ? "时间范围" : "Time range"}<Select aria-label={zh ? "时间范围" : "Time range"} value={filters.get("days")} onChange={v => change("days", v)} options={["7", "15", "30", "all"].map(v => ({ value: v, label: v === "all" ? (zh ? "全部时间" : "All time") : (zh ? `最近 ${v} 天` : `Last ${v} days`) }))} /></label>
      <label>{zh ? "研究模式" : "Mode"}<Select aria-label={zh ? "研究模式" : "Research mode"} value={filters.get("mode")} onChange={v => change("mode", v)} options={[{ value: "all", label: zh ? "全部模式" : "All modes" }, { value: "normal", label: zh ? "普通研究" : "Normal research" }, { value: "optimization", label: zh ? "专项优化" : "Optimization" }, { value: "unassigned", label: zh ? "未关联运行" : "Unassigned" }]} /></label>
      <label>{zh ? "运行批次" : "Run"}<Select aria-label={zh ? "运行批次" : "Run"} className="ag-diagnosis-run-select" showSearch optionFilterProp="label" value={filters.get("run_id")} onChange={v => change("run_id", v)} options={[{ value: "", label: zh ? "全部批次" : "All runs" }, ...(data?.runs || []).map(id => ({ value: id, label: id, title: id }))]} /></label>
      <label>{zh ? "来源" : "Source"}<Select aria-label={zh ? "来源" : "Source"} value={filters.get("source")} onChange={v => change("source", v)} options={[{ value: "", label: zh ? "全部来源" : "All sources" }, ...["exploration", "mutation", "sc", "reversal"].map(v => ({ value: v, label: sourceName(v, zh) }))]} /></label>
      <Button className="ag-diagnosis-reset" onClick={() => { setParams({}, { replace: true }); setSelection(undefined); setPage(1); setProblemOnly(false); setHiddenGroups({}); }}>{zh ? "重置筛选" : "Reset"}</Button>
    </div>
    {quality.error && <Alert showIcon type="error" message={errorMessage(quality.error, zh)} />}
    {!data ? quality.loading && <Skeleton active paragraph={{ rows: 8 }} /> : <>
      <div className="ag-diagnosis-scope">
        <span>{zh ? "当前范围" : "Scope"} <strong>{data.total}</strong> · {zh ? "已完成" : "Completed"} <button onClick={() => choose({ kind: "all", value: "" })}>{data.completed}</button> · {zh ? "处理中" : "In progress"} {data.inflight} · {zh ? "回测异常" : "Backtest errors"} {data.errors}</span>
        <label>{zh ? "回测配置" : "Configuration"}<Tooltip title={currentSetting && Object.entries(currentSetting.settings).map(([k, v]) => `${k}: ${v}`).join("; ")}><Select aria-label={zh ? "回测配置" : "Configuration"} value={data.configuration || undefined} placeholder="—" onChange={v => change("configuration", v)} options={data.configurations.map(c => ({ value: c.id, label: `${settingLabel(c.settings)} (${c.count})` }))} /></Tooltip></label>
      </div>
      {data.total === 0 ? <div className="ag-diagnosis-empty"><Empty description={zh ? "当前条件下没有已进入回测的公式" : "No backtests match these filters"} /><Button onClick={() => { setParams({}, { replace: true }); setSelection(undefined); }}>{zh ? "重置筛选" : "Reset filters"}</Button></div> : <>
        <div className="ag-diagnosis-grid">
          <section className="ag-diagnosis-panel"><div className="ag-diagnosis-panel-title"><h3><BarChartOutlined />{zh ? "夏普比率 × Fitness" : "Sharpe × Fitness"}</h3><span>{zh ? "有效样本" : "Samples"} {data.analyzable}</span></div>
            <div className="ag-diagnosis-legend">{Object.entries(groupStyles).map(([key, style]) => <button key={key} aria-pressed={!hiddenGroups[key]} title={zh ? (hiddenGroups[key] ? "点击显示该类型" : "点击隐藏该类型") : (hiddenGroups[key] ? "Show this group" : "Hide this group")} onClick={() => setHiddenGroups(current => ({ ...current, [key]: !current[key] }))}><i aria-hidden="true" className="ag-diagnosis-symbol" style={{ color: hiddenGroups[key] ? undefined : style[theme] }} />{groupName(key, zh)}</button>)}</div>
            {data.analyzable ? <Scatter data={data} theme={theme} zh={zh} hiddenGroups={hiddenGroups} onOpen={setTaskId} /> : <div className="ag-diagnosis-chart ag-diagnosis-empty"><Empty description={zh ? "尚无完整的夏普与 Fitness 指标" : "No complete Sharpe and Fitness pairs"} /></div>}
            {problemCount > 0 && <button className="ag-diagnosis-insight" onClick={() => choose({ kind: "metric", value: "fitness" })}><InfoCircleOutlined /><span>{zh ? `夏普已过线、Fitness 未过线：${problemCount} 条，值得单独检查换手与收益。` : `${problemCount} formulas pass Sharpe but fail Fitness. Inspect their turnover and returns.`}</span><span>{zh ? "查看这组 →" : "Inspect →"}</span></button>}
            <p className="ag-diagnosis-footnote">{data.thresholds ? (zh ? `平台参考线：Sharpe ${data.thresholds[0]}，Fitness ${data.thresholds[1]}。两项过线不等于可提交。` : `Saved platform limits: Sharpe ${data.thresholds[0]}, Fitness ${data.thresholds[1]}. Passing both does not imply submission eligibility.`) : (zh ? "平台阈值缺失或不一致，不显示统一参考线。" : "No shared reference lines: platform thresholds are missing or differ.")}</p>
          </section>
          <section className="ag-diagnosis-panel"><div className="ag-diagnosis-panel-title"><h3><BarChartOutlined />{zh ? "换手率分布" : "Turnover distribution"}</h3><span>{zh ? "中位数" : "Median"} <strong>{metric(quantile(turnovers, .5), true)}</strong></span></div>
            <Segmented size="small" value={problemOnly ? "problem" : "all"} onChange={v => setProblemOnly(v === "problem")} options={[{ value: "all", label: zh ? "全部结果" : "All results" }, { value: "problem", label: zh ? "夏普过线、Fitness 未过线" : "Sharpe passed, Fitness failed" }]} />
            <div className="ag-diagnosis-turnover-meta"><span>{turnovers.length} {zh ? "条样本" : "samples"} · P75 {metric(quantile(turnovers, .75), true)}</span><Select size="small" aria-label={zh ? "查看换手区间" : "Inspect turnover range"} placeholder={zh ? "查看区间" : "Inspect range"} value={null} onChange={selectTurnover} options={turnoverBins.map(b => ({ value: b.key, label: b.label }))} /></div>
            {turnovers.length ? <Turnover rows={turnoverRows} theme={theme} zh={zh} onSelect={selectTurnover} /> : <div className="ag-diagnosis-chart ag-diagnosis-empty"><Empty description={zh ? "暂无换手率样本" : "No turnover samples"} /></div>}
            <p className="ag-diagnosis-footnote">{zh ? "分位数描述样本分布，不是平台门槛；点击柱体查看公式。" : "Percentiles describe this sample, not platform limits. Select a bar to inspect formulas."}</p>
          </section>
          <section className="ag-diagnosis-panel"><div className="ag-diagnosis-panel-title"><h3><TableOutlined />{zh ? "检测未通过原因" : "Failed checks"}</h3><Tooltip title={zh ? "比例 = 未通过 ÷（通过 + 未通过）；待定和缺失不计入分母。" : "Rate = failed / (passed + failed). Pending and missing checks are excluded."}><InfoCircleOutlined /></Tooltip></div>
            <div className="ag-diagnosis-reason-heading"><span>{zh ? "检查项" : "Check"}</span><span>{zh ? "未通过 / 明确" : "Failed / decided"}</span><span>{zh ? "比例" : "Rate"}</span></div>
            {data.reasons.map(reason => { const denominator = reason.passed + reason.failed, ratio = denominator ? reason.failed / denominator : null; return <Tooltip key={reason.name} title={`${zh ? "待定或缺失" : "Unresolved"}: ${reason.unresolved}`}><button className="ag-diagnosis-reason" disabled={!reason.failed} onClick={() => choose({ kind: "check", value: reason.name })}><span>{checkFailureName(reason.name, zh)}</span><span className="ag-diagnosis-reason-track"><span style={{ width: `${(ratio || 0) * 100}%` }} /></span><span>{reason.failed} / {denominator}</span><span>{ratio == null ? "—" : `${(ratio * 100).toFixed(1)}%`}</span></button></Tooltip>; })}
            <p className="ag-diagnosis-footnote">{zh ? "一条公式可能多项未通过，各项数量不能相加为公式总数。" : "A formula may fail multiple checks; these counts are not additive."}</p>
          </section>
          <div className="ag-diagnosis-correlation-stack">
            <section className="ag-diagnosis-panel"><div className="ag-diagnosis-panel-title"><h3><SafetyCertificateOutlined />{zh ? "平台自相关检查" : "Platform self-correlation"}</h3><span>{data.completed} {zh ? "条" : "results"}</span></div><p className="ag-diagnosis-description">{zh ? "采用最新保存的平台结论；待定与未取得单列。" : "Latest saved platform result; pending and unavailable remain separate."}</p><States counts={data.platform} zh={zh} onSelect={value => choose({ kind: "platform", value })} /></section>
            <section className="ag-diagnosis-panel"><div className="ag-diagnosis-panel-title"><h3><SafetyCertificateOutlined />{zh ? "本地收益相关性参考" : "Local return correlation"}</h3><Tooltip title={zh ? "基于本地已保存的日收益，与同账号已提交公式比较。" : "Saved daily returns compared with this account's submitted formulas."}><InfoCircleOutlined /></Tooltip></div>
              {local.error ? <Alert type="warning" showIcon message={errorMessage(local.error, zh)} /> : data.completed === 0 ? <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={zh ? "暂无已完成结果" : "No completed results"} /> : !correlation ? <Skeleton active title={false} paragraph={{ rows: 2 }} /> : <><p className="ag-diagnosis-description">{zh ? `${correlation.reference_count} 条已提交参考 · 缺少曲线或重叠区间不足时保持待定。` : `${correlation.reference_count} submitted references · missing curves or insufficient overlap remain pending.`}</p><States counts={correlation.counts} zh={zh} onSelect={value => choose({ kind: "local", value })} /><p className="ag-diagnosis-footnote">{zh ? `含 ${correlation.improvement_passed} 条高相关但 Sharpe 提升至少 10% 的通过结果。仅供参考，不替代提交检查。` : `Includes ${correlation.improvement_passed} high-correlation passes with ≥10% Sharpe improvement. Does not replace submission checks.`}</p></>}
            </section>
          </div>
        </div>
        <div ref={detailRef} className="ag-diagnosis-detail">
          {selection ? <><div className="ag-diagnosis-panel-title"><h3>{selectionTitle(selection, zh)}<span className="ag-diagnosis-count">{selectedRows.length}</span></h3><Button type="text" icon={<CloseOutlined />} aria-label={zh ? "收起明细" : "Close records"} onClick={() => setSelection(undefined)} /></div><Table<DiagnosticRecord> size="middle" bordered rowKey="task_id" tableLayout="fixed" dataSource={selectedRows} columns={columns} onRow={r => formulaRowProps(r, zh, setTaskId)} pagination={{ current: Math.min(page, Math.max(1, Math.ceil(selectedRows.length / pageSize))), pageSize, showSizeChanger: true, pageSizeOptions: [10, 20, 50, 100], onChange: (p, size) => { setPage(size !== pageSize ? 1 : p); setPageSize(size); }, showTotal: total => zh ? `共 ${total} 条` : `${total} results` }} /></> : <button className="ag-diagnosis-details-hint" onClick={() => choose({ kind: "all", value: "" })}><TableOutlined />{zh ? "点击柱体或检测项展开公式明细，也可查看全部已完成公式；散点图例可切换类型显示。" : "Select a bar or check to inspect formulas, or view all completed results. Scatter legends toggle group visibility."}</button>}
        </div>
      </>}
    </>}
    <Modal open={help} title={zh ? "统计口径" : "About this analysis"} footer={null} onCancel={() => setHelp(false)} width={650}>
      <div className="ag-diagnosis-help">{(zh ? [
        ["研究范围", "按当前账号统计已进入回测的任务，每个任务一条。已结束任务按结束时间、处理中任务按提交时间筛选；时间不是策略的回测年份。未发送候选不计入。"],
        ["同一回测配置", "每次展示一组完全相同的回测设置，默认选样本最多的一组，可切换查看其他组。不会混合不同股票池、延迟或中性化设置。"],
        ["平台检测", "优先采用最新保存的完整检查，没有完整检查时采用回测结束快照。较新的待定、错误或缺失明细不会回退成旧的通过。指标分组依据检查结论；阈值一致时显示参考线。"],
        ["本地相关性与 10% 条件", "复用研究引擎的日收益比较规则，要求至少 252 个匹配区间。相关系数低于 0.7，或高相关且 Sharpe 至少达到参考公式的 1.10 倍，才满足该比较对象。所有参考均满足才显示通过；已有明确失败则显示未通过。缺失、平坦或重叠不足的曲线保持待定，无比较对象单列。"],
        ["数据读取", "分析统计只使用本地已保存数据。公式详情与公式库一致：打开时获取平台 PnL，切换标签时按需获取 Sharpe 或 Turnover；服务以只读模式启动时不获取平台曲线。这些操作不触发回测或提交，本地通过也不绕过既有提交检查与研究机会保护。"],
      ] : [
        ["Scope", "One row per task for the current account. Finished tasks use completion time; in-progress tasks use submission time. Unsent candidates are excluded. The range is not the strategy's backtest years."],
        ["Configuration", "One exact set of backtest settings at a time, initially the largest group. Other settings can be selected without mixing universes, delays or neutralization."],
        ["Platform checks", "Use the latest saved full check, otherwise the backtest snapshot. A newer pending, error or incomplete observation never revives an older pass. Grouping follows check results; shared limits provide reference lines."],
        ["Local correlation and 10% improvement", "Reuses the engine's daily-return comparison, requiring at least 252 matched intervals. Each reference must have correlation below 0.7 or candidate Sharpe at least 1.10 times the reference Sharpe. All references must satisfy the rule for a pass; a known failure rejects. Missing, flat or insufficient curves remain pending. No references is a separate state."],
        ["Data access", "Analysis uses saved local data. Formula details behave like the formula library: opening a drawer fetches platform PnL, and selecting Sharpe or Turnover fetches that series on demand. Read-only services disable platform curves. These actions do not trigger backtests or submissions; local passes do not bypass formal checks or research-opportunity protection."],
      ]).map(([title, description]) => <section key={title}><h4>{title}</h4><p>{description}</p></section>)}</div>
    </Modal>
    {taskId && <FormulaDetails key={taskId} taskId={taskId} zh={zh} onClose={() => setTaskId(undefined)} />}
  </div></ModulePage>;
}
