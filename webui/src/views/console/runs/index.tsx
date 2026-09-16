import { Alert, Button, Card, Descriptions, Drawer, Form, InputNumber, Modal, Radio, Space, Table, Tag, Tooltip, Typography } from "antd";
import { useState } from "react";
import { useOutletContext } from "react-router-dom";
import type { ConsoleLocale } from "@/layouts/console/locale";
import { useApi } from "@/api/use-api";
import { useConsole } from "@/api/use-console";
import { errorMessage, type JobRequest, type Page, type Run, type Settings } from "@/api/console";
import { runStatusTagColors, statusName, timestamp } from "@/api/presentation";
import ModulePage from "../components/module-page";
import { JobPanel } from "../components/job-panel";
import RunDetails from "./run-details";
import FormulaDetails from "../formulas/formula-details";

type RunPanel = { view: "details"; run: Run; taskId?: string } | { view: "logs"; jobId: string };

export default function RunsPage() {
  const { locale } = useOutletContext<{ locale: ConsoleLocale }>();
  const zh = locale === "zh-CN";
  const { canOperate, start, pending, jobs, events } = useConsole();
  const [modal, modalContext] = Modal.useModal();
  const [mode, setMode] = useState("normal");
  const [cycles, setCycles] = useState(1);
  const [panel, setPanel] = useState<RunPanel | null>(null);
  const formulaTaskId = panel?.view === "details" ? panel.taskId : undefined;
  const openFormula = (taskId?: string) => setPanel(current => current?.view === "details" ? { ...current, taskId } : current);
  const [historyPage, setHistoryPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const runs = useApi<Page<Run>>(`/api/runs?page=${historyPage}&page_size=${pageSize}`);
  const settings = useApi<Settings>("/api/settings");
  const latestJob = jobs.data?.items[0];
  const activeJob = events.connected && !jobs.error && jobs.data?.busy && latestJob?.request.kind !== "submit"
    && ["running", "stopping"].includes(latestJob?.status ?? "") ? latestJob : undefined;
  const logJob = activeJob || (latestJob?.request.kind !== "submit" && ["failed", "needs_attention"].includes(latestJob?.status ?? "") ? latestJob : undefined);
  const displayedRuns = runs.data?.items.map(run => activeJob?.run_id === run.run_id
    ? { ...run, status: activeJob.status } : run);
  const panelJob = panel?.view === "logs" ? jobs.data?.items.find(job => job.id === panel.jobId) : undefined;
  const panelOpen = panel?.view === "details" || (panel?.view === "logs" && (!panelJob || panel.jobId === logJob?.id));
  const startWithLogs = async (request: JobRequest) => {
    const jobId = await start(request);
    if (jobId) setPanel({ view: "logs", jobId });
  };
  const confirmStart = () => modal.confirm({
    title: zh ? "开始实际回测" : "Start real backtests",
    content: zh ? `将执行 ${cycles} 轮${mode === "optimization" ? "专项优化" : "普通回测"}，每轮最多 ${settings.data?.limits.backtest_count ?? "—"} 条，不自动提交公式。新运行会收尾旧的中断计划；继续旧计划请使用恢复。` : `Start ${cycles} cycles with up to ${settings.data?.limits.backtest_count ?? "—"} backtests per cycle. Automatic submission is disabled. Interrupted plans are settled before the new run; use Resume to continue an existing plan.`,
    okText: zh ? "开始运行" : "Start run", cancelText: zh ? "取消" : "Cancel",
    onOk: () => startWithLogs({ kind: "run", mode, cycles }),
  });
  const resume = (run: Run) => modal.confirm({
    title: zh ? "恢复已有运行" : "Resume run",
    content: <Space direction="vertical" size={12} style={{ width: "100%" }}>
      <Descriptions column={1} size="small" items={[
        { key: "mode", label: zh ? "研究模式" : "Research mode", children: <Tag color={run.optimization_only ? "volcano" : "blue"}>{run.optimization_only ? (zh ? "专项优化" : "Focused optimization") : (zh ? "普通回测" : "Normal research")}</Tag> },
        { key: "time", label: run.started_at ? (zh ? "开始时间" : "Started at") : (zh ? "创建时间" : "Created at"), children: timestamp(run.started_at || run.created_at) },
        { key: "progress", label: zh ? "已结束 / 已计划" : "Finished / Planned", children: `${run.finished ?? "—"} / ${run.planned ?? "—"}` },
        { key: "submission", label: zh ? "自动提交" : "Automatic submission", children: run.automatic_submissions_enabled ? (zh ? "开启" : "Enabled") : (zh ? "关闭" : "Disabled") },
      ]} />
      <Typography.Text type="secondary">{zh ? "继续执行原有计划，沿用原配置和停止条件。" : "Continue the existing plan with its original configuration and stop conditions."}</Typography.Text>
    </Space>,
    okText: zh ? "恢复运行" : "Resume", cancelText: zh ? "取消" : "Cancel", onOk: () => startWithLogs({ kind: "resume", run_id: run.run_id }),
  });
  return <ModulePage>
    {modalContext}
    <Card className="ag-list-controls ag-runs-controls">
      <Form layout="inline" className="ag-run-form">
        <Form.Item label={zh ? "研究模式" : "Research mode"}><Radio.Group value={mode} onChange={e => setMode(e.target.value)} options={[{ label: zh ? "普通回测" : "Normal research", value: "normal" }, { label: zh ? "专项优化" : "Focused optimization", value: "optimization" }]} optionType="button" buttonStyle="solid" /></Form.Item>
        <Form.Item label={zh ? "运行轮数" : "Cycles"}><InputNumber aria-label={zh ? "运行轮数" : "Cycles"} min={1} max={1000} precision={0} value={cycles} onChange={value => setCycles(value ?? 1)} /></Form.Item>
        <Form.Item><Button type="primary" disabled={!canOperate || !settings.data} loading={pending} onClick={confirmStart}>{zh ? "开始运行" : "Start run"}</Button></Form.Item>
        {logJob && !logJob.run_id && <Form.Item><Button type="link" className="ag-table-action" onClick={() => setPanel({ view: "logs", jobId: logJob.id })}>{zh ? "启动日志" : "Startup logs"}</Button></Form.Item>}
      </Form>
      <Typography.Text type="secondary">{mode === "normal" ? (zh ? "结合探索、普通变异和自相关治理。" : "Combines exploration, mutation and self-correlation work.") : (zh ? "仅优化检查全部通过、评级不足且仍有额度的活动父代。" : "Optimizes eligible active parents below the target grade with attempts remaining.")}</Typography.Text>
    </Card>
    <div className="ag-list-table ag-runs-table" role="region" aria-label={zh ? "运行记录" : "Run history"}>
      {runs.error ? <Alert type="error" message={errorMessage(runs.error, zh)} /> : <Table<Run> bordered rowKey="run_id" size="middle" loading={runs.loading} dataSource={displayedRuns || []} pagination={{ current: historyPage, pageSize, total: runs.data?.total || 0, onChange: (nextPage, nextSize) => { setHistoryPage(nextSize === pageSize ? nextPage : 1); setPageSize(nextSize); }, showSizeChanger: true, pageSizeOptions: [10, 20, 50, 100] }} scroll={{ x: 1200 }} columns={[
        { title: zh ? "运行编号" : "Run ID", dataIndex: "run_id", width: 250, ellipsis: { showTitle: false }, render: id => <Tooltip title={id}><span>{id}</span></Tooltip> },
        { title: zh ? "模式" : "Mode", dataIndex: "optimization_only", width: 140, render: value => <Tag color={value ? "volcano" : "blue"}>{value ? (zh ? "专项优化" : "Optimization") : (zh ? "普通回测" : "Normal")}</Tag> },
        { title: zh ? "状态" : "Status", dataIndex: "status", width: 140, render: (value, row) => <Tooltip title={row.resume_blocked_reason ? errorMessage(row.resume_blocked_reason, zh) : undefined}><Tag color={runStatusTagColors[value]}>{statusName(value, zh)}</Tag></Tooltip> },
        { title: zh ? "轮次" : "Cycle", width: 110, render: (_, row) => `${row.current_cycle}/${row.max_cycles === -1 ? "∞" : row.max_cycles}` },
        { title: zh ? "已结束 / 已计划" : "Finished / Planned", width: 180, render: (_, row) => `${row.finished}/${row.planned}` },
        { title: zh ? "开始时间" : "Started at", dataIndex: "started_at", width: 180, render: timestamp },
        { title: zh ? "操作" : "Action", fixed: "right", width: 188, render: (_, row) => <Space size={16}>
          <Button type="link" className="ag-table-action" onClick={() => setPanel({ view: "details", run: row })}>{zh ? "详情" : "Details"}</Button>
          <Button type="link" className="ag-table-action" disabled={logJob?.run_id !== row.run_id} onClick={() => { if (logJob?.run_id === row.run_id) setPanel({ view: "logs", jobId: logJob.id }); }}>{zh ? "运行日志" : "Logs"}</Button>
          <Button type="link" className="ag-table-action" disabled={!row.can_resume || !canOperate} onClick={() => resume(row)}>{zh ? "恢复" : "Resume"}</Button>
        </Space> },
      ]} />}
    </div>
    <Drawer title={panel?.view === "details" ? (zh ? "运行详情" : "Run details") : (zh ? "运行日志" : "Run logs")} open={panelOpen} maskClosable={!formulaTaskId} keyboard={!formulaTaskId} onClose={() => { if (!formulaTaskId) setPanel(null); }} width="min(1180px, 100vw)" destroyOnHidden>
      {panel?.view === "details" && <RunDetails key={panel.run.run_id} run={displayedRuns?.find(run => run.run_id === panel.run.run_id) || panel.run} zh={zh} onOpenFormula={openFormula} />}
      {panel?.view === "logs" && panelOpen && <JobPanel zh={zh} scope={{ jobId: panel.jobId }} />}
      {formulaTaskId && <FormulaDetails key={formulaTaskId} taskId={formulaTaskId} zh={zh} onClose={() => openFormula()} />}
    </Drawer>
  </ModulePage>;
}
