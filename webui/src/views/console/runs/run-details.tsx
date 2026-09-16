import { ConfigProvider, Descriptions, Segmented, Tag, Typography, theme } from "antd";
import { useState } from "react";
import { useApi } from "@/api/use-api";
import type { Formula, Page, Run } from "@/api/console";
import { runStatusTagColors, statusName, timestamp } from "@/api/presentation";
import { FormulaTable } from "../components/formula-table";

export default function RunDetails({ run, zh, onOpenFormula }: { run: Run; zh: boolean; onOpenFormula: (taskId: string) => void }) {
  const { token } = theme.useToken();
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [execution, setExecution] = useState("finished");
  const results = useApi<Page<Formula> & { unstarted: number; inflight: number }>(`/api/formulas?run_id=${encodeURIComponent(run.run_id)}&execution=${execution}&page=${page}&page_size=${pageSize}`);
  return <ConfigProvider theme={{
    token: { colorBgContainer: token.colorBgElevated, colorBorderSecondary: token.colorSplit },
    components: { Table: { headerBg: token.colorFillAlter, rowHoverBg: token.colorFillSecondary, borderColor: token.colorSplit } },
  }}><div className="ag-module-page">
    <Descriptions bordered size="middle" column={{ xs: 1, sm: 2, md: 2, lg: 3, xl: 3, xxl: 3 }} styles={{ label: { width: 130, whiteSpace: "nowrap" }, content: { overflowWrap: "anywhere" } }} items={[
      { key: "id", label: zh ? "运行编号" : "Run ID", span: "filled", children: <Typography.Text copyable>{run.run_id}</Typography.Text> },
      { key: "mode", label: zh ? "模式" : "Mode", children: <Tag color={run.optimization_only ? "volcano" : "blue"}>{run.optimization_only ? (zh ? "专项优化" : "Optimization") : (zh ? "普通回测" : "Normal")}</Tag> },
      { key: "status", label: zh ? "状态" : "Status", children: <Tag color={runStatusTagColors[run.status]}>{statusName(run.status, zh)}</Tag> },
      { key: "cycle", label: zh ? "轮次" : "Cycle", children: `${run.current_cycle}/${run.max_cycles === -1 ? "∞" : run.max_cycles}` },
      { key: "progress", label: zh ? "已结束 / 已计划" : "Finished / Planned", children: `${run.finished ?? "—"}/${run.planned ?? "—"}` },
      { key: "started", label: zh ? "开始时间" : "Started", children: timestamp(run.started_at) },
      { key: "finished", label: zh ? "结束时间" : "Finished", children: timestamp(run.finished_at) },
      { key: "reason", label: zh ? "停止原因" : "Stop reason", span: "filled", children: run.stop_reason === "user_paused" ? (zh ? "手动暂停，恢复后继续原计划" : "Paused by user; resume continues the existing plan") : run.stop_reason || "—" },
    ]} />
    <div><Segmented value={execution} onChange={value => { setExecution(value); setPage(1); }} options={[
      { label: zh ? "回测记录" : "Backtests", value: "finished" },
      { label: `${zh ? "回测中" : "In progress"} (${results.data?.inflight ?? "—"})`, value: "inflight" },
      { label: `${zh ? "未开始" : "Not started"} (${results.data?.unstarted ?? "—"})`, value: "planned" },
    ]} /></div>
    <FormulaTable {...results} zh={zh} page={page} pageSize={pageSize} onOpen={onOpenFormula} onPage={(nextPage, nextSize) => { setPage(nextSize === pageSize ? nextPage : 1); setPageSize(nextSize); }} />
  </div></ConfigProvider>;
}
