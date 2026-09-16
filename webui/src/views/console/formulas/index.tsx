import { QuestionCircleOutlined } from "@ant-design/icons";
import { Button, Drawer, Input, Modal, Select, Space, Tag, Tooltip, Typography, theme } from "antd";
import { useState } from "react";
import { useOutletContext, useSearchParams } from "react-router-dom";
import type { ConsoleLocale } from "@/layouts/console/locale";
import type { Formula, Page } from "@/api/console";
import { useApi } from "@/api/use-api";
import { useConsole } from "@/api/use-console";
import { grades, gradeName, sourceName } from "@/api/presentation";
import ModulePage from "../components/module-page";
import { FormulaTable } from "../components/formula-table";
import { JobPanel } from "../components/job-panel";

import FormulaDetails from "./formula-details";

export default function FormulasPage({ category = "all" }: { category?: string }) {
  const { locale } = useOutletContext<{ locale: ConsoleLocale }>();
  const zh = locale === "zh-CN";
  const { token } = theme.useToken();
  const [params, setParams] = useSearchParams();
  const taskId = params.get("task"), runId = params.get("run_id");
  const [search, setSearch] = useState("");
  const [grade, setGrade] = useState<string>();
  const [source, setSource] = useState<string>();
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const query = new URLSearchParams({ category, run_id: runId || "", search, grade: grade || "", page: String(page), page_size: String(pageSize) });
  if (category === "all") query.set("execution", "started");
  if (category === "optimization" && source) query.set("source", source);
  const data = useApi<Page<Formula>>(`/api/formulas?${query}`);
  const { canOperate, pending, start } = useConsole();
  const [modal, modalContext] = Modal.useModal();
  const [submissionJob, setSubmissionJob] = useState<string>();
  const submit = (row: Formula) => modal.confirm({
    title: zh ? "提交公式到 WorldQuant BRAIN" : "Submit Alpha to WorldQuant BRAIN",
    content: <Space direction="vertical">
      <Typography.Text strong>{row.alpha_id} · {gradeName(row.grade)}</Typography.Text>
      <Typography.Text>{zh ? "仅提交这条公式。提交前会重新核对相关性、10% 改善机会和平台检查，条件变化时暂缓提交。" : "Submit only this formula. Correlation, 10% improvement opportunities and platform checks are rechecked before submission; changed eligibility defers the action."}</Typography.Text>
      <Typography.Text type="secondary">{zh ? "若有中断的回测，会先收尾该计划：取消未发送任务，收取已发送回测结果。" : "Interrupted research is settled first: unsent tasks are cancelled and results of sent backtests are collected."}</Typography.Text>
    </Space>,
    okText: zh ? "确认提交" : "Confirm submission", cancelText: zh ? "取消" : "Cancel",
    okButtonProps: { disabled: !canOperate },
    onOk: async () => {
      const jobId = await start({ kind: "submit", task_id: row.task_id, ...(category === "archive" ? { source: "qualified_archive" as const } : {}) });
      if (jobId) setSubmissionJob(jobId);
    },
  });
  return <ModulePage>
    {modalContext}
    <div><Space className="ag-filter-row" wrap>
      <Input.Search placeholder={zh ? "搜索 Alpha ID" : "Search Alpha ID"} aria-label={zh ? "搜索 Alpha ID" : "Search Alpha ID"} maxLength={128} allowClear onSearch={value => { setSearch(value); setPage(1); }} style={{ width: 240 }} />
      <Select allowClear aria-label={zh ? "筛选评级" : "Filter by grade"} placeholder={zh ? "全部评级" : "All grades"} value={grade} options={[...grades, "UNKNOWN"].map(value => ({ value, label: gradeName(value) }))} onChange={value => { setGrade(value); setPage(1); }} style={{ width: 180 }} />
      {category === "optimization" && <Select allowClear aria-label={zh ? "筛选来源" : "Filter by source"} placeholder={zh ? "全部来源" : "All sources"} value={source} options={["exploration", "sc", "mutation", "reversal"].map(value => ({ value, label: sourceName(value, zh) }))} onChange={value => { setSource(value); setPage(1); }} style={{ width: 180 }} />}
    </Space>
    {runId && <Tag closable onClose={() => { const next = new URLSearchParams(params); next.delete("run_id"); setParams(next); setPage(1); }}>{zh ? "指定运行" : "Selected run"} · {runId.slice(0, 18)}</Tag>}
    {category === "all" && <Typography.Paragraph type="secondary" className="ag-formula-note">
      <QuestionCircleOutlined aria-hidden style={{ color: token.colorPrimary, fontSize: 16 }} />
      <span>{zh ? "展示已发起回测的公式及其结果；尚未发起回测的候选可在运行详情的“未开始”中查看。点击整行可打开详情。" : "Browse formulas whose backtests have started and their results. Unsent candidates are listed under Not started in run details. Click a row to view details."}</span>
    </Typography.Paragraph>}
    {category !== "all" && <Typography.Paragraph type="secondary" className="ag-formula-note"><QuestionCircleOutlined aria-hidden style={{ color: token.colorPrimary, fontSize: 16 }} /><span>{category === "optimization" ? (zh ? "均已通过平台检查，每条公式最多有 20 次变异机会，用于继续探索提高等级的可能性。提交时会保护其他保留公式的 10% 改善机会，资格不足或证据待定时暂缓提交。" : "Checked formulas have up to 20 mutation attempts to explore grade improvements. Submission preserves other retained formulas’ 10% improvement opportunities and waits when eligibility or evidence is unresolved.") : category === "archive" ? (zh ? "展示已结束优化、通过检查且当前可提交的公式。根据本地相关性与 Sharpe 提升 10% 的条件安排提交顺序，保护继续研究的公式；每次提交后重新评估。" : "Checked formulas whose optimization has ended and which can currently be submitted. Local correlation and 10% Sharpe improvements determine the submission order while protecting ongoing research. Eligibility is recalculated after each submission.") : (zh ? "已同步的平台提交结果。" : "Synced platform submission results.")}</span></Typography.Paragraph>}
    <div className="ag-list-table"><FormulaTable {...data} zh={zh} page={page} pageSize={pageSize} actions={category === "optimization" || category === "archive" ? row => <Tooltip title={row.can_submit ? undefined : (zh ? "需保留其他公式的改善机会，或相关性证据尚不完整。" : "Other improvement opportunities are protected, or correlation evidence is incomplete.")}><span><Button type="link" className="ag-table-action" disabled={!canOperate || pending || !row.can_submit || row.full_check !== "passed" || !row.alpha_id} onClick={() => submit(row)}>{zh ? "提交" : "Submit"}</Button></span></Tooltip> : undefined} onOpen={task => { const next = new URLSearchParams(params); next.set("task", task); setParams(next); }} onPage={(nextPage, nextSize) => { setPage(nextSize === pageSize ? nextPage : 1); setPageSize(nextSize); }} /></div>
    </div>
    {taskId && <FormulaDetails key={taskId} taskId={taskId} zh={zh} onClose={() => { const next = new URLSearchParams(params); next.delete("task"); setParams(next); }} />}
    <Drawer title={zh ? "提交操作与日志" : "Operations and logs"} open={Boolean(submissionJob)} onClose={() => setSubmissionJob(undefined)} width="min(960px, 100vw)" destroyOnHidden>{submissionJob && <JobPanel zh={zh} scope={{ jobId: submissionJob }} />}</Drawer>
  </ModulePage>;
}
