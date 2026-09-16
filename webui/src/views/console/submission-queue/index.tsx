import { Alert, Button, Card, Drawer, Form, InputNumber, Modal, Select, Table, Tabs, Typography } from "antd";
import { useState } from "react";
import { useOutletContext } from "react-router-dom";
import type { ConsoleLocale } from "@/layouts/console/locale";
import { errorMessage, type Formula, type Page, type Submission } from "@/api/console";
import { useApi } from "@/api/use-api";
import { useConsole } from "@/api/use-console";
import { grades, gradeName, statusName, timestamp } from "@/api/presentation";
import ModulePage from "../components/module-page";
import { FormulaTable } from "../components/formula-table";
import { JobPanel } from "../components/job-panel";
import { formulaRowProps } from "../components/formula-row";
import FormulaDetails from "../formulas/formula-details";

export default function SubmissionPage() {
  const { locale } = useOutletContext<{ locale: ConsoleLocale }>();
  const zh = locale === "zh-CN";
  const { canOperate, pending, start } = useConsole();
  const [modal, modalContext] = Modal.useModal();
  const [grade, setGrade] = useState("SPECTACULAR");
  const [count, setCount] = useState(2);
  const [logsOpen, setLogsOpen] = useState(false);
  const [taskId, setTaskId] = useState<string>();
  const continueSubmission = (row: Submission) => modal.confirm({
    title: zh ? "继续提交这条公式" : "Continue this submission",
    content: zh ? `继续处理 ${row.alpha_id} 的现有提交记录；如果请求结果待确认，将先核实平台状态。` : `Continue the existing submission for ${row.alpha_id}. If its outcome is uncertain, the platform status is checked first.`,
    okText: zh ? "继续提交" : "Continue", cancelText: zh ? "取消" : "Cancel",
    onOk: async () => { if (await start({ kind: "submit", task_id: row.task_id, ...(row.source === "qualified_archive" ? { source: "qualified_archive" as const } : {}) })) setLogsOpen(true); },
  });
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [historyPage, setHistoryPage] = useState(1);
  const [historyPageSize, setHistoryPageSize] = useState(10);
  const candidates = useApi<Page<Formula>>(`/api/formulas?category=candidates&grade=${grade}&page=${page}&page_size=${pageSize}`);
  const history = useApi<Page<Submission>>(`/api/submissions?page=${historyPage}&page_size=${historyPageSize}`);
  const submit = () => modal.confirm({
    title: zh ? "提交公式到 WorldQuant BRAIN" : "Submit Alphas to WorldQuant BRAIN",
    content: zh ? `从 ${gradeName(grade)} 候选中复检并提交，成功 ${count} 条后停止；先恢复未完成项，检查失败跳过，候选不足则提前结束。` : `Recheck and submit ${gradeName(grade)} candidates until ${count} submissions succeed. Unfinished submissions are resumed first; failed checks are skipped and an empty queue ends processing.`,
    okText: zh ? "确认提交" : "Confirm submission", cancelText: zh ? "取消" : "Cancel",
    onOk: () => start({ kind: "submit", grade, count }),
  });
  return <ModulePage>
    {modalContext}
    <Card className="ag-list-controls" title={zh ? "提交参数" : "Submission configuration"} extra={<Button type="link" className="ag-table-action" onClick={() => setLogsOpen(true)}>{zh ? "操作日志" : "Logs"}</Button>}>
      <Form layout="inline" size="large" className="ag-run-form">
        <Form.Item label={zh ? "公式评级" : "Alpha grade"}><Select value={grade} onChange={value => { setGrade(value); setPage(1); }} aria-label={zh ? "公式评级" : "Alpha grade"} style={{ width: 200 }} options={grades.map(value => ({ value, label: gradeName(value) }))} /></Form.Item>
        <Form.Item label={zh ? "成功提交数量" : "Successful submissions"}><InputNumber aria-label={zh ? "成功提交数量" : "Successful submissions"} value={count} onChange={value => setCount(value ?? 1)} min={1} max={1000} precision={0} /></Form.Item>
        <Form.Item><Button type="primary" disabled={!canOperate || !candidates.data} loading={pending} onClick={submit}>{zh ? "开始提交" : "Start submitting"}</Button></Form.Item>
      </Form>
      <Typography.Paragraph type="secondary">{zh ? "未完成的提交会先恢复；如有中断的回测，会先收尾。没有待处理项时直接结束。" : "Unfinished submissions are resumed first. Interrupted research is settled before submission; no pending work ends processing."}</Typography.Paragraph>
      <Typography.Text type="secondary">{grade === "SPECTACULAR" ? (zh ? "从 Spectacular 队列中选择当前可提交的候选。" : "Currently eligible candidates from the Spectacular queue.") : (zh ? "从合格归档中选择该评级当前可提交的公式，按相关性和 10% 改善条件保护提交顺序。" : "Currently eligible archived formulas at this grade, ordered to preserve correlation and 10% improvement opportunities.")}</Typography.Text>
    </Card>
    <Tabs items={[
      { key: "queue", label: zh ? "待提交候选" : "Candidates", children: <div className="ag-list-table"><FormulaTable {...candidates} zh={zh} page={page} pageSize={pageSize} onOpen={setTaskId} onPage={(nextPage, nextSize) => { setPage(nextSize === pageSize ? nextPage : 1); setPageSize(nextSize); }} /></div> },
      { key: "history", label: zh ? "提交记录" : "Submission history", children: <div className="ag-list-table">{history.error ? <Alert type="error" message={errorMessage(history.error, zh)} /> : <Table<Submission> bordered size="large" rowKey="task_id" onRow={row => formulaRowProps(row, zh, setTaskId)} loading={history.loading} dataSource={history.data?.items || []} scroll={{ x: 780 }} pagination={{ current: historyPage, pageSize: historyPageSize, total: history.data?.total || 0, onChange: (nextPage, nextSize) => { setHistoryPage(nextSize === historyPageSize ? nextPage : 1); setHistoryPageSize(nextSize); }, showSizeChanger: true, pageSizeOptions: [10, 20, 50, 100] }} columns={[
        { title: "Alpha ID", dataIndex: "alpha_id" }, { title: zh ? "评级" : "Grade", dataIndex: "grade", render: gradeName },
        { title: zh ? "状态" : "Status", dataIndex: "status", render: value => statusName(value, zh) },
        { title: zh ? "来源" : "Source", dataIndex: "source", render: value => value === "queue" ? (zh ? "提交队列" : "Queue") : value === "optimization" ? (zh ? "专项优化公式" : "Optimization formulas") : (zh ? "合格归档" : "Qualified archive") },
        { title: zh ? "更新时间" : "Updated", dataIndex: "updated_at", render: timestamp }, { title: zh ? "原因" : "Reason", dataIndex: "failure_code", ellipsis: true, render: value => value || "—" },
        { title: zh ? "操作" : "Action", width: 130, render: (_, row) => ["optimization", "qualified_archive"].includes(row.source) && ["detail_pending", "check_pending", "ready", "submitting", "confirmation_pending", "submission_unknown"].includes(row.status) ? <Button type="link" className="ag-table-action" disabled={!canOperate || pending} onClick={() => continueSubmission(row)}>{zh ? "继续提交" : "Continue"}</Button> : "—" },
      ]} />}</div> },
    ]} />
    <Drawer title={zh ? "提交操作与日志" : "Operations and logs"} open={logsOpen} onClose={() => setLogsOpen(false)} width="min(960px, 100vw)" destroyOnHidden><JobPanel zh={zh} scope={{ kind: "submit" }} /></Drawer>
    {taskId && <FormulaDetails key={taskId} taskId={taskId} zh={zh} onClose={() => setTaskId(undefined)} />}
  </ModulePage>;
}
