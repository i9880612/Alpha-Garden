import { QuestionCircleOutlined } from "@ant-design/icons";
import { Alert, Button, Input, Table, Tabs, Tag, Typography, theme } from "antd";
import { useState } from "react";
import { useOutletContext } from "react-router-dom";
import type { ConsoleLocale } from "@/layouts/console/locale";
import type { Seed, Seeds } from "@/api/research";
import { useApi } from "@/api/use-api";
import { errorMessage } from "@/api/console";
import { gradeTagColors, gradeName, metric, timestamp } from "@/api/presentation";
import ModulePage from "../components/module-page";
import { formulaRowProps } from "../components/formula-row";
import FormulaDetails from "../formulas/formula-details";

export default function SeedsPage() {
  const { locale } = useOutletContext<{ locale: ConsoleLocale }>(), zh = locale === "zh-CN";
  const { token } = theme.useToken();
  const [mode, setMode] = useState("roots"), [search, setSearch] = useState("");
  const [page, setPage] = useState(1), [pageSize, setPageSize] = useState(10);
  const [taskId, setTaskId] = useState<string>();
  const { data, loading, error } = useApi<Seeds>(`/api/seeds?${new URLSearchParams({ mode, search, page: String(page), page_size: String(pageSize) })}`);
  const tabs = [
    { key: "roots", label: zh ? "已入池种子" : "Admitted seeds", description: zh ? "种子由研究引擎按真实结果和相关性检查自动入池，种子本身可能已由更好的后代接替。" : "Seeds are admitted by the research engine using actual results and correlation checks. A seed may have been replaced by a stronger descendant." },
    { key: "normal", label: zh ? "活动研究分支" : "Active research branches", description: zh ? "活动研究分支是当前可继续投入普通研究的公式。" : "Active research branches are formulas currently available for further normal research." },
  ];
  return <ModulePage>
    <Tabs activeKey={mode} destroyOnHidden onChange={value => { setMode(value); setPage(1); }}
      tabBarExtraContent={{ right: <Input.Search allowClear maxLength={128} aria-label={zh ? "搜索种子" : "Search seeds"} placeholder={zh ? "搜索 Alpha ID 或种子编号" : "Search Alpha or seed ID"} style={{ width: 280 }} onSearch={value => { setSearch(value); setPage(1); }} /> }}
      items={tabs.map(tab => ({ key: tab.key, label: tab.label, children: <>
      <Typography.Paragraph type="secondary" className="ag-formula-note"><QuestionCircleOutlined aria-hidden style={{ color: token.colorPrimary, fontSize: 16 }} /><span>{tab.description}</span></Typography.Paragraph>
      {error ? <Alert type="error" showIcon message={errorMessage(error, zh)} /> : <Table<Seed> bordered rowKey={row => `${row.root_task_id}:${row.task_id}`} onRow={row => formulaRowProps(row, zh, setTaskId)} dataSource={data?.items || []} loading={loading} scroll={{ x: "max-content" }} pagination={{ current: page, pageSize, total: data?.total || 0, showSizeChanger: true, pageSizeOptions: [10, 20, 50, 100], showTotal: total => zh ? `共 ${total} 条` : `${total} records`, onChange: (next, size) => { setPage(size === pageSize ? next : 1); setPageSize(size); } }} columns={[
        { title: mode === "roots" ? (zh ? "种子 Alpha ID" : "Seed Alpha ID") : "Alpha ID", dataIndex: "alpha_id", render: value => value || "—" },
        { title: zh ? "评级" : "Grade", dataIndex: "grade", render: value => <Tag color={gradeTagColors[value]}>{gradeName(value)}</Tag> },
        { title: "Sharpe", dataIndex: "sharpe", render: value => metric(value) },
        { title: "Fitness", dataIndex: "fitness", render: value => metric(value) },
        { title: "Turnover", dataIndex: "turnover", render: value => metric(value, true) },
        { title: zh ? "已用 / 剩余变异额度" : "Attempts used / left", render: (_, row) => `${row.attempts_used ?? "—"} / ${row.attempts_remaining ?? "—"}` },
        ...(mode === "normal" ? [{ title: zh ? "根种子" : "Root seed", key: "root_seed", render: (_: unknown, row: Seed) => <Button type="link" className="ag-table-action" onClick={() => setTaskId(row.root_task_id)}>{zh ? "查看根种子" : "View root seed"}</Button> }] : []),
        { title: zh ? "入池时间" : "Admitted", dataIndex: "promoted_at", render: value => timestamp(value) },
      ]} />}
      </> }))}
    />
    {taskId && <FormulaDetails key={taskId} taskId={taskId} zh={zh} onClose={() => setTaskId(undefined)} />}
  </ModulePage>;
}
