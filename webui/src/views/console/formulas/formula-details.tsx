import { Alert, Button, Card, Descriptions, Drawer, Empty, Space, Spin, Table, Tag, Typography, theme as antdTheme } from "antd";
import { useState } from "react";
import { Link, useOutletContext } from "react-router-dom";
import { useApi } from "@/api/use-api";
import { errorMessage } from "@/api/console";
import type { FormulaDetail } from "@/api/research";
import { checkFailureName, gradeName, gradeTagColors, metric, mutationDescription, sourceName, statusName, timestamp } from "@/api/presentation";
import type { ConsoleTheme } from "@/layouts/console/theme";
import { FormulaChecks } from "../components/formula-table";
import FormulaChildren from "./formula-children";
import FormulaCode from "./formula-code";
import FormulaSeries from "./formula-series";

export default function FormulaDetails({ taskId, zh, onClose }: { taskId: string; zh: boolean; onClose: () => void }) {
  const { theme } = useOutletContext<{ theme: ConsoleTheme }>();
  const { token } = antdTheme.useToken();
  const [relatedTaskId, setRelatedTaskId] = useState<string>();
  const { data, error, loading } = useApi<FormulaDetail>(`/api/formula?task_id=${encodeURIComponent(taskId)}`);
  const row = data?.summary;
  return <Drawer open title={zh ? "公式详情" : "Formula details"} width="min(1180px, 100vw)"
    maskClosable={!relatedTaskId} keyboard={!relatedTaskId} onClose={() => { if (!relatedTaskId) onClose(); }}>
    {error ? <Alert type="error" showIcon message={errorMessage(error, zh)} /> : loading && !data ? <div style={{ padding: 80, textAlign: "center" }}><Spin /></div> : data && row && <div className="ag-module-page">
      <Space wrap><Typography.Title level={4} style={{ margin: 0 }}>{row.alpha_id || (zh ? "待回测公式" : "Formula awaiting backtest")}</Typography.Title>
        <Tag color={gradeTagColors[row.grade || ""]}>{gradeName(row.grade)}</Tag>
        {data.seed_promoted_at && <Tag color="green">{zh ? "种子公式" : "Seed formula"}</Tag>}
        <FormulaChecks row={row} zh={zh} />
      </Space>
      <Card title={zh ? "表达式" : "Expression"}>
        <FormulaCode expression={data.expression} theme={theme} />
      </Card>
      <Descriptions bordered styles={{ label: { background: token.colorBgContainer, whiteSpace: "nowrap" }, content: { background: token.colorBgContainer } }} column={{ xs: 1, sm: 2, lg: 3 }} items={[
        { key: "task", label: zh ? "任务编号" : "Task ID", span: "filled", children: <Typography.Text copyable style={{ overflowWrap: "anywhere" }}>{row.task_id}</Typography.Text> },
        { key: "status", label: zh ? "回测状态" : "Backtest", children: statusName(row.status, zh) },
        { key: "source", label: zh ? "来源" : "Source", children: sourceName(row.source, zh) },
        { key: "finished", label: zh ? "结束时间" : "Finished", children: timestamp(row.finished_at) },
        { key: "run", label: zh ? "所属运行" : "Run", children: row.run_id ? <Link to={`/analysis?run_id=${encodeURIComponent(row.run_id)}&days=all`}>{zh ? "查看运行分析" : "View run analysis"}</Link> : "—" },
        { key: "cycle", label: zh ? "轮次" : "Cycle", children: row.cycle_number ?? "—" },
        { key: "parent", label: zh ? "直接父代" : "Direct parent", children: row.parent_task_id ? <Button type="link" className="ag-table-action" onClick={() => setRelatedTaskId(row.parent_task_id!)}>{row.parent_alpha_id || row.parent_task_id.slice(0, 16)}</Button> : "—" },
        { key: "attempts", label: zh ? "已用 / 剩余变异额度" : "Attempts used / left", children: `${row.attempts_used ?? "—"} / ${row.attempts_remaining ?? "—"}` },
        { key: "seed", label: zh ? "种子入池时间" : "Seed admission", children: timestamp(data.seed_promoted_at) },
        { key: "check", label: zh ? "完整检查时间" : "Full check observed", children: timestamp(row.check_at) },
      ]} />
      {(row.failure_code || row.check_error) && <Alert type="warning" showIcon message={row.failure_code || row.check_error} />}
      <Card title={zh ? "回测指标" : "Backtest metrics"}>
        {data.result ? <Descriptions column={{ xs: 1, sm: 3 }} items={(["sharpe", "fitness", "turnover", "returns", "drawdown", "margin"] as const).map(name => ({ key: name, label: name === "sharpe" ? "Sharpe" : name[0].toUpperCase() + name.slice(1), children: metric(data.result![name], ["turnover", "returns", "drawdown", "margin"].includes(name)) }))} /> : <Empty description={zh ? "尚未获得回测指标" : "Backtest metrics are not available yet"} />}
      </Card>
      <FormulaSeries key={`series-${taskId}`} taskId={taskId} alphaId={row.alpha_id} savedPnl={data.pnl} theme={theme} zh={zh} />
      <Card title={zh ? "年度表现" : "Yearly performance"}>
        <Table bordered size="middle" rowKey={item => `${item.stage}-${item.year}`} dataSource={data.yearly || []} pagination={false} scroll={{ x: "max-content" }} locale={{ emptyText: <Empty description={data.yearly === null ? (zh ? "尚未获取年度数据" : "Yearly statistics have not been retrieved") : (zh ? "未返回年度记录" : "No yearly records returned")} /> }} columns={[
          { title: zh ? "年度" : "Year", dataIndex: "year" }, { title: zh ? "阶段" : "Stage", dataIndex: "stage" },
          ...(["sharpe", "fitness", "turnover", "returns", "drawdown"] as const).map(name => ({ title: name[0].toUpperCase() + name.slice(1), dataIndex: name, render: (value: number | null) => metric(value, ["turnover", "returns", "drawdown"].includes(name)) })),
        ]} />
      </Card>
      <Card title={zh ? "平台检查结果" : "Platform check results"} extra={<Typography.Text type="secondary">{data.checks.source === "full" ? (zh ? "完整检查" : "Full check") : (zh ? "回测结束快照" : "Backtest snapshot")} · {timestamp(data.checks.observed_at)}</Typography.Text>}>
        <Table bordered size="middle" rowKey="name" dataSource={data.checks.items} pagination={false} scroll={{ x: "max-content" }} locale={{ emptyText: <Empty description={data.checks.error ? errorMessage(data.checks.error, zh) : data.checks.source === "full" ? (zh ? "本次完整检查尚未返回检查明细" : "This full check has not returned check details") : (zh ? "尚未保存检查明细" : "Check details are not saved yet")} /> }} columns={[
          { title: zh ? "检查项" : "Check", dataIndex: "name", render: (value, check) => <Space><Typography.Text>{value}</Typography.Text>{zh && check.status === "FAIL" && <Typography.Text type="secondary">{checkFailureName(value, true)}</Typography.Text>}</Space> },
          { title: zh ? "结果" : "Result", dataIndex: "status", render: value => <Tag color={value === "PASS" ? "green" : value === "FAIL" ? "red" : "default"}>{value === "PASS" ? (zh ? "通过" : "Passed") : value === "FAIL" ? (zh ? "未通过" : "Failed") : value === "PENDING" ? (zh ? "待定" : "Pending") : value === "ERROR" ? (zh ? "检查错误" : "Check error") : value}</Tag> },
          { title: zh ? "实际值" : "Value", dataIndex: "actual", render: value => value ?? "—" }, { title: zh ? "阈值" : "Limit", dataIndex: "threshold", render: value => value ?? "—" },
        ]} />
      </Card>
      <Card title={zh ? "研究设置与引用" : "Research settings and references"}>
        <Descriptions column={{ xs: 1, sm: 2, lg: 3 }} items={Object.entries(data.settings).map(([key, value]) => ({ key, label: key, children: String(value) }))} />
        <Space direction="vertical" size={12} style={{ marginTop: 16 }}>
          <Space wrap><Typography.Text type="secondary">{zh ? "字段 / 标识符" : "Fields / identifiers"}</Typography.Text>{data.references.names.map(name => <Link key={name} to={`/data?search=${encodeURIComponent(name)}`}><Tag color="blue">{name}</Tag></Link>)}</Space>
          <Space wrap><Typography.Text type="secondary">{zh ? "算子" : "Operators"}</Typography.Text>{data.references.operators.map(name => /^[a-z_]/i.test(name) ? <Link key={name} to={`/operators?search=${encodeURIComponent(name)}`}><Tag color="purple">{name}</Tag></Link> : <Tag key={name}>{name}</Tag>)}</Space>
        </Space>
      </Card>
      {data.mutation && <Card title={zh ? "本次变异" : "Mutation"}>
        <Typography.Paragraph>{mutationDescription(data.mutation, zh)}</Typography.Paragraph>
        <dl className="ag-mutation-expressions">
          <dt><Typography.Text type="secondary">{zh ? "变异前" : "Before"}:</Typography.Text></dt>
          <dd><FormulaCode expression={data.mutation.before} theme={theme} /></dd>
          <dt><Typography.Text type="secondary">{zh ? "变异后" : "After"}:</Typography.Text></dt>
          <dd><FormulaCode expression={data.mutation.after} theme={theme} /></dd>
        </dl>
      </Card>}
      <FormulaChildren key={`children-${taskId}`} taskId={taskId} zh={zh} onOpen={setRelatedTaskId} />
    </div>}
    {relatedTaskId && <FormulaDetails key={relatedTaskId} taskId={relatedTaskId} zh={zh} onClose={() => setRelatedTaskId(undefined)} />}
  </Drawer>;
}
