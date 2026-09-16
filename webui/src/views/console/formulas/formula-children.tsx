import { Alert, Card, Table } from "antd";
import { useState } from "react";
import { errorMessage, type Formula, type Page } from "@/api/console";
import { metric, sourceName, statusName } from "@/api/presentation";
import { useApi } from "@/api/use-api";
import { formulaRowProps } from "../components/formula-row";

export default function FormulaChildren({ taskId, zh, onOpen }: { taskId: string; zh: boolean; onOpen: (taskId: string) => void }) {
  const [page, setPage] = useState(1);
  const { data, error, loading } = useApi<Page<Formula>>(`/api/formula/children?task_id=${encodeURIComponent(taskId)}&page=${page}&page_size=10`);
  // Keep the current rows mounted while the next page loads, so the drawer does not collapse.
  const [displayed, setDisplayed] = useState(data);
  if (data && data !== displayed) setDisplayed(data);
  const rows = data ?? displayed;
  const total = rows?.total ?? "—";

  return <Card title={zh ? `直接衍生公式（${total}）` : `Direct descendants (${total})`}>
    {error && <Alert type="error" showIcon message={errorMessage(error, zh)} style={{ marginBottom: 16 }} />}
    <Table bordered rowKey="task_id"
      onRow={child => child.detail_available ? formulaRowProps(child, zh, onOpen) : {}}
      dataSource={error ? [] : rows?.items} loading={loading}
      scroll={{ x: "max-content", scrollToFirstRowOnChange: false }}
      pagination={{ current: page, pageSize: 10, total: rows?.total, showSizeChanger: false, onChange: setPage }}
      columns={[
        { title: "Alpha ID", dataIndex: "alpha_id", render: value => value || "—" },
        { title: zh ? "状态" : "Status", dataIndex: "status", render: value => statusName(value, zh) },
        { title: zh ? "来源" : "Source", dataIndex: "source", render: value => sourceName(value, zh) },
        { title: "Sharpe", dataIndex: "sharpe", render: value => metric(value) },
        { title: "Fitness", dataIndex: "fitness", render: value => metric(value) },
        { title: "Turnover", dataIndex: "turnover", render: value => metric(value, true) },
      ]} />
  </Card>;
}
