import { Alert, Table, Tag, Tooltip } from "antd";
import type { ReactNode } from "react";
import type { Formula, Page } from "@/api/console";
import { errorMessage } from "@/api/console";
import {
  checkFailureName,
  gradeTagColors,
  gradeName,
  metric,
  sourceName,
  statusName,
  timestamp,
} from "@/api/presentation";
import { UnavailableData } from "./connection-notice";
import { formulaRowProps } from "./formula-row";

export function FormulaChecks({ row, zh }: { row: Formula; zh: boolean }) {
  if (row.status === "created") return <Tag>{zh ? "待回测" : "Awaiting backtest"}</Tag>;
  if (row.status === "submission_unknown")
    return <Tag>{zh ? "等待接收确认" : "Awaiting acceptance"}</Tag>;
  if (row.status === "pending")
    return <Tag>{zh ? "等待回测结果" : "Awaiting backtest result"}</Tag>;
  const formalObserved = Boolean(row.check_at || row.check_error);
  const backtestFailures =
    !formalObserved && row.full_check !== "passed" ? row.backtest_failed_checks || [] : [];
  const fromBacktest = backtestFailures.length > 0;
  const failed = fromBacktest || row.full_check === "failed";
  const names = fromBacktest
    ? backtestFailures.map((check) => check.name)
    : row.failed_checks || [];
  const belowThreshold = names.some((name) =>
    ["LOW_SHARPE", "LOW_FITNESS", "LOW_TURNOVER", "HIGH_TURNOVER"].includes(name),
  );
  const tooltip = fromBacktest
    ? [
        zh ? "回测检查" : "Backtest checks",
        timestamp(row.finished_at),
        ...backtestFailures.map((check) =>
          [
            check.name,
            check.actual == null ? null : `${zh ? "实际" : "Value"} ${check.actual}`,
            check.threshold == null ? null : `${zh ? "阈值" : "Limit"} ${check.threshold}`,
          ]
            .filter(Boolean)
            .join(" · "),
        ),
        zh ? "尚未执行完整检查" : "Full check has not been performed",
      ]
    : [zh ? "完整检查" : "Full check", timestamp(row.check_at), ...names, row.check_error];
  const unchecked = zh ? "待检查" : "Awaiting checks";
  return (
    <Tooltip
      styles={{
        root: { maxWidth: "calc(100vw - 24px)" },
        body: { whiteSpace: "nowrap", overflowX: "auto" },
      }}
      title={tooltip.filter(Boolean).join(" · ")}
    >
      {failed ? (
        <div className="ag-check-failures">
          {belowThreshold ? (
            <Tag color="error">{zh ? "未达标" : "Below threshold"}</Tag>
          ) : names.length ? (
            names.map((name) => (
              <Tag color="error" key={name}>
                {checkFailureName(name, zh)}
              </Tag>
            ))
          ) : (
            <Tag color="error">{zh ? "失败（原因未记录）" : "Failed (reason not recorded)"}</Tag>
          )}
        </div>
      ) : (
        <Tag
          color={row.check_error ? "warning" : row.full_check === "passed" ? "success" : "default"}
        >
          {row.check_error
            ? zh
              ? "检查异常"
              : "Check error"
            : !formalObserved && row.full_check !== "passed"
              ? unchecked
              : statusName(row.full_check, zh)}
        </Tag>
      )}
    </Tooltip>
  );
}

export function FormulaTable({
  data,
  loading,
  error,
  zh,
  page,
  pageSize,
  onPage,
  onOpen,
  actions,
}: {
  data?: Page<Formula> | null;
  loading: boolean;
  error: string | null;
  zh: boolean;
  page: number;
  pageSize: number;
  onPage: (page: number, pageSize: number) => void;
  onOpen: (taskId: string) => void;
  actions?: (row: Formula) => ReactNode;
}) {
  if (error) return <Alert type="error" showIcon message={errorMessage(error, zh)} />;
  return (
    <Table<Formula>
      className="ag-formula-table"
      bordered
      rowKey="task_id"
      size="large"
      loading={loading}
      dataSource={data?.items || []}
      scroll={{ x: actions ? 1240 : 1120 }}
      onRow={(row) => (row.detail_available ? formulaRowProps(row, zh, onOpen) : {})}
      locale={{ emptyText: <UnavailableData /> }}
      pagination={{
        current: page,
        total: data?.total || 0,
        pageSize,
        showSizeChanger: true,
        pageSizeOptions: [10, 20, 50, 100],
        onChange: onPage,
        showTotal: (total) => (zh ? `共 ${total} 条` : `${total} records`),
      }}
      columns={[
        {
          title: "Alpha ID",
          dataIndex: "alpha_id",
          width: 120,
          render: (value, row) => (
            <Tooltip title={row.task_id}>
              <span>
                {value || (row.detail_available ? (zh ? "查看公式" : "View formula") : "—")}
              </span>
            </Tooltip>
          ),
        },
        {
          title: zh ? "评级" : "Grade",
          dataIndex: "grade",
          render: (value) => <Tag color={gradeTagColors[value]}>{gradeName(value)}</Tag>,
        },
        {
          title: zh ? "来源" : "Source",
          dataIndex: "source",
          render: (value) => sourceName(value, zh),
        },
        { title: "Sharpe", dataIndex: "sharpe", align: "left", render: (value) => metric(value) },
        { title: "Fitness", dataIndex: "fitness", align: "left", render: (value) => metric(value) },
        {
          title: "Turnover",
          dataIndex: "turnover",
          align: "left",
          render: (value) => metric(value, true),
        },
        {
          title: zh ? "检查结果" : "Check results",
          width: zh ? 180 : 240,
          render: (_, row) => <FormulaChecks row={row} zh={zh} />,
        },
        {
          title: zh ? "回测状态" : "Backtest",
          dataIndex: "status",
          render: (value) =>
            value === "created" ? (zh ? "未开始" : "Not started") : statusName(value, zh),
        },
        {
          title: zh ? "剩余额度" : "Attempts left",
          dataIndex: "attempts_remaining",
          align: "left",
          render: (value, row) => (
            <Tooltip
              title={
                zh
                  ? `已使用 ${row.attempts_used ?? "—"} 次；活动资格由后端另行判断`
                  : `Used: ${row.attempts_used ?? "—"}. Active eligibility is determined separately.`
              }
            >
              {value ?? "—"}
            </Tooltip>
          ),
        },
        {
          title: zh ? "直接父代" : "Direct parent",
          dataIndex: "parent_alpha_id",
          render: (value, row) => <Tooltip title={row.parent_task_id}>{value || "—"}</Tooltip>,
        },
        ...(actions
          ? [
              {
                title: zh ? "操作" : "Action",
                key: "actions",
                fixed: "right" as const,
                width: 100,
                render: (_: unknown, row: Formula) => actions(row),
              },
            ]
          : []),
      ]}
    />
  );
}
