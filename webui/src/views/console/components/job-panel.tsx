import { Alert, Button, Empty, Select, Space, Tag, Typography } from "antd";
import { Fragment, useLayoutEffect, useRef, useState } from "react";
import { useConsole } from "@/api/use-console";
import { statusName, timestamp } from "@/api/presentation";
import { presentLog } from "@/api/log-presentation";
import { errorMessage } from "@/api/console";
import AlphaLoading from "@/components/alpha-loading";

type JobScope = { jobId: string } | { kind: "submit" };

const logTokenColors: Record<string, string> = {
  "[通过]": "success",
  "[未通过]": "error",
  "[待定]": "warning",
  "[异常]": "error",
  "[SPECTACULAR]": "purple",
  "[EXCELLENT]": "success",
  "[GOOD]": "lime",
  "[AVERAGE]": "info",
  "[INFERIOR]": "muted",
  "[未知]": "muted",
  "[探索]": "info",
  "[变异]": "purple",
  "[SC治理]": "warning",
  "[反转]": "lime",
  "[PASSED]": "success",
  "[FAILED]": "error",
  "[PENDING]": "warning",
  "[ERROR]": "error",
  "[UNKNOWN]": "muted",
  "[EXPLORATION]": "info",
  "[MUTATION]": "purple",
  "[SC REPAIR]": "warning",
  "[REVERSAL]": "lime",
};

function LogTokens({ text }: { text: string }) {
  return text.split(/(\[[^\]\r\n]+\])/g).map((part, index) => {
    const color = logTokenColors[part.toUpperCase()];
    return color ? <span key={index} className={`ag-log-${color}`}>{part}</span> : part;
  });
}

function LogText({ message, zh }: { message: string; zh: boolean }) {
  // Terminal space padding cannot align CJK fallback fonts in the browser.
  // Copy the displayed text while keeping result fields aligned in both languages.
  const { text, fields, detail, tone } = presentLog(message, zh);
  if (fields && detail !== undefined) {
    const metrics = /^(Sharpe\s+)(-?\d+\.\d+)(\s+Fitness\s+)(-?\d+\.\d+)(\s+Turnover\s+)(-?\d+\.\d+%)(.*)$/.exec(detail);
    return (
      <span className="ag-log-result">
        {fields.map((part, index) => <span key={index}><LogTokens text={part} /></span>)}
        {metrics ? (
          <span className="ag-log-metrics">
            {metrics.slice(1, 7).map((part, index) => <span key={index} className={index % 2 ? "ag-log-number" : undefined}>{part}</span>)}
            {metrics[7]}
          </span>
        ) : <span>{detail}</span>}
      </span>
    );
  }
  return (
    <span className={tone ? `ag-log-${tone}` : undefined}>
      <LogTokens text={text} />
    </span>
  );
}

export function JobPanel({ zh, scope }: { zh: boolean; scope: JobScope }) {
  const { jobs, session, pending, stop } = useConsole();
  const [selected, setSelected] = useState<string>();
  const logRef = useRef<HTMLPreElement>(null);
  const items =
    jobs.data?.items.filter((item) =>
      "jobId" in scope ? item.id === scope.jobId : item.request.kind === scope.kind,
    ) || [];
  const job = items.find((item) => item.id === selected) || items[0];
  useLayoutEffect(() => {
    const log = logRef.current;
    if (!log) return;
    const followLatest = () => {
      log.scrollTop = log.scrollHeight;
    };
    followLatest();
    const observer = new ResizeObserver(followLatest);
    observer.observe(log);
    return () => observer.disconnect();
  }, [job?.id, job?.logs, jobs.error]);
  if (jobs.error) return <Alert type="error" message={errorMessage(jobs.error, zh)} />;
  if ((jobs.loading && !jobs.data) || (pending && "jobId" in scope && !job))
    return <AlphaLoading />;
  return (
    <div className="ag-job-panel">
      <Typography.Paragraph type="secondary">
        {zh
          ? "仅显示本次网页服务保留的相关操作，每次操作最多 500 条日志。暂停会等待当前请求结束。"
          : "Related operations retained by this web service, up to 500 log lines each. Pause waits for the current request to finish."}
      </Typography.Paragraph>
      {job ? (
        <>
          <Space wrap className="ag-filter-row">
            {items.length > 1 ? (
              <Select
                aria-label={zh ? "选择操作日志" : "Select operation logs"}
                value={job.id}
                onChange={setSelected}
                style={{ width: 290, maxWidth: "100%" }}
                options={items.map((item) => ({
                  value: item.id,
                  label: `${timestamp(item.created_at)} · ${item.request.kind === "resume" ? (zh ? "恢复" : "Resume") : item.request.kind === "run" ? (zh ? "运行" : "Run") : zh ? "提交" : "Submit"}`,
                }))}
              />
            ) : (
              <Typography.Text>{timestamp(job.created_at)}</Typography.Text>
            )}
            <Tag color={job.status === "running" ? "success" : undefined}>
              {statusName(job.status, zh)}
            </Tag>
            <Typography.Text copyable={Boolean(job.run_id)}>{job.run_id}</Typography.Text>
            {job.id === jobs.data?.items[0]?.id && ["running", "stopping"].includes(job.status) && (
              <Button
                disabled={session.data?.read_only || pending || job.status === "stopping"}
                onClick={() => stop(job.id)}
              >
                {zh ? "暂停操作" : "Pause operation"}
              </Button>
            )}
          </Space>
          {job.result && (
            <Typography.Paragraph>
              {job.request.kind === "submit"
                ? zh
                  ? `本次成功 ${job.result.new_submitted_count ?? 0} 条，已提交 ${job.result.already_active_count ?? 0} 条，不符合条件 ${job.result.ineligible_count ?? 0} 条，失败 ${job.result.failed_count ?? 0} 条，待确认 ${job.result.unresolved_count ?? 0} 条。`
                  : `Submitted: ${job.result.new_submitted_count ?? 0}; already active: ${job.result.already_active_count ?? 0}; ineligible: ${job.result.ineligible_count ?? 0}; failed: ${job.result.failed_count ?? 0}; unresolved: ${job.result.unresolved_count ?? 0}.`
                : `${zh ? "停止原因" : "Stop reason"}: ${job.result.stop_reason || "—"}`}
            </Typography.Paragraph>
          )}
          {job.error && (
            <Alert
              type="warning"
              showIcon
              message={errorMessage(job.error, zh)}
              description={job.error}
            />
          )}
          {job.progress && ["running", "stopping"].includes(job.status) && (
            <div className="ag-job-progress" role="status">
              <Typography.Text type="secondary">
                {zh ? "最近进度" : "Latest progress"} ·{" "}
                {new Date(job.progress.at).toLocaleTimeString()}
              </Typography.Text>
              <Typography.Text>{presentLog(job.progress.message, zh).text}</Typography.Text>
            </div>
          )}
          <pre
            ref={logRef}
            className="ag-operation-log"
            lang={zh ? "zh-CN" : "en-US"}
            aria-label={zh ? "操作日志" : "Operation logs"}
          >
            {job.logs.length
              ? job.logs.map((line, index) => {
                  return (
                    <Fragment key={`${line.at}-${index}`}>
                      {index > 0 ? "\n" : ""}
                      <span className="ag-log-muted">{new Date(line.at).toLocaleTimeString()}</span>{" "}
                      <LogText message={line.message} zh={zh} />
                    </Fragment>
                  );
                })
              : zh
                ? "等待操作日志…"
                : "Waiting for operation logs…"}
          </pre>
        </>
      ) : (
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description={zh ? "暂无保留的相关操作日志" : "No retained logs for this operation"}
        />
      )}
    </div>
  );
}
