import { Alert, Card, Empty, Space, Tabs, Typography } from "antd";
import { useEffect, useRef, useState } from "react";
import { useApi } from "@/api/use-api";
import { requireLogin } from "@/api/auth";
import { errorMessage, type Session } from "@/api/console";
import type { FormulaDetail, PlatformSeries, SeriesMetric } from "@/api/research";
import { chartPalette, timestamp } from "@/api/presentation";
import { Chart } from "@/components/research-chart";
import AlphaLoading from "@/components/alpha-loading";
import type { ConsoleTheme } from "@/layouts/console/theme";

const labels = { pnl: "PnL", sharpe: "Sharpe", turnover: "Turnover" };
const formatValue = (metric: SeriesMetric, value: number) => metric === "pnl"
  ? `${(value / 1000).toLocaleString("en-US", { maximumFractionDigits: 2 })}K`
  : metric === "turnover" ? `${(value * 100).toFixed(2)}%` : value.toFixed(2);

export default function FormulaSeries({ taskId, alphaId, savedPnl, theme, zh }: {
  taskId: string; alphaId: string | null; savedPnl: FormulaDetail["pnl"]; theme: ConsoleTheme; zh: boolean;
}) {
  const { data: session } = useApi<Session>("/api/session");
  const [active, setActive] = useState<SeriesMetric>("pnl");
  return <Card title={zh ? "时序曲线" : "Time series"}>
    <Tabs activeKey={active} onChange={key => setActive(key as SeriesMetric)} items={(Object.keys(labels) as SeriesMetric[]).map(metric => ({
      key: metric, label: labels[metric], children: <MetricSeries taskId={taskId} alphaId={alphaId} savedPnl={savedPnl}
        metric={metric} active={active === metric} session={session} theme={theme} zh={zh} />,
    }))} />
  </Card>;
}

function MetricSeries({ taskId, alphaId, savedPnl, metric, active, session, theme, zh }: {
  taskId: string; alphaId: string | null; savedPnl: FormulaDetail["pnl"]; metric: SeriesMetric;
  active: boolean; session: Session | null; theme: ConsoleTheme; zh: boolean;
}) {
  const [remote, setRemote] = useState<PlatformSeries | null>(null);
  const [loading, setLoading] = useState(false), [error, setError] = useState<string | null>(null);
  const [attemptCount, setAttemptCount] = useState(0);
  const attempts = useRef(0);
  const csrfToken = session?.csrf_token, readOnly = session?.read_only;
  const colors = chartPalette(theme);
  useEffect(() => {
    if (!active || !alphaId || !csrfToken || readOnly || error || (remote && remote.state !== "pending") || attempts.current >= 3) return;
    const delay = remote?.refresh_after_seconds ?? 0;
    if (delay > 30) return;
    const request = new AbortController();
    // Defer dispatch so a discarded render does not issue a platform request.
    const timer = setTimeout(async () => {
      setLoading(true);
      try {
        const response = await fetch(`/api/formula/series/${metric}`, { method: "POST", headers: {
          "Content-Type": "application/json", "X-Console-Token": csrfToken,
        }, body: JSON.stringify({ task_id: taskId }), signal: AbortSignal.any([request.signal, AbortSignal.timeout(90000)]) });
        const value = await response.json();
        if (response.status === 401) requireLogin();
        if (!response.ok) throw new Error(value.error || "console_series_unavailable");
        if (!request.signal.aborted) {
          attempts.current += 1;
          setAttemptCount(attempts.current);
          setRemote(value);
        }
      } catch (cause) {
        if (!request.signal.aborted) setError(cause instanceof Error ? cause.message : "console_series_unavailable");
      } finally {
        if (!request.signal.aborted) setLoading(false);
      }
    }, delay * 1000);
    return () => { clearTimeout(timer); request.abort(); setLoading(false); };
  }, [active, alphaId, csrfToken, readOnly, taskId, metric, remote, error]);
  const fallback = metric === "pnl" && remote?.state !== "ready" ? savedPnl : null;
  const points = remote?.state === "ready" ? remote.points : fallback?.points;
  const observed = remote?.state === "ready" ? remote.observed_at : fallback?.observed_at;
  const waiting = loading || (!readOnly && !!alphaId && !error && (!remote
    || (remote.state === "pending" && attemptCount < 3 && remote.refresh_after_seconds <= 30)));
  return <Space direction="vertical" style={{ width: "100%" }} size={12}>
      {error && <Alert type="error" showIcon message={errorMessage(error, zh)} />}
          {remote?.state === "error" && <Alert type="error" showIcon style={{ marginBottom: 12 }}
            message={seriesError(remote, zh)}
            description={[remote.error, remote.http_status ? `HTTP ${remote.http_status}` : null, remote.retry_after_seconds ? (zh ? `平台建议等待 ${remote.retry_after_seconds} 秒` : `Platform retry delay: ${remote.retry_after_seconds}s`) : null].filter(Boolean).join(" · ")} />}
          {!waiting && observed && <Typography.Text type="secondary">{remote?.state === "ready" ? (zh ? "平台数据" : "Platform data") : (zh ? "本地已保存" : "Saved locally")} · {timestamp(observed)}</Typography.Text>}
          {waiting ? <div className="ag-pnl-chart ag-chart-empty"><AlphaLoading /></div> : points?.some(point => point[1] !== null) ? <Chart className="ag-research-chart ag-pnl-chart" label={`${labels[metric]} ${zh ? "时序曲线" : "time series"}`} option={{
            grid: { left: 16, right: 24, top: 24, bottom: 16, containLabel: true },
            tooltip: { trigger: "axis", confine: true, backgroundColor: colors.background, textStyle: { color: colors.text }, borderColor: colors.grid,
              valueFormatter: (value: unknown) => typeof value === "number" ? formatValue(metric, value) : "—" },
            xAxis: { type: "category", data: points.map(point => point[0]), axisLabel: { color: colors.text, hideOverlap: true, margin: 28 } },
            yAxis: { type: "value", scale: true, axisLabel: { color: colors.text, formatter: (value: number) => formatValue(metric, value) }, splitLine: { lineStyle: { color: colors.grid } } },
            series: [{ name: labels[metric], type: "line", data: points.map(point => point[1]), showSymbol: false, connectNulls: false, lineStyle: { color: "#5b7aff", width: 2 } }],
          }} /> : <div className="ag-pnl-chart ag-chart-empty"><Empty description={readOnly
            ? (zh ? "只读模式不获取平台曲线" : "Platform series are unavailable in read-only mode") : !alphaId
            ? (zh ? "尚未获得 Alpha ID" : "An Alpha ID is not available yet") : remote?.state === "ready"
            ? (zh ? "平台未返回可用曲线数据" : "The platform returned no usable data") : remote?.state === "pending"
            ? (zh ? "曲线暂未就绪" : "Series not ready yet") : (zh ? "暂无曲线数据" : "No series data available")} /></div>}
    </Space>;
}

function seriesError(series: PlatformSeries, zh: boolean) {
  if (series.http_status === 401) return zh ? "平台登录已失效，请检查账号配置。" : "Platform authentication expired. Check account settings.";
  if (series.http_status === 403) return zh ? "平台拒绝访问这条公式的曲线。" : "The platform denied access to this series.";
  if (series.http_status === 404) return zh ? "平台未找到这条公式的曲线。" : "The platform could not find this series.";
  if (series.http_status === 429) return zh ? "平台请求频率受限，请稍后重试。" : "Platform rate limit reached. Try again later.";
  if (series.error?.includes("invalid")) return zh ? "平台曲线数据格式异常。" : "The platform returned an invalid series.";
  return zh ? "未能获取这条曲线，请稍后重试。" : "This series could not be retrieved. Try again later.";
}
