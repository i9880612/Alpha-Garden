import { requireLogin } from "./auth.ts";

export interface Page<T> { items: T[]; total: number; page?: number; page_size?: number }
export interface Formula {
  task_id: string; detail_available?: boolean; alpha_id: string | null; grade?: string | null; source?: string; status: string;
  sharpe?: number | null; fitness?: number | null; turnover?: number | null;
  full_check?: string; check_at?: string | null; check_error?: string | null; failed_checks?: string[];
  backtest_failed_checks?: { name: string; actual: number | null; threshold: number | null }[];
  attempts_used?: number; attempts_remaining?: number; parent_alpha_id?: string | null; parent_task_id?: string | null;
  run_id?: string | null; cycle_number?: number; finished_at?: string | null; date_submitted?: string;
  failure_code?: string | null; can_submit?: boolean;
}
export interface Run {
  run_id: string; status: string; current_cycle: number; max_cycles: number; max_backtests: number;
  optimization_only: boolean; automatic_submissions_enabled: boolean; created_at: string;
  started_at: string | null; finished_at: string | null; stop_reason: string | null;
  planned?: number; finished?: number; completed?: number;
  can_resume: boolean; resume_blocked_reason: string | null;
}
export interface Session { csrf_token: string; read_only: boolean; ready: boolean; account_scope: string | null; error: string | null }
export interface Progress { source: string; completed: number; planned: number }
export interface DashboardActivity {
  observed_at: string; dates: string[]; trend: number[];
  metrics: { value: number; spark: number[] }[]; weekly_backtests: number; weekly_submissions: number;
}
export interface SubmittedGrades { grades: { label: string; value: number }[] }
export interface DashboardResearch {
  observed_at: string; optimization_parents: number; optimization_attempts: number; manual_candidates: number;
}
export interface DashboardProgress { progress: Progress[]; run: Run | null }
export interface RecentBacktests { items: { task_id: string; alpha_id: string | null; formula: string; grade: string | null; sharpe: number | null }[] }
export interface BacktestPolicy {
  catalogContext: { instrumentType: string; region: string; universe: string; delay: number };
  decay: number;
  neutralization: { default: string; rootGroupNeutralize: string; byFieldCategory: Record<string, string> };
  truncation: { default: number; tailRisk: number };
  pasteurization: string; unitHandling: string; nanHandling: string; language: string;
  visualization: boolean; maxTrade: string; maxPosition: string;
}
export interface RunDefaults {
  generationCount: number; backtestCount: number; explorationPercent: number;
  selfCorrelationPercent: number; mutationPercent: number; directionValidationPercent: number;
  maxPendingSeconds: number; maxConsecutiveFailures: number; maxRequestFailures: number;
  maxInFlightBacktests: number; explorationSeedAttemptMultiplier: number;
}
export interface Settings {
  policy: BacktestPolicy; run_config: RunDefaults;
  revisions: Record<"backtest" | "run", string>;
  limits: Record<string, number | boolean>;
}
export interface Submission { task_id: string; alpha_id: string; grade: string | null; status: string; source: string; updated_at: string; failure_code: string | null }
export type JobRequest = { kind: "run"; mode: string; cycles: number } | { kind: "resume"; run_id: string } | { kind: "submit"; grade: string; count: number } | { kind: "submit"; task_id: string; source?: "qualified_archive" };
export interface Job {
  id: string; request_id: string; request: JobRequest; status: string; run_id: string | null; created_at: string;
  result: { submitted_count?: number; new_submitted_count?: number; already_active_count?: number; ineligible_count?: number; failed_count?: number; unresolved_count?: number; completed: boolean; stop_reason?: string } | null;
  error: string | null; progress: { at: string; message: string } | null; logs: { at: string; message: string }[];
}
export interface Jobs { items: Job[]; busy: boolean }

export async function getJson<T>(path: string, signal: AbortSignal): Promise<T> {
  const response = await fetch(path, { signal: AbortSignal.any([signal, AbortSignal.timeout(15000)]), cache: "no-store" });
  const value = await response.json();
  if (response.status === 401) requireLogin();
  if (!response.ok) throw new Error(value.error || "console_data_unavailable");
  return value;
}

export async function postJson(path: string, body: unknown, token: string, requestId: string) {
  const response = await fetch(path, { method: "POST", headers: {
    "Content-Type": "application/json", "X-Console-Token": token, "Idempotency-Key": requestId,
  }, body: JSON.stringify(body), signal: AbortSignal.timeout(15000) });
  const value = await response.json();
  if (response.status === 401) requireLogin();
  if (!response.ok) throw new Error(value.error || "console_operation_failed");
  return value;
}

export function errorMessage(error: string, zh: boolean) {
  const known: Record<string, [string, string]> = {
    console_settings_invalid: ["设置不符合运行规则，请检查数值、分配比例和截断范围。", "Check the values, allocation percentages and truncation limits."],
    console_settings_conflict: ["配置已被其他页面或本地文件修改。本次未保存，请重新载入后再编辑。", "Settings changed elsewhere. Nothing was saved. Reload the saved settings before editing again."],
    console_settings_busy: ["研究或提交正在进行，请暂停或等待完成后保存设置。", "Research or submission is running. Pause it or wait before saving settings."],
    formal_submission_task_not_found: ["当前账号中未找到这条公式。", "This formula was not found in the current account."],
    formal_submission_candidate_unavailable: ["当前提交条件不满足或需保留其他公式的改善机会，请刷新列表。", "Submission is unavailable or another improvement opportunity is protected. Refresh the list."],
    formal_submission_already_attempted: ["该公式已有提交记录，请到提交管理查看结果。", "This formula has a submission record. Check Submissions for the result."],
    formal_submission_other_attempt_active: ["另有未完成的提交，请先到提交管理继续处理该记录。", "Another submission is unfinished. Continue it in Submissions first."],
    formal_submission_storage_update_required: ["本地数据库尚未启用专项优化手动提交，请先完成数据库更新。", "Update the local database before manually submitting optimization formulas."],
    console_series_busy: ["另一条曲线正在获取，请稍后重试。", "Another series is being fetched. Try again shortly."],
    console_series_cooldown: ["平台暂时限制访问，请稍后重试。", "Platform access is temporarily limited. Try again later."],
    console_series_unavailable: ["曲线请求未完成，请稍后重试。", "The series request did not complete. Try again later."],
    console_formula_alpha_missing: ["这条公式尚未取得 Alpha ID。", "This formula does not have an Alpha ID yet."],
    console_formula_not_found: ["当前账号中未找到这条公式。", "This formula was not found in the current account."],
    console_read_only: ["当前服务为只读模式。", "The service is read-only."],
    console_operation_busy: ["已有操作正在执行，请等待完成或暂停后再启动。", "Another operation is running. Wait or pause it first."],
    console_data_unavailable: ["暂时无法读取本地数据，请检查服务、数据库和账号配置。", "Local data is unavailable. Check the service, database and account configuration."],
    console_token_required: ["网页会话已更新，请刷新页面后再操作。", "The console session changed. Reload before continuing."],
    console_run_not_found: ["未找到当前账号的运行记录。", "The run was not found for this account."],
    console_run_not_resumable: ["该运行已结束或没有可恢复的工作，请开始新一轮回测。", "This run has ended or has no recoverable work. Start a new run."],
    automated_run_already_failed: ["该运行已失败，且没有剩余可恢复工作。请开始新一轮回测。", "This failed run has no recoverable work. Start a new run."],
    automated_run_already_completed: ["该运行已经完成，无需恢复。", "This run is already completed."],
    console_database_busy: ["已有命令正在使用数据库，请先停止该进程后再恢复。", "A command is using the database. Stop that process before resuming."],
  };
  return known[error]?.[zh ? 0 : 1] || (zh ? "请求未完成，请查看当前操作状态和日志。" : "The request did not complete. Check the current operation and logs.");
}
