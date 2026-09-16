// Translate only the engine's known message templates. Platform diagnostics,
// identifiers and unknown messages remain verbatim; stored logs are never changed.
const messages: Record<string, string> = {
  "通过": "Passed", "未通过": "Failed", "待定": "Pending", "异常": "Error", "未知": "Unknown",
  "探索": "Exploration", "变异": "Mutation", "SC治理": "SC repair", "反转": "Reversal",
  "平台认证中...": "Authenticating with the platform...",
  "平台认证完成": "Platform authentication completed",
  "评级提升专项：读取完整检查通过的活动父代...": "Grade optimization: loading active parents that passed all checks...",
  "公式生成中，读取研究证据并准备探索与改善候选...": "Generating formulas: loading research evidence and preparing exploration and improvement candidates...",
  "公式筛选与回测计划冻结中...": "Selecting formulas and finalizing the backtest plan...",
  "开始回测，按可用并发名额推进": "Starting backtests using available concurrent slots",
  "结算与学习反馈中...": "Settling results and processing learning feedback...",
  "结算与学习反馈完成": "Settlement and learning feedback completed",
  "评级提升专项无可继续变异的合格候选，本次结束": "Grade optimization finished: no eligible candidates remain for further mutation",
  "状态为完成，但指标缺失，不能展示得分": "Marked completed, but metrics are missing; scores are unavailable",
  "发送失败（429重试耗尽，未被平台接受）": "Request failed (429 retries exhausted; not accepted by the platform)",
  "结果读取失败；远端结果未确认": "Could not read results; remote outcome is unconfirmed",
  "接收超时；远端未知，不重发": "Acceptance timed out; remote outcome is unknown; no resend",
  "等待超时；远端结果未决": "Wait timed out; remote outcome is unresolved",
  "回测请求发送中，尚未确认接收": "Sending backtest request; acceptance is not yet confirmed",
  "提交结果未知，保留名额，只对账、不重发": "Submission outcome unknown; slot retained; reconciling without resending",
  "旧请求收取暂未成功，保留名额，稍后重查": "Could not collect the previous request; slot retained; checking again later",
  "指标已收到，等待年度数据；尚未完整完成": "Metrics received; waiting for annual data; not fully completed",
  "平台计算已结束，等待完整指标与检查": "Platform calculation finished; waiting for complete metrics and checks",
  "已提交回测，等待结果": "Backtest submitted; waiting for results",
  "收取结果遇到429限流，该任务延后检查，结果保持待定": "Result polling rate limited (429); checking this task later; outcome remains pending",
  "收取结果暂未成功，该任务延后检查，不影响发送限流计数": "Result polling unsuccessful; checking this task later; send rate-limit count unchanged",
  "发送回测处于限流冷却，批次继续保留": "Backtest requests are cooling down after rate limiting; batch retained",
  "回测名额已满，等待结果并对账，不重复发送": "All backtest slots occupied; waiting for results and reconciling; no duplicate requests",
  "回测仍在进行，按平台要求等待后继续检查": "Backtest still running; waiting as requested by the platform before checking again",
  "入队检查等待平台重试时间；不会生成或发送下一批公式": "Queue checks are waiting for the platform retry time; next batch will not be generated or sent yet",
  "进度暂时无法读取；任务仍按原流程执行。": "Progress temporarily unavailable; tasks continue as planned.",
  "提交进度暂不可读；不改变提交状态": "Submission progress temporarily unavailable; submission state unchanged",
  "操作未完成；请检查运行状态、账号配置及是否已有命令占用数据库。": "Operation did not complete; check the run status, account configuration and whether another command is using the database.",
  "正式提交：平台认证中": "Formal submission: authenticating with the platform",
  "正式提交：平台认证完成": "Formal submission: platform authentication completed",
  "正式提交：会话已重新认证": "Formal submission: session reauthenticated",
  "正在收尾已中断的研究批次：取消未发送任务，收取已发送回测；本次只提交已选定的公式。": "Finishing an interrupted research batch: cancelling unsent tasks and collecting sent backtests; only the selected formulas will be submitted.",
  "读取详情": "Reading details", "检查中": "Checking", "检查通过": "Checks passed",
  "发送中": "Sending", "确认中": "Confirming", "结果未知": "Outcome unknown",
  "已提交": "Submitted", "检查未过": "Checks failed", "失败": "Failed",
  "提交前等待超时，已跳过": "Pre-submission wait timed out; skipped",
  "提交前读取失败，已跳过": "Pre-submission read failed; skipped",
  "评级未确认，已跳过": "Grade unconfirmed; skipped",
  "通过，已核对入队资格": "Passed; queue eligibility checked",
  "未通过，不入队": "Failed; not queued",
  "待定，不入队": "Pending; not queued",
  "检查通过；评级未确认，不入队": "Checks passed; grade unconfirmed; not queued",
  "暂未就绪": "not ready yet",
};

const templates: [RegExp, string][] = [
  [/^变异候选已选定，共 (\d+) 条$/, "Selected $1 mutation candidates"],
  [/^公式生成完成，共 (\d+) 条候选$/, "Formula generation completed: $1 candidates"],
  [/^筛选完成，已冻结本轮 (\d+) 条回测任务$/, "Selection completed: finalized $1 backtest tasks for this cycle"],
  [/^候选规划未完成（(.+)），按已有任务状态收尾$/, "Candidate planning incomplete ($1); finishing according to existing task states"],
  [/^平台认证暂未成功（(.+)），按原故障边界处理$/, "Platform authentication unsuccessful ($1); applying the configured failure limits"],
  [/^请求暂未成功（(.+)），([\d.]+) 秒后重试$/, "Request unsuccessful ($1); retrying in $2 seconds"],
  [/^请求退避等待中，剩余 ([\d.]+) 秒；不会提前重发。$/, "Request backoff: $1 seconds remaining; no early resend."],
  [/^发送回测遇到429限流，冷却([\d.]+)秒；重试(\d+\/\d+)，继续收取在途结果$/, "Backtest request rate limited (429); cooling down for $1 seconds; retry $2; continuing to collect in-flight results"],
  [/^自动正式提交：(.+)$/, "Automatic formal submission: $1"],
  [/^正式提交：本次处理 (\d+) 条（含待恢复项）$/, "Formal submission: processing $1 candidates (including resumable items)"],
  [/^正式提交：本次候选 (\d+) 条（含待恢复项），成功提交 (\d+) 条后停止；检查失败跳过，原已提交的同步不计入数量。$/, "Formal submission: $1 candidates (including resumable items); stop after $2 successful submissions; skip failed checks; syncing previously submitted formulas does not count."],
  [/^正式提交：(\S+) 本地相关性或改善机会待重新确认，暂缓提交；未发送提交请求。$/, "Formal submission: $1 local correlation or improvement opportunities need rechecking; deferred; no submission request sent."],
  [/^提交前读取暂未成功（(.+)），([\d.]+) 秒后重试；尚未发送正式提交。$/, "Pre-submission read unsuccessful ($1); retrying in $2 seconds; formal submission has not been sent."],
  [/^评级 (\S+)，未达 SPECTACULAR，已跳过$/, "Grade $1 is below SPECTACULAR; skipped"],
  [/^评级从 (\S+) 变为 (\S+)，不符合所选等级，已跳过$/, "Grade changed from $1 to $2; no longer matches the selected grade; skipped"],
  [/^正在处理 (\d+\/\d+) (\S+)$/, "Processing $1 $2"],
  [/^通过(\d+) 已交(\d+) 未过(\d+) 失败(\d+) 待定(\d+)$/, "Passed $1 Submitted $2 Ineligible $3 Failed $4 Pending $5"],
  [/^请求异常（(.+)），不入队$/, "Request error ($1); not queued"],
  [/^检查通过；评级 (\S+) 未达 SPECTACULAR，不入队$/, "Checks passed; grade $1 is below SPECTACULAR; not queued"],
  [/^正在获取时序数据 (\S+)（已获取 (\d+\/\d+)）$/, "Fetching time series for $1 (collected $2)"],
  [/^时序数据 (\S+) 获取完成（已获取 (\d+\/\d+)）$/, "Time series collected for $1 (collected $2)"],
  [/^读取失败（(.+)）$/, "read failed ($1)"],
  [/^已确认提交事实已保存，但公式清单更新失败（(.+)）；可运行 (.+) 重建。$/, "Confirmed submission saved, but formula export failed ($1); run $2 to rebuild it."],
];

const resultPattern = /^(\[(?:通过|未通过|待定|异常)\])\s+(\[[^\]\r\n]+\])\s+(第\d+轮)\s+(\d+\/\d+)\s+(\[(?:探索|变异|SC治理|反转)\])\s+(\S+)\s+(.*)$/;

function expandMetrics(message: string) {
  return message.replace(/^S\s*(-?\d+\.\d+) F\s*(-?\d+\.\d+) T\s*(-?\d+\.\d+%)(?= 旧\S+$|$)/,
    (_, sharpe: string, fitness: string, turnover: string) =>
      `Sharpe ${sharpe.padStart(6)}  Fitness ${fitness.padStart(6)}  Turnover ${turnover.padStart(6)}`);
}

function translate(message: string): string {
  if (Object.hasOwn(messages, message)) return messages[message];
  let match = /^\[(通过|未通过|待定|异常|未知|探索|变异|SC治理|反转)\]$/.exec(message);
  if (match) return `[${messages[match[1]]}]`;
  match = /^第(\d+)轮$/.exec(message);
  if (match) return `Cycle ${match[1]}`;
  match = /^(?:第 (\d+) 轮 )?阶段\[(\d+)\] (.*)$/.exec(message);
  if (match) return `${match[1] ? `Cycle ${match[1]} ` : ""}Phase[${match[2]}] ${translate(match[3])}`;
  match = /^第(\d+)轮 完成(\d+\/\d+) 在途(\d+) 待发(\d+) \| (\[(?:探索|变异|SC治理|反转)\]) (\S+) (.*)$/.exec(message);
  if (match) return `Cycle ${match[1]} Completed ${match[2]} In flight ${match[3]} Queued ${match[4]} | ${translate(match[5])} ${match[6]} ${translate(match[7])}`;
  match = /^(.*) 旧(\S+)$/.exec(message);
  if (match) return `${translate(match[1])} Previous run ${match[2]}`;
  match = /^回测失败（([^）]+)）(：.*)?$/.exec(message);
  if (match) return `Backtest failed (${match[1] === "原因未提供" ? "reason not provided" : match[1]})${match[2] ? `: ${match[2].slice(1)}` : ""}`;
  match = /^(种子|恢复)相关性：(.*)$/.exec(message);
  if (match) return `${match[1] === "种子" ? "Seed" : "Recovery"} correlation: ${translate(match[2])}`;
  match = /^时序数据 (\S+) (.+)，证据待定，([\d.]+) 秒后可重试$/.exec(message);
  if (match) return `Time series for ${match[1]} ${translate(match[2])}; evidence pending; retry available in ${match[3]} seconds`;
  match = /^时序数据 (\S+) (读取失败（.+）)$/.exec(message);
  if (match) return `Time series for ${match[1]} ${translate(match[2])}`;
  match = /^提交进度 (\d+\/\d+) \| (.*)$/.exec(message);
  if (match) return `Submission progress ${match[1]} | ${translate(match[2])}`;
  match = /^提交 (\d+\/\d+) (\S+) (.*) \| (.*)$/.exec(message);
  if (match) return `Submission ${match[1]} ${match[2]} ${translate(match[3])} | ${translate(match[4])}`;
  match = /^(旧|本批)入队检查 (\S+)：(.*)$/.exec(message);
  if (match) return `${match[1] === "旧" ? "Previous" : "Current batch"} queue check ${match[2]}: ${translate(match[3])}`;
  const deferred = "；已安排延后自动复查，不阻塞回测";
  if (message.endsWith(deferred)) return `${translate(message.slice(0, -deferred.length))}; automatic recheck scheduled; backtests are not blocked`;
  for (const [pattern, replacement] of templates) {
    if (pattern.test(message)) return message.replace(pattern, replacement);
  }
  return message;
}

const resultLegend = "结果列：状态 | 平台评级 | 轮次/序号 | 探索/SC治理/变异/反转 | Alpha ID | S=夏普 F=适应度 T=换手率；通过/未通过/待定仅表示基础检查，评级缺失显示未知，异常见行末原因";

export function presentLog(message: string, zh: boolean) {
  const match = resultPattern.exec(message);
  const fields = match?.slice(1, 7).map((part) => `${zh ? part : translate(part)} `);
  const rawDetail = match ? expandMetrics(match[7]) : undefined;
  const detail = rawDetail === undefined ? undefined : zh ? rawDetail : translate(rawDetail);
  let text = fields ? fields.join("") + detail : zh ? message : translate(message);
  if (message === resultLegend) {
    text = zh ? message.replace("；", "\n说明：")
      : "Columns: Status | Platform grade | Cycle/Sequence | Exploration/SC repair/Mutation/Reversal | Alpha ID | S=Sharpe F=Fitness T=Turnover\nNote: Passed/Failed/Pending refer to basic checks only; missing grades show Unknown; errors include their reason at the end of the line.";
  }
  const warning = /^(?:请求暂未成功|发送回测遇到429限流|收取结果遇到429限流|收取结果暂未成功|发送回测处于限流冷却|进度暂时无法读取|提交进度暂不可读|入队检查等待平台重试时间|提交前读取暂未成功|已确认提交事实已保存)/.test(message)
    || /^(?:恢复|种子)相关性：.*(?:读取失败|证据待定)/.test(message);
  const tone = message.startsWith("操作未完成；") ? "error" : warning ? "warning"
    : /^(?:第 \d+ 轮 )?阶段\[\d+\]/.test(message) ? "info" : undefined;
  return { text, fields, detail, tone };
}
