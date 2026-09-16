import assert from "node:assert/strict";
import test from "node:test";
import { presentLog } from "../../src/api/log-presentation.ts";

test("backtest results switch languages without changing grades, IDs or metrics", () => {
  const raw = "[通过]   [SPECTACULAR] 第12轮 041/100 [变异]   Demo041    S  2.64 F  3.68 T 10.9%";
  const english = presentLog(raw, false);
  assert.deepEqual(english.fields, ["[Passed] ", "[SPECTACULAR] ", "Cycle 12 ", "041/100 ", "[Mutation] ", "Demo041 "]);
  assert.equal(english.detail, "Sharpe   2.64  Fitness   3.68  Turnover  10.9%");
  const chinese = presentLog(raw, true);
  assert.equal(chinese.fields[0], "[通过] ");
  assert.equal(chinese.fields[2], "第12轮 ");
  assert.equal(chinese.detail, english.detail);
  assert.deepEqual(presentLog(raw, false), english);
});

test("all result states and research sources are localized, including recovered results", () => {
  for (const [status, expectedStatus, source, expectedSource] of [
    ["未通过", "Failed", "探索", "Exploration"],
    ["待定", "Pending", "SC治理", "SC repair"],
    ["异常", "Error", "反转", "Reversal"],
  ]) {
    const result = presentLog(`[${status}] [未知] 第1轮 02/16 [${source}] alpha-02 Sharpe -1.20 Fitness -0.70 Turnover 2.0% 旧abcdef12`, false);
    assert.equal(result.fields[0], `[${expectedStatus}] `);
    assert.equal(result.fields[1], "[Unknown] ");
    assert.equal(result.fields[4], `[${expectedSource}] `);
    assert.equal(result.detail, "Sharpe -1.20 Fitness -0.70 Turnover 2.0% Previous run abcdef12");
  }
});

test("live progress translates counters and pending outcomes accurately", () => {
  const raw = "第1轮 完成24/100 在途3 待发73 | [变异] Demo025 已提交回测，等待结果";
  assert.equal(presentLog(raw, false).text, "Cycle 1 Completed 24/100 In flight 3 Queued 73 | [Mutation] Demo025 Backtest submitted; waiting for results");
  assert.equal(presentLog(raw, true).text, raw);
  for (const [detail, english] of [
    ["指标已收到，等待年度数据；尚未完整完成", "Metrics received; waiting for annual data; not fully completed"],
    ["提交结果未知，保留名额，只对账、不重发", "Submission outcome unknown; slot retained; reconciling without resending"],
    ["状态为完成，但指标缺失，不能展示得分", "Marked completed, but metrics are missing; scores are unavailable"],
  ]) assert.equal(presentLog(detail, false).text, english);
});

test("stages, retry limits and correlation collection retain their numbers and tone", () => {
  for (const [raw, english, tone] of [
    ["阶段[0] 平台认证中...", "Phase[0] Authenticating with the platform...", "info"],
    ["第 2 轮 阶段[2] 筛选完成，已冻结本轮 16 条回测任务", "Cycle 2 Phase[2] Selection completed: finalized 16 backtest tasks for this cycle", "info"],
    ["请求暂未成功（worldquant_poll_request_failed:TimeoutError），1.5 秒后重试", "Request unsuccessful (worldquant_poll_request_failed:TimeoutError); retrying in 1.5 seconds", "warning"],
    ["发送回测遇到429限流，冷却30秒；重试2/3，继续收取在途结果", "Backtest request rate limited (429); cooling down for 30 seconds; retry 2/3; continuing to collect in-flight results", "warning"],
    ["种子相关性：时序数据 Demo025 读取失败（HTTP 503），证据待定，600 秒后可重试", "Seed correlation: Time series for Demo025 read failed (HTTP 503); evidence pending; retry available in 600 seconds", "warning"],
    ["恢复相关性：时序数据 Demo025 获取完成（已获取 3/65）", "Recovery correlation: Time series collected for Demo025 (collected 3/65)", undefined],
  ]) {
    assert.equal(presentLog(raw, false).text, english);
    assert.equal(presentLog(raw, false).tone, tone);
    assert.equal(presentLog(raw, true).tone, tone);
  }
});

test("formal submission and queue checks preserve eligibility and deferred states", () => {
  assert.equal(presentLog("提交 2/10 Demo002 评级从 SPECTACULAR 变为 GOOD，不符合所选等级，已跳过 | 通过3 已交1 未过1 失败0 待定8", false).text,
    "Submission 2/10 Demo002 Grade changed from SPECTACULAR to GOOD; no longer matches the selected grade; skipped | Passed 3 Submitted 1 Ineligible 1 Failed 0 Pending 8");
  assert.equal(presentLog("本批入队检查 Demo001：待定，不入队；已安排延后自动复查，不阻塞回测", false).text,
    "Current batch queue check Demo001: Pending; not queued; automatic recheck scheduled; backtests are not blocked");
  assert.equal(presentLog("正式提交：本次候选 20 条（含待恢复项），成功提交 3 条后停止；检查失败跳过，原已提交的同步不计入数量。", false).text,
    "Formal submission: 20 candidates (including resumable items); stop after 3 successful submissions; skip failed checks; syncing previously submitted formulas does not count.");
});

test("platform diagnostics and unknown messages are preserved verbatim", () => {
  const original = "Invalid expression: [变异] 未通过 x > 0; $1";
  assert.equal(presentLog(`[异常] [未知] 第1轮 02/16 [探索] Demo002 回测失败（platform_error）：${original}`, false).detail,
    `Backtest failed (platform_error): ${original}`);
  assert.equal(presentLog(original, false).text, original);
  assert.equal(presentLog("未知的新日志格式：数据仍待定", false).text, "未知的新日志格式：数据仍待定");
  assert.equal(presentLog("toString", false).text, "toString");
});

test("column legend is bilingual and does not confuse checks with grades", () => {
  const raw = "结果列：状态 | 平台评级 | 轮次/序号 | 探索/SC治理/变异/反转 | Alpha ID | S=夏普 F=适应度 T=换手率；通过/未通过/待定仅表示基础检查，评级缺失显示未知，异常见行末原因";
  assert.match(presentLog(raw, false).text, /\nNote: Passed\/Failed\/Pending refer to basic checks only/);
  assert.match(presentLog(raw, true).text, /\n说明：通过\/未通过\/待定/);
});
