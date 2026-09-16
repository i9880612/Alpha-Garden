export type ConsoleLocale = "zh-CN" | "en-US";
export const defaultConsoleLocale: ConsoleLocale = "zh-CN";
export const isConsoleLocale = (value: string | null): value is ConsoleLocale => value === "zh-CN" || value === "en-US";

export const consoleMessages = {
  "zh-CN": {
    header: { language: "语言", lightTheme: "切换浅色模式", darkTheme: "切换深色模式", systemSettings: "运行设置", userName: "本地控制台", role: "研究控制台" },
    home: {
      metrics: ["今日完成回测", "待提交公式", "合格归档", "已提交公式"],
      trend: "回测趋势", distribution: "已提交评级分布", research: "研究概况", currentRun: "批次进度", recent: "最近回测结果",
      countUnit: "条",
      period: "近 15 天",
      modules: [
        { title: "运行中心", desc: "查看批次进度与运行日志", path: "/runs", tone: "blue" },
        { title: "公式库", desc: "查看评级、谱系与剩余额度", path: "/formulas", tone: "cyan" },
        { title: "提交管理", desc: "按评级和成功数量提交", path: "/submissions", tone: "yellow" },
        { title: "运行设置", desc: "自定义回测参数、研究预算与运行边界", path: "/settings", tone: "purple" },
      ],
    },
  },
  "en-US": {
    header: { language: "Language", lightTheme: "Switch to light mode", darkTheme: "Switch to dark mode", systemSettings: "Run settings", userName: "Local console", role: "Research console" },
    home: {
      metrics: ["Backtests completed today", "Submission queue", "Qualified archive", "Submitted Alphas"],
      trend: "Backtest activity", distribution: "Submitted grades", research: "Research overview", currentRun: "Batch progress", recent: "Recent backtest results",
      countUnit: "",
      period: "Last 15 days",
      modules: [
        { title: "Runs", desc: "Follow batch progress and run logs", path: "/runs", tone: "blue" },
        { title: "Formulas", desc: "Inspect grades, lineage and remaining attempts", path: "/formulas", tone: "cyan" },
        { title: "Submissions", desc: "Submit by grade and success count", path: "/submissions", tone: "yellow" },
        { title: "Settings", desc: "Customize backtests, research budgets and run limits", path: "/settings", tone: "purple" },
      ],
    },
  },
};
