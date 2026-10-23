# 运维手册

[English](operations.md) · **简体中文** · [返回 README](../README.zh-CN.md)

本文汇总命令行操作、目录同步、运行恢复、SC 定向研究、数据库升级、限流与存储。研究规则本身见[研究规则](research-rules.zh-CN.md)。

所有命令默认使用 `data/alpha_garden.sqlite3`、`config/` 和 `.env`，可以用 `--database`、`--settings`、`--env` 改路径。

## 命令一览

| 操作 | 命令 | 是否访问平台 |
| --- | --- | --- |
| 初始化数据库和目录 | `alpha-garden init` | 需要时同步目录 |
| 只更新算子目录 | `alpha-garden sync-operators` | 是 |
| 同步已提交公式与 PnL | `alpha-garden sync-submitted` | 是 |
| 运行 N 轮（`-1` 为持续运行） | `alpha-garden run [N]` | 是 |
| 专项优化 | `alpha-garden run -opt [N]` | 是 |
| 恢复已有运行 | `alpha-garden resume <run_id>` | 是 |
| 预览 SC 定向研究 | `alpha-garden sc-plan <父任务ID>` | 否 |
| 执行 SC 定向研究 | `alpha-garden run 3 --sc-parent <父任务ID> --sc-plan-key <指纹>` | 是 |
| 提交 | `alpha-garden submit [N]` / `alpha-garden submit <评级> [N]` | 是 |
| 重建已提交公式清单 | `alpha-garden export-submitted` | 否 |
| 研究反馈统计 | `alpha-garden research-status <run_id>` | 否 |
| 容量查看 / 清理预览 | `alpha-garden storage inspect` / `storage prune --dry-run` | 否 |
| 网页控制台 | `alpha-garden web [--read-only]` | 按操作 |

## 初始化与目录同步

- `init` 创建或核对数据库；缺少完整且匹配的数据目录时，认证并同步字段和算子。不会启动回测或提交。
- 字段每下载一页就保存到暂存区。中断后用相同参数重跑 `init` 即可接续。断点按账号、平台地址、Instrument Type、Region、Universe 和 Delay 隔离。
- 全部校验通过后才原子替换正式目录；不完整的下载不会覆盖旧目录。发现总数变化或分页异常时清除暂存，从第一页重来。
- 旧数据库第一次使用分页保存时，执行 `alpha-garden init --upgrade-catalog-storage`。
- 算子权限或平台算子列表变化时，执行 `sync-operators` 只更新算子。有活动运行或未决提交时会拒绝切换目录。

## 已提交公式基线

- `sync-submitted` 分页读取账号下所有已提交公式（包括隐藏和已停用的），保存详情和 PnL。只写已提交事实，不导入种子、不提交公式。
- **未完成全量同步的账号不能启动或恢复研究**；确认零提交的新账号可以直接冷启动。
- 曲线暂缺时保留已完成部分，再次执行可接续。
- 列表前后不一致、缺页或身份冲突会明确失败。
- 保持 `WQB_ACCOUNT_SCOPE` 与实际账号一一对应，切换账号不要复用。

## 运行、暂停与恢复

- `run` 每次创建新批次，并自动收尾已停止的旧批次；继续原批次用 `resume`，会沿用原模式和冻结的设置。
- **正式提交默认关闭**，只有显式加 `--auto-submit` 才开启；网页启动的运行不会自动提交。
- 网页上的暂停会让引擎在安全边界停下。关闭浏览器不会停止运行。
- 恢复直接读取冻结的任务，不重新生成；已经发出的请求结果不明时不会重发。
- 新设置只用于之后的新运行。

## 提交

- `submit 2`：从 Spectacular 队列成功提交 2 条。
- `submit good 2`：从合格归档中选 Good 级别、当前满足相关性条件的公式，成功提交 2 条。也支持 `inferior`、`average`、`excellent`、`spectacular`。
- 数量按**成功提交条数**计算：复检不通过的跳过，候选不足时提前结束，结果不明时先确认已有请求再继续。
- 网页「全部评级（从高到低）」模式按质量顺序逐条提交，规则见[研究规则 §7](research-rules.zh-CN.md#7-入队检查与提交)。
- 每次正式提交前都会读取平台详情并做最新检查。
- 平台检查返回的 SC 数值会和本地算出的值对账，写一行"SC 对账"日志；两者相差超过 0.05 时以警告提示，不影响提交流程。

## SC 定向研究

用于仅 SC 失败的父代，比较三类降相关方向（见[研究规则 §6](research-rules.zh-CN.md#6-自相关性sc治理)）：

1. `sc-plan <父任务ID>` 只读预览六条候选，给出计划指纹。
2. `run 3 --sc-parent <父任务ID> --sc-plan-key <指纹>` 执行：每批两条，最多三批，共用父代 20 次额度。本地事实在预览后变化时会拒绝启动。

要求：
- 父代仅 SC 失败、阻挡证据完整、至少剩六次机会，账号已完成全量基线同步。
- 正式提交关闭；首次发送前核对平台算子可用性，缺失时停止新请求，不消耗额度。

检查重试：
- 每条任务先短重试最多 3 次，再延期补采最多 3 次，间隔至少 600 秒。
- 耗尽后保留待定，不改成通过或失败。

## 限流与故障

- 认证失败，或 429 耗尽有界重试时，立即停止后续请求；按输出的 `Retry-After` 等待后再执行。
- 正式检查遇到 429 时至少等 60 秒，日志显示"入队检查被平台限速"，不影响回测结果。
- 每次采集 PnL 后至少间隔 2 秒再发下一条请求；429 时整个运行一起退避，重启后仍遵守已保存的冷却时间。2 秒是本地保守值，不保证不触发 429。
- 平台有时把用了"别名字段"的公式归入一条已有 Alpha（返回的公式是另一种写法），这类任务会直接记为 `platform_duplicate_alpha`，不再反复读取。
- 运行配置中的 `maxConsecutiveFailures`、`maxRequestFailures`、`maxPendingSeconds` 控制故障边界。

## 数据库升级

运行命令**不会自动迁移**数据库。旧库需要经明确授权后，在同一个 `BEGIN IMMEDIATE` 事务内按顺序执行 `persistence.schema` 中的升级函数，出错整体回滚：

1. `add_optimization_submission_source`（仍使用较早的提交来源约束时）
2. `add_mutation_reference_storage`
3. `add_sc_research_plan_storage`
4. `add_submitted_sync_storage`（也可以退出事务后直接执行 `sync-submitted` 新建）
5. `add_quality_research_storage`

这些升级只新增表或列，不修改历史任务、额度和结果。新的空数据库直接执行 `init`。

## 研究反馈

`research-status <run_id>` 只读统计本次运行按探索、普通变异、SC 修复和质量研究区分的冻结、发送、合格、拒绝、待定与请求错误数量，并列出最近 30 天的质量动作结果。统计的是已冻结任务，不包含生成后未送测的公式。

## 本地容量与清理

```bash
alpha-garden storage inspect --daily-curves 1000
alpha-garden storage prune --dry-run --retention-days 90
```

- 两条命令都是只读的：不读账号、不访问平台、不迁移数据库。
- `inspect` 分别列出数据库文件、WAL、空闲页和 PnL 大小，并按每天新增曲线数估算增长。
- `prune` **目前只有预览**：保留期默认 90 天、最少 30 天；只列出旧的、非 SC 检查明确失败且没有任何保护依赖的曲线。已提交参照、谱系、种子、未完成运行和提交相关记录都受保护。
- 正式删除、清理标记和 `VACUUM` 维护命令尚未实现。

## 日志

网页服务每次操作最多保留 500 条日志，重启后清空，不影响已保存的研究结果。PnL 采集以 `[PnL采集]` 行单独显示。
