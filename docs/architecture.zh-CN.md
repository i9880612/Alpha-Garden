# 架构

[English](architecture.md) · **简体中文** · [返回 README](../README.zh-CN.md)

本文记录现行模块职责、主流程与数据所有权。研究规则见[研究规则](research-rules.zh-CN.md)，操作见[运维手册](operations.zh-CN.md)。

## 依赖方向

```text
alpha_garden            命令行与网页入口
    ↓
execution               用例编排、事务、调度、恢复
    ↓
generation / selection / evaluation / learning / submission
    ↓
persistence / worldquant
```

代码按业务能力平铺，不使用类型分层、全局服务定位器或第二套编排框架。`tests/` 直接镜像 `src/`。

## 模块职责

| 模块 | 负责 | 不负责 |
| --- | --- | --- |
| `alpha_garden` | 命令输入、网页接口、结果展示 | 质量判断、事务和平台协议 |
| `execution` | 初始化、运行计划、轮次调度、事务组织、外部操作顺序、本地入队检查、恢复与停止 | 复制生成、选择、学习或提交算法 |
| `generation` | 公式树、解析、规范化、类型与单位校验、想法模板、变异、组合增强 | 声称某个变异一定能提高收益 |
| `selection` | 风险筛选、探索与改善名额分配、多样性选择 | 生成公式、预测平台评分 |
| `evaluation` | 解释完整的平台结果和逐项检查、相关性计算 | 修改平台结论或决定下一轮公式 |
| `learning` | 从真实结果重算种子前沿、目标、父代预算、短板诊断和动作偏好 | 保存可漂移的模型分或活动指针 |
| `submission` | 本地资格、质量排序、检查解释、成功详情核对 | 网络读写和状态推进 |
| `persistence` | 事实存取、完整性校验、事务、显式升级 | 质量算法和运行决策 |
| `worldquant` | 认证、目录、请求、限流、响应解析 | 判断公式好坏或是否继续研究 |

## 主流程

```text
冻结账号、设置、运行边界与提交授权
→ 收尾旧运行未完成的入队检查
→ learning 重算种子、活动父代、目标和动作证据
→ generation 生成合法公式（模板、随机、变异、组合增强），selection 分配与筛选
→ 原子保存最终任务、完整设置、血缘和轮次关系
→ 按在途上限创建、轮询并采集平台结果与 PnL
→ 本轮任务全部结束后做本地入队检查；Spectacular 入队，其余合格公式继续研究或归档
→ 按冻结授权处理或跳过正式提交（提交前平台复检）
→ 结算本轮，继续下一轮或停止
```

关键边界：
- 任务所属轮次不会变化；恢复直接读取冻结任务，不重新生成和选择。
- 平台请求前先原子领取唯一提交权；结果不明时不重发。
- 详情、PnL 和年度数据分步采集，各自原子保存；部分事实不冒充完整结果。
- 自动与手动正式提交共用同一状态机；只有平台详情确认成功才写入已提交对象。

## 数据所有权

SQLite 是唯一可修改的业务事实来源；文档、日志、缓存和内存不能反向覆盖它。

| 事实 | 表 |
| --- | --- |
| 平台字段、算子及同步身份 | `platform_fields`、`platform_operators`、`platform_catalog_syncs` |
| 标准窗口和项目生成语义 | `generation_windows`、`generation_operator_roles`、`generation_operator_outputs` |
| 公式、实际设置、远端身份与任务状态 | `backtest_tasks` |
| 平台结果与逐项检查 | `backtest_results`、`backtest_checks` |
| 年度统计 | `backtest_yearly_stats_captures`、`backtest_yearly_stats` |
| 每日 PnL 曲线 | `platform_pnl_series` |
| 父子关系、种子根 | `backtest_mutations`、`signal_seeds` |
| 合格归档候选 | `qualified_alpha_archive` |
| 运行边界、授权与状态 | `automated_runs`、`automated_run_allocations` |
| 任务所属运行与轮次、轮次结算 | `automated_run_backtests`、`automated_cycle_settlements` |
| 入队检查记录（本地或平台） | `submission_checks` |
| 待提交队列 | `submission_queue` |
| 正式提交与对账状态 | `formal_submission_attempts` |
| 已确认提交的平台对象 | `platform_submitted_alphas` |

生成池、未入选公式、活动分支、目标、父代剩余额度和动作偏好都从上述事实重算，不另存副本。

`submitted-formulas.md` 是 `platform_submitted_alphas` 的展示文件，可用 `export-submitted` 重建，不参与业务判断（不进 Git）。

## 身份与证据

- 任务身份由账号范围、规范公式和完整回测设置共同决定；同一身份只恢复原任务，不重复回测。
- 只有同账号、同设置的父子结果可以直接比较。配置变化不改写已有运行和任务。
- 平台目录与项目自有语义分开：新增但没有本地语义的算子只保存为平台事实，不自动参与生成。
- 学习只使用可信的真实结果，不预测未见公式的 Sharpe、Fitness 或成功概率。
