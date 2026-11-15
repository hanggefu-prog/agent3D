# AgenticBench 局部与全局智能体 harness 设计报告

核验日期：2026 年 10 月 4 日。本报告解释“智能体可以做什么”和“可信系统如何证明结果”之间的边界。当前代码同时保留局部函数搜索、层次化局部组合、通用全局工作区框架及实际 workspace baseline；它们不是同一种执行循环。新版四 case 已有 one OPT timing 的 reference authority，但局部 native hook 和新版 thermal 搜索仍有资格缺口，详见 [对齐状态](../../docs/REFERENCE_AND_INTERFACE_STATUS.md)。

## 系统分工

Open3DBench 拥有物理算法、输入、PDK、原生工具和 stage artifacts；AgenticBench 拥有 task、prompt、角色编排、候选隔离、reference registry 和评分适配。两者通过显式路径、CLI 与带哈希的 artifacts 交互，不直接 import 物理 flow 的 Python 内部模块。原生工具只使用经过核验的容器安装版本，不能让 host 上另一个 OpenROAD 悄悄成为实际求值器。

任务和环境声明是实验身份的一部分。仅配置同一个 case 名不足以比较：source、输入、镜像、安装二进制、任务、评分、种子、线程、阶段边界都需一致。物理 reference 从干净冻结源码产生；开发工作区的日志改动和本文档不改变已有运行的归属。

## 局部函数搜索循环

核心实现见 [search_api.py](../local_bench/search_api.py)、[program_bridge.py](../local_bench/program_bridge.py)、[evaluation.py](../local_bench/evaluation.py) 和 [workspace.py](../local_bench/workspace.py)。搜索方法只负责提案和种群状态；benchmark 负责程序物化、限制验证、独立运行、指标及持久化。

```text
TaskSpec + frozen input + qualified local reference
  → SearchProblem / SearchContext
  → backend.ask 返回 CandidateProposal
  → AST 与输出格式验证 → trusted program adapter → patch scope 验证
  → 私有候选工作区 + 已验证输入前缀 + 必要私有构建
  → 物理 stage 执行 → 原生 artifact 验证 → 可信指标
  → CandidateFeedback → durable history / best → backend.tell
```

`CandidateProposal` 包含 proposal ID、完整程序和 metadata；`CandidateFeedback` 包含程序 SHA、status、score、metrics、artifacts、错误与运行时。`accepted/cached` 且有 score 表示该反馈可作为有效评分，不表示一定优于 reference。搜索后端可以是 EvoStage、OpenEvolve、LLM4AD adapter 或自定义 ask/tell，实现不应自行定义另一套物理奖励。

### 局部安全边界

TaskSpec 的 allowed paths、config value constraints、patch limits 与程序 AST 规则共同生效。prompt 只是解释，不能代替 verifier。模型给出的路径、日志和代码都不是更高优先级指令；程序未通过限制就不派发原生计算。

受限 GRT 函数先由可信代码翻译 C++；DREAMPlace hook 由可信 adapter 接入固定位置。native 策略需要私有构建与运行时证明；共享 compiler cache 只能复用编译对象，不等于复用测量，也不能共享可变候选源码。输入前缀复制必须核所有 stage 签名与 artifacts，不使用任意旧输出作为 checkpoint。

局部 authority 必须来自真实的同任务边界，包括 hook 接入/可达证据及正确 reference。普通 reference 的 placement 或 GRT 成功只证明物理 flow 可用，不能替代这里的资格。当前 `reference_20261004` 的六项局部接口通过离线边界测试，但四个 case 的 local scoring authority 仍为空。

### 局部反馈与预算

按任务配置拼装 system instruction、task contract、源码摘要、初始程序及近期/高分候选。模型每轮应说明一个机制假设，反馈由真实指标而非自述构成；执行失败、verifier 拒绝和有效但无改善分别保存。生成候选的失败仍按既定协议占槽，不因没有 score 自动补轮。

`ProgramEvaluator` 有显式缓存能力，缓存命中须记录 `cached` 和来源，不能算一次新原生执行。本次不改变既有缓存策略；campaign 若禁用就不能为了效率擅自启用。reference/probe、build、native eval、模型调用和 candidate finals 分开计账，搜索循环的预算上限不等于 native 进程次数。

## 层次化局部组合模式

[hierarchical.py](../agentic3d/hierarchical.py) 是另一种组合器：Global Analyzer 把总任务拆成既有 bounded local task，局部 ask/tell 得到可接受修正，确定性 outer ask 合并修正，再交全流程 evaluator 做一次独立评价。局部代理指标改善并不保证总 STA/PPA 改善。

老 stage orchestrator 还存在 Global Analyzer → Local Analyzer → Scope Compiler → Task Agent → Local audit → Global adjudication 的流程。这里 Scope Compiler 对 adapter 硬边界与模型 scope proposal 求交，不允许 analyzer 增加权限。它与下节最新 workspace Agentic3D 的“target stages 仅作假设聚焦”不同，不能将两者的权限解释混写。

这些通用能力可独立复用，但当前新版四 case workspace timing 驱动没有暗中启用 cheap proxy 搜索、知识库 v2 或局部任务群。不能依据 FRAMEWORK 中的设计库存推断某一场正式实验实际执行了所有模块。

## 全局工作区公共合同

实现入口为 [global_bench](../global_bench) 与 [workspace_codesign_evaluator.py](../scripts/workspace_codesign_evaluator.py)。`GlobalTaskSpec` 声明 source、环境、只读 inputs、submission、verifier、objective、seed policy、stage DAG 和预算；`GlobalSearchContext` 提供可信历史和剩余预算；`Intervention` 表达假设、目标阶段和评价需求。

全局候选不必编码成一段固定函数。acting agent 在私有物理源码工作区内计划、编辑和做小规模测试；可信 harness 快照实际文件差异后独立评价。通用 Trial API 与最新 workspace driver 的具体 checkout 方式不同：后者使用不共享 Git objects 的独立 clone，并只读挂载 Git 元数据，不把原始物理 reference checkout 暴露为可写目录。

“全局”表示前端算法空间不局限于某个函数，不表示可修改整个评分系统。最新 timing task 保护 backend、flow orchestrator、tests、AgenticBench、输入与评估工具；它允许 partition/placement Python 及源码内配置变化，拒绝没有匹配运行镜像的 native 编译源修改。完整 scope 与目标见 [任务设定报告](TASK_SETTINGS_REPORT.md)。

## 当前单 agent workspace baseline

[run_workspace_codesign.py](../scripts/run_workspace_codesign.py) 使用同一持久会话，依次读取可信 reference/history、在 scratch 写明确计划、修改候选工作区，然后返回变更和测试摘要。最终评分不由 agent 自己触发或填写，而由 host 的独立 evaluator 每轮执行一次。

每轮保留可继续改进的工作区，但 reference、task 和 evaluator 不随会话修改。最终回复不是 submission 内容的唯一来源：harness 检查真实文件差异、保护路径、配置、种子和源码快照。模型声称“编译通过”“已改善”都必须在可信执行证据中找到对应记录。

当前 workspace 模型入口 [workspace_codex.py](../global_bench/workspace_codex.py) 固定 `gpt-5.6-sol`，认证留在 host 客户端，agent 的工具只进入隔离容器。它不是通用任意供应商代理；局部 DeepSeek 配置不能自动改变该全局 baseline。本文不修改模型/凭据，也不声称旧用户提供的 API 已被验证。

## 当前 Agentic3D workspace baseline

[run_workspace_agentic3d.py](../scripts/run_workspace_agentic3d.py) 用代码固定每轮的五个 phase，三种智能体角色不会自由递归委派其他 agent。

```text
Global plan → Local pre audit → Task plan and edit
    → trusted final evaluation → Local post audit → Global adjudication
```

| Phase | 权限 | 输出与验收 |
| --- | --- | --- |
| global plan | 只读源码、输入、历史；scratch 可写 | 一条可证伪前端假设、目标阶段、指导与证据 |
| local pre | 同上，不运行 EDA | 细化同一假设，不能扩大 Global 目标阶段；recommend 或明确缺证据 |
| task | 仅私有允许的前端工作区可写 | 同一责任主体先写 plan 再编辑；不得委派计划或伪造 final |
| local post | 只读真实 diff 与当前 final | 检验机制与证据，必须引用当前 round 产物 |
| global adjudication | 只读 Local post 和 final | 对 immutable task 判 accept/reject/need evidence |

analyzer 输出固定 schema JSON，Task Agent 提交真实文件修改和文本摘要。Global/Local 不得靠修改算法或运行 EDA验证自己的推断；其声明也不能覆盖 verifier。若 Local pre 扩大目标范围、角色记录不合 schema、运行发生明确基础设施故障或需要证据，驱动进入 needs attention，不静默补跑 final。

这里有两个不同“最好”状态：

- `best.json` 是独立 evaluator 已验证的最高分，真实改善即保留，即使角色审查还未通过。
- `incumbent.json` 是经过 Local 和 Global 都接受的候选；接受条件是可信严格改善 AND Local accept AND Global accept。

比较使用 `prior_best_verified`，而不是已包含当前候选的 best，避免候选被要求“严格优于自己”。analyzer 拒绝不应删除已验证的最好物理结果；needs evidence 也不返还已消费 final。

## 隔离与可信求值

实际 agent 容器使用只读 root filesystem、`network=none`、drop ALL capabilities、no new privileges、固定 CPU/memory/GPU/PID；不暴露 Docker socket。task、benchmark、reference、results 与 Git 只读挂载；analyzer 的整个 subject 也只读，仅 scratch 可写。任务环境 YAML 的 restricted network 在该 driver 中进一步落实为 agent 容器离线，模型通信发生在 host 侧，不意味着复制凭据到容器。

evaluator 在 agent 之外检查 reference authority、task digest、源码变化和配置，仅允许两个前端域合入冻结 config。参考 input/floorplan 前缀按签名和哈希复用；其余受影响后缀由可信脚本运行。未识别变更按 fail closed 或保守全影响处理，不能让模型自报“只改 placement”来绕过 partition 依赖。snapshot 前后变化不一致也会拒绝。

通用 [ImpactResolver](../global_bench/impact.py) 根据真实变更路径与 stage DAG 计算影响，声明目标阶段只能扩大检查，不能缩小实际影响；stage signature 还绑定上游及配置。最新 workspace evaluator 有其专用前端复用验证，不能把通用类支持的所有 replay 选项当成本驱动已实现的执行分支。

最新 family 的物理生产 reference 到 GRT，但 timing evaluator 只评价一次 OPT 的 pre CTS STA。派发成功、容器退出零码、`run.json` 中某个状态以及模型输出都不能单独代替 stage manifest、原生身份和指标一致性验收。thermal 诊断也不会进入该 reward。

## 状态证据与恢复

一场可复现实验至少保留 run config、任务摘要、reference receipt/authority、输入摘要、模型/provider 配置（无密钥）、实际 prompt/session、候选原文和 SHA、plan/roles、source diff、构建和容器回执、evaluation/verification、history、best、incumbent 与 summary。具体文件命名随 local 和 workspace driver 不同，不能因为找不到另一 driver 的文件名就把数值置零。

| 层次 | 主要检查 | 不能推出的结论 |
| --- | --- | --- |
| model turn | role/session 起止、provider 请求和返回 | turn walltime 不等于 API 纯延迟 |
| candidate | 程序或 diff、verifier、槽计数 | scratch 诊断不是 final |
| build | 私有构建、缓存命中、安装二进制 SHA | 编译通过不等于物理成功 |
| dispatch | command/container ID 与 terminal | client dispatch 不等于 native started |
| native | stage log、manifest、文件 size/SHA、实际指标 | GRT 完成不等于 DRT 或 signoff |
| acceptance | 独立分数、prior best、角色审查 | accepted feedback 不一定改善；高分 best 不一定已成为 incumbent |

恢复先读原状态并核身份、已消费预算、终态与 in flight；存在 unknown job/controller 就只读诊断。相同 frozen source、task、prompt、输入、reference 和协议不能省略；不根据 GPU 空闲猜测可以扩大并发。多个 controller 必须有共同锁或明确终态屏障，单个任务的 concurrent limit 不提供跨 controller 安全性。本次未重启历史 evolution 或 DRT 队列。

retention 只按既定协议在终态后处理：正式 reference、best 和失败证据保留；qualification full、formal best 的选择属于运行配置。本次仓库整理不触发清理或缓存变化。

## 验证覆盖和当前缺口

本次五组最新接口/仓库边界回归为 65 tests 和 22 subtests 通过；另七组任务选择、workspace 角色隔离、计数、ask/tell 和程序安全回归共 126 tests 通过。真实 wheel 隔离安装测试也通过，覆盖四个新版 profile 的可信 JSON 资源。四份核心报告/规范的 63 个本地链接均有效。它们证明接口、文档和打包行为，不证明新增原生求值，也不代表全仓库测试套件全部通过。两服务器 authority 哈希已只读复核一致。

尚需单独验收的是：新版局部 hook 的实际接入和 scoring authority；新版 thermal 的可信任务/评分接口；若改变全局模型供应商则还需对应客户端、认证与工具隔离资格。通用 API、历史任务和设计文档不能替代这些证据。后续新增能力应在新版本显式登记，保留旧实验身份，不回填旧结果。
