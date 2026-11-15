# AgenticBench 任务设定报告

核验日期：2026 年 10 月 4 日。本报告面向四个 LoL ng45 case：`bp_fe`、`bp`、`ariane133`、`swerv_wrapper`，说明 prompt、修改范围、目标、约束及证据阅读方式。它描述已有实现，不启动新实验，也不把兼容任务自动迁移为新版任务。资格状态以 [对齐状态报告](../../docs/REFERENCE_AND_INTERFACE_STATUS.md)和 [reference 索引](../../outputs/reference-alignment-20261004/reference-index.json)为准。

## 2026-10-16 native HBT update

HBT now has a connected native evaluator, frozen five-stage input reuse, bounded
policy execution and independent geometry/STA verification. The current contract
and qualification scope are in [HBT assignment](HBT_ASSIGNMENT_TASK.md). The
2026-10-06 definition-only notes below describe historical status.

## 任务定义与 prompt 来源

### 2026-10-06 旧局部任务代码清理

当前仅发布下文列出的八个局部任务。11 个旧任务的定义、prompt、初始程序和专用 adapter 已删除；新任务依赖的校验与转换逻辑提取到任务无关的共享模块。带日期的旧章节仅保留为设计历史，不再代表旧目录可执行。历史实验、reference 数据、全局 Agentic3D 任务及 Open3DBench 的 spatial-bin 物理算法接口未删除。本次是本地源码清理，不表示服务器已更新或新原生资格已通过。

### 2026-10-06 切分基线切回 global 2-way，时序 fitness 不变

当前核心切分任务为 [`tp_gnn_post_placement_timing`](../local_tasks/tp_gnn_post_placement_timing/task.yaml)，使用 `tp_gnn_fast.json` 的 global-2way decoder；prompt、system prompt、默认入口、运行前配置检查和原生结果校验已一致。仍只演化 timing-context-v2 loss，执行 partition → placement → PDN → HBT → 联合 STA，**无 OPT**，fitness 保持 0.6 WNS/0.4 TNS 亏损成本倒数，不退回 cut fitness。旧 `tp_gnn_bin_timing` 和 cut-score 任务作为独立历史变体保留，不重命名结果、不继承其 reference 资格；bin 算法与接口仍可显式调用。详见 [基线切换说明](PARTITION_GLOBAL_2WAY_TIMING.md)。

### 2026-10-06 当前清单：HBT 替换 net ordering

按用户决定，当前八个局部任务调整为：TP-GNN learning objective、placement initialization、placement preconditioner、dynamic Nesterov control、HBT assignment policy、routing rip-up、maze cost、iteration controller。net ordering 从默认清单和 Agentic3D 自动建模模板中退出，旧实现与历史结果保留。统一 partition reference 仍为 global-2way；此前 bin-based 方案作为显式研究变体保留，不覆盖 reference。

新增 HBT 任务按照“背景→优化问题→评估→接口与约束”的自然英文组织，演化网络优先级与合法站点代价，采用 HBT 后、无 OPT 的联合 STA（0.6 WNS/0.4 TNS 归一化亏损成本）。仅向模型提供明确的几何输入和可追溯的评估反馈，不虚构逐网时序或拥塞信息。prompt/system prompt、修改边界、评分和日志口径见 [HBT 任务定义](HBT_ASSIGNMENT_TASK.md)。当前为 definition-only，离线接口通过不代表已有原生 hook 或专用 reference；正式执行入口已阻止误启动。

以下带日期的章节保留原有修订历史，不代表上述已退出任务仍在当前清单。

### 2026-10-06 新切分与布局任务（原生资格待补齐）

本次新增 `tp_gnn_bin_timing`：reference 选择宏先聚类、真实宏/bin 覆盖负载、标准单元分 bin 聚类的 spatial-bin decoder；loss 仍是 timing-context-v2 接口。候选执行 partition → placement → PDN → HBT → 统一两 die 的 placement-estimated STA，明确不执行 OPT、CTS、GRT、DRT 或 thermal。fitness 改为 WNS/TNS 亏损的 0.6/0.4 加权归一化成本倒数，cut、loss 和 embedding 仅作诊断。

原联合源码任务拆为 `placement_initialization` 和 `placement_preconditioner`。前者只改 `initialize_position`/BasicPlace，后者只改 `PreconditionOp`/PlaceObj；另一部件始终是原生代码，不继承另一任务最优解。两者沿用 Nesterov、4000 次迭代及 0.9 HPWL/0.1 overflow 权重，但 overflow 奖励在既有停止阈值 0.01 饱和，避免趋零时支配总分。动态 Nesterov 任务与四个布线任务本轮不变。

实现、测试和待决策项见 [迁移报告](TASK_REVISION_20261006.md)。下文保留历史任务记录；旧 cut/联合源码结果不能当作这三个新任务的 reference。接口预检不等于 GPT 或物理运行已通过。

### 2026-10-05 当前修订：统一七任务的模型上下文与反馈

今天的 guidance 来自用户提出的优化语义优先原则、粘贴评审对搜索策略与反馈解释的建议，以及随后对写作方式的要求：先背景、再具体问题、最后技术细节；使用自然英文；专业缩写首次展开；不用刻意的 causal/versus-frozen 表述。落地时以当前 task、adapter、原生日志生产代码和评分函数为准，不把建议中的潜在信号写成已经存在的接口。

两个布局任务和四个布线任务已沿用 TP-GNN 的组织方式，修改各自 `prompt.md`、`config.yaml` 中的 system prompt，以及 OpenEvolve 每轮的生成/示例模板。保留函数接口、真实输入语义、可修改组件、硬限制、指标方向和数值建议；路径、版本历史、manifest、容器及资格工程信息继续由宿主合同维护，不再拼进模型问题说明。选择设计及搜索相关固定设置仍由实际配置绑定。完整程序输出不等于每轮全部重写；鼓励一次主要改动，也允许组合经可比结果支持的改进。

| 任务 | 模型可用的评估反馈 |
| --- | --- |
| Nesterov 动态步长 | 最终加权 HPWL、最差/分 die overflow、最大密度、评分分项；控制调用/阶段段数、identity 状态和乘数统计。原生步长提示不是最终接受步长。 |
| 初始化与预条件 | 相同最终布局指标；当前没有单独的初始化质量、梯度条件或收敛轨迹诊断，不补造。 |
| Net ordering | 分 die/总 GRT 线长、overflow、拥塞 bin；排序改变、溢出子集改变、top-10% 重合率及分数跨度等。 |
| Rip-up | 同上物理指标；观测到的累计调用、决策变化、选中数量/比例和输出分数样本。 |
| Maze cost | 同上物理指标；观测到的累计调用、乘数变化及已记录乘数 min/max/last。 |
| Iteration controller | 同上物理指标；按日志记录数统计调用、按每次布尔标记统计变化，另外保留可重置的迭代编号和实际控制量。 |

Rip-up/maze 日志是稀疏的，其累计计数只是观测下界，不能冒充最终精确次数；样本极值不是每次调用的全量极值。通用 `signal` 历史别名已去掉，缺失可选观测不再补零；真实零保留。原始回执和历史文件不重写。筛选后的相同反馈供 ask/tell 后端与原生 OpenEvolve 导出使用；未改变评分公式、物理 hook、预算、验证器或 reference 准入，未启动或部署新实验。目录名和历史 task ID 不代表资格状态。

### 较早的 2026-10-05 修订记录

以下为早先版本的记录；其中将完整配置/路径/SHA 拼进模型 prompt 的方式，已由上面的当前修订替代。历史资格描述不作为新的准入依据。

2026-10-05 修订：已修复 TP-GNN 误用布局探索主题、net ordering 表格语法文字与验证器矛盾、reference 任务正文硬编码旧输入版本、零 reference 评分边界遗漏，以及 iteration controller 输入范围表述不准确的问题。生成的 prompt 现在展示解析后的配置路径、配置 SHA、物理 recipe 和真实评估边界；历史 task ID 不代表输入家族，选择 profile 也不等于取得 scoring authority。TP-GNN 补充张量形状/类型、标量与 Torch 运算区别、空 mask/单样本归约风险及 loss 仅为最终 cut 代理目标的说明，并扩展源码摘录至真实训练调用。两个布局任务明确加权奖励允许指标间取舍，且不证明合法化。上述修订不改变评分公式、算法、训练/候选预算、种子、物理设置或资格门；不启动实验，不覆盖历史 prompt/结果，也不把 CPU 回归通过当原生资格。

同时将上述七个局部任务统一为完整程序输出（`diff_based_evolution: false`），并在 OpenEvolve 导出端强制遵守 task-adapter 的完整程序协议，消除“prompt 要完整函数、后端却要求 diff”的冲突。EvoStage 与 smoke 驱动的模型上下文使用调用者实际选择的配置/profile，不再误报模板默认的 bp_fe；三个路由策略补齐真实输入归一化公式。net ordering 的奖励仍为加权收益比，另外三个路由策略仍为加权归一化成本的倒数，二者没有被混同。

本地验证：针对 prompt、导出、评分、作用域、语言翻译及最新 reference 接口的扩展回归得到 351 passed、1 skipped、49 subtests passed；另有 4 个历史 20260923 测试因本地缺少 `Place-3D/config/partition/tp_gnn_reference_20260923.json` 未通过，未伪造该历史文件或改用新版配置冒充旧合同。新增检查覆盖七个完整 prompt/导出、四 case 新旧 profile 的 24 种绑定及 net ordering 表格策略的 C++/Python 结果一致性。它们不证明原生模型/EDA 运行资格；最新家族 net ordering 排除及其他 local authority 缺口仍按原规则处理。

一个局部任务由 `local_tasks/<目录>/task.yaml`、`prompt.md`、`config.yaml`、`initial_program.py`、`program_adapter.py`（适用时）及 verifier 共同定义。`prompt.md` 不是模型收到的全部内容：实际 system prompt 是 [prompting.py](../local_bench/prompting.py) 的公共约束加任务 `config.yaml` 中的 `prompt.system_message`；用户上下文再加入 task contract、只读源码摘录、初始程序和历史反馈。源码阅读范围不等于修改范围。

公共 prompt 要求：只依据真实黑盒评估提出一个可证伪假设，遵守输出格式、参数和源码边界，不编造字段或结果。默认可拼入近期及高分反馈，具体数量和长度由各 task 的 `prompt_context` 冻结；三个 reference 局部主任务均为近期 3 条、高分 3 条，partition 反馈上限 24,000 字符、两个 placement 任务各 16,000 字符。运行时实际 prompt、模型配置、候选程序和 SHA 才是复现实验的精确记录；本报告中的中文解释不替换英文原始 prompt。

全局任务由 `global_tasks/<目录>/task.yaml` 与 `instruction.md` 定义，驱动再拼入当轮参考指标、历史、预算及计划要求。全局 agent 输出的是实际工作区修改，不是固定局部函数。应同时保存 instruction、task digest、驱动版本、每个角色实际消息和隔离回执。

## 最新家族的共同设置

可信配置见 [partition.json](../local_bench/reference_configs/partition.json)、[placement.json](../local_bench/reference_configs/placement.json)，四个配置目录命名为 `local_bench/benchmark_inputs/CASE_lol_ng45_reference_20261004`。

| 项目 | reference 固定值与语义 |
| --- | --- |
| 物理版本 | flow `4d882f0b`、OpenROAD `56ea5f41`、OpenSTA `5d8d515e`，完整身份见对齐报告 |
| 物理资源 | 4 个物理线程；CPU quota 与物理线程数分开记录 |
| 种子 | I/O 1、partition 1001、placement 1000、routing -1 |
| TP GNN | flat、global 2way decoder、partition last timing、CUDA、embedding 128、最多 20 epochs、最低 10 epochs、最低 100 optimizer steps；其余值以可信 JSON 为准 |
| Placement | Concurrent DREAMPlace、Nesterov、4,000 iterations、初始步长 0.01、512 bins、density 0.60、overflow stop 0.01、density weight 0.00003、cross die net weight 1、每侧 halo 10 µm |
| 后端 | track aware greedy HBT；一次 OPT；CTS target skew 0.1 ns，hold repair 关闭 |
| GRT | congestion iterations 4、resource reduction 0、固定种子；本次不执行 DRT |
| 输入来源 | 已验证 manifest 和逐文件哈希；新版 timing input 属于 partition 目录，无独立 timing stage |
| reference 策略 | 使用精确匹配的 ADOPT；不静默 FRESH、EXTEND 或换家族 |

这里的固定 reference 设置不等于“全局候选的每个前端参数都不可动”：局部任务只暴露其特定函数；全局 timing 可改 partition 和 placement 算法/配置，但不得改种子、输入、后端、评分、时钟等受保护边界。各自 task contract 决定可编辑面。

## 局部 TP GNN 学习目标

实际任务：[task.yaml](../local_tasks/tp_gnn_learning_objective_reference_20260923/task.yaml)，[prompt.md](../local_tasks/tp_gnn_learning_objective_reference_20260923/prompt.md)。目录保留历史日期是为了保持任务身份；选用新版时必须另外显式绑定 `reference_20261004` 的 case profile，不能使用默认旧输入后声称新版已对齐。

模型只返回完整的 `compute_tp_gnn_loss(...)` 函数，可信 adapter 仅替换 `Place-3D/src/partition/lol/methods/tp_gnn.py` 的对应定义。允许 1 个修改文件，patch 最多增加 220 行、删除 12 行；不得修改图构建、采样、训练预算、聚类、解码、平衡修复或评分实现。冻结 timing 特征、权重和连接关系来自 `20_partition/timing_input`，必须保留 `full_coverage_partition_last` 与 `paper_delay_slack_hyperedge_v1` 合同。

最终目标不是训练 loss 最低，而是合法完整划分的 timing weighted cut 更小。正 reference 下分数为 `reference_weighted_cut / max(candidate_weighted_cut, 1e-12)`，乘硬约束通过标记；reference 为零时，仅 candidate 也为零得 1，否则为 0。指标校验要求 assignment 完整、权重总量匹配、cut ratio 一致；area balance error 不超过 0.06，宏数量和宏面积偏离半分不超过 0.05。配置中的 balance tolerance 0.03 与指标 `area_balance_error` 的定义不同，不能直接把两者数值互换。

日志阅读先区分图加载、PlaceDB 名称映射、embedding training、聚类和平衡修复。重点对比 `partition_metrics.json`、method report、raw 与 repaired assignment 的 cut 变化、训练步数及时间；“loss 下降”或输出一个 assignment 文件不足以证明正式改善。这项新版输入/接口已适配，但当前四 case 没有 local scoring authority。

## 局部布局初始化与预条件器

实际任务：[task.yaml](../local_tasks/placement_init_preconditioner_reference_20260923/task.yaml)，[prompt.md](../local_tasks/placement_init_preconditioner_reference_20260923/prompt.md)。一个原子候选同时包含完整 `initialize_position(params, placedb, die_contexts=None)` 和 `PreconditionOp` 类；类接口为 `__init__`、`set_overflow`、`__call__`。可信 adapter 只修改 DREAMPlace 的 `BasicPlace.py` 与 `PlaceObj.py` 对应定义，不允许模型自行修改其他优化流程。

`die_contexts` 提供不可变的 `(die_name, region_id, movable_utilization, target_density, core_aspect_ratio)`；候选可共享机制，也可依据物理上下文做分 die 响应。必须保留 fixed/stopped mask、dtype/device、filler 和 fence 语义，不改随机种子，不做 I/O、网络、进程启动或动态代码。最多 2 文件、增加 800 行、删除 400 行，每个受控定义 AST 上限 3,000，程序字节上限 96 KiB；大程序被 verifier 拒绝不构成补轮理由。

目标取最后一行完整的原生 GP 日志：`score = 0.9 × R(HPWL) + 0.1 × R(max_die_overflow)`，其中正 reference 下 `R(x)=x_ref/max(x,1e-12)`；零 reference 下仅双零得 1。HPWL 是 DREAMPlace 原生 weighted HPWL，不可替换成后端线长；overflow 是两 die 中较大的密度 overflow，不是 GRT congestion。运行时长和最大密度是诊断，不偷偷加入奖励。GP 成功不等于宏/单元合法化或全流程成功。

分析日志时核候选确实进入初始位置/预条件路径，再看两 die 曲线、最后同一行 HPWL/overflow 和异常；不能选更早的低 HPWL 行配最后的低 overflow 行。现有普通 placement reference 不自动成为该可修改函数边界的评分 authority。

## 局部 Nesterov 动态控制

实际任务：[task.yaml](../local_tasks/placement_parameters_reference_20260923/task.yaml)，[prompt.md](../local_tasks/placement_parameters_reference_20260923/prompt.md)。历史名称含 parameters，但它不是任意 JSON 调参任务。候选完整定义 `adjust_learning_rate(...)`，输入包括 native step hint、步数、前后 log HPWL、两 die overflow、log lambda、梯度范数及历史 multiplier，输出 `(shared_multiplier, (bottom_multiplier, top_multiplier))`。

修改范围仅为可信 adapter 在 `NesterovAcceleratedGradientOptimizer.py` 与 `NonLinearPlace.py` 的受控接入。输出作用于 acceleration 前的梯度；不是绝对 learning rate，也不是任意缩放 momentum。native no BB 路径、回溯上限 10、停止条件及 4,000 步预算不变。共享 multiplier 必须有限且位于 `[0.0001,100]`；die multiplier 为有限正数，可信代码施加 ratio `[0.25,4]`、几何均值 1 和 log smoothing 0.2。

奖励与上一节相同。每个真实 optimizer step 调一次，不把回溯试探当多个策略步骤；`NesterovStepControl` 轨迹必须证明 hook 活跃，即使 identity 候选也应有接入证据。先核 hook 调用和应用值，再解释 HPWL/overflow 变化；native step hint 不能叫 Adam learning rate。本任务不能复用旧 Adam 2,000 步实验分数。

## 三个新版 GRT 源码策略接口

三者都从冻结的 input 至 CTS 前缀开始，仅执行 standalone `global-route`，读取 `80_route` 的 GRT checkpoint、metrics 与两 die `grt.log`。模型提交受限 Python 数值函数，可信 adapter 解析 AST 并翻译为 C++，而不是直接执行候选 Python。适用的 native 改动需要私有构建和已核安装二进制身份；编译成功不等于策略被调用。

| 任务与原始 prompt | 模型函数 | 原生作用点 | 特有约束 |
| --- | --- | --- | --- |
| [ripup policy](../local_tasks/global_routing_ripup_policy/prompt.md) | `score_ripup_edge(ctx)` | `RipUp.cpp` 中 `newRipupCheck` 的真实边 eligibility | 返回值 ≥0.5 则 ripup；非有限回退原决定；每循环最多 64 次，嵌套总 256 |
| [maze cost](../local_tasks/global_routing_maze_cost/prompt.md) | `score_maze_step(ctx)` | `maze.cpp` 的 maze relaxation | 返回乘数，有限值夹到 `[0.05,20]`，非有限为 1；每循环 8 次，嵌套总 16 |
| [iteration controller](../local_tasks/global_routing_iteration_controller/prompt.md) | `control_reroute_iteration(ctx)` | `FastRoute.cpp` 的 overflow reroute 迭代 | 输出 congestion scale、search scale、ripup offset；保留可信正值/范围夹取；每循环 64 次，嵌套总 256 |

上下文包含当前 congestion/history、die/HBT/criticality 等有界特征，必须按各 prompt 的归一化定义理解。例如 maze 的 usage/(usage+capacity)=0.5 表示 usage=capacity，不是 50% 利用率。允许局部标量、分支、受限循环和只读数值表，不允许 import、I/O、随机数、动态循环或任意外部调用。patch 增加上限 1,800 行包括可信接入代码，不是要求候选生成长函数；YAML 中 runner/route wrapper 路径是可信物化所需，不授予模型自由重写 evaluator 的权限。

三个任务统一奖励为：`score = hook_gate / [0.9 × W/W_ref + 0.1 × (O+1000)/(O_ref+1000)]`。W 为两 die FastRoute `GRT-0018` 线长之和，O 为两 die 最终 GRT overflow 之和；两 die 必须完成，且至少一 die 有活跃 hook 诊断。零 overflow reference 仍可有平滑拥塞信号，但**如果 reroute 根本不进入 hook，普通 GRT 成功不能给它补造调用证据**。新版四 case 的原生 hook 资格尚未建立，net ordering 不属于本家族。

阅读顺序是构建身份 → 安装二进制 → 冻结 CTS 输入 → hook 调用/变化计数 → 两 die 原生线长/overflow → 分数。修改乘数而 guide 未变可能是路径不敏感，不一定是构建失效；没有诊断则不能先假定 hook 生效。DRT、天线修复和 DRC 不在这些任务目标内。

## 全局 one OPT timing 任务

当前四 case authority 对应 [十轮 task.yaml](../global_tasks/lol_partition_placement_timing_20261004_10/task.yaml) 和 [instruction.md](../global_tasks/lol_partition_placement_timing_20261004_10/instruction.md)。五轮变体是独立任务，不可把十轮 authority 的 task digest 改成五轮。

模型优化 partition 与 placement 的共同机制，不受某个局部函数或五文件上限约束；可修改前端 Python 算法及 `agentic_candidate.yaml` 中的两个域、工作区内的 method JSON。但 `AgenticBench`、`tests`、`open3dbench`、`OpenROAD-3D`、`docker`、Place-3D API/validation/evaluation/HBT/scripts 均受保护。输入、assignment/checkpoint override、绝对配置路径、种子、后端和评分不可改；native C/C++/CUDA/build 修改要求另行匹配的镜像，当前驱动直接拒绝，不能“改了源码却仍用旧二进制”来评估。

冻结评估为 input → floorplan → partition → placement → PDN → HBT → **一次 OPT**；评分来自 pre CTS 全局 setup STA。令 `dW=max(0,-WNS)`、`dT=max(0,-TNS)`，参考正 deficit 下：

`score = 1 / max(0.6 × dW_candidate/max(dW_reference,1e-12) + 0.4 × dT_candidate/max(dT_reference,1e-12), 1e-12)`。

实现还处理 reference deficit 为零的边界，不能把公式直接套到任意旧数据；以可信 verifier 的结果为准。当前四 case 两个 reference deficit 均为正，self score=1。CTS skew、GRT 线长和诊断温度不混入该分数。

task 上限为 10 global rounds、10 expensive finals、最多 2 concurrent trials；容器模板 16 CPU、48 GiB、1 GPU、256 GiB 存储预算、6 小时任务 walltime、4,096 PIDs。实际 host admission 可能更严格；并发上限不是强制并发，也不是跨 controller 锁。agent turn 的 1,800 秒与物理评估预算分开。候选 final 包括失败，模型在 scratch 里做诊断不是 harness final。

单 agent 与 Agentic3D workspace baseline 共用同一任务、reference 和独立 evaluator；差异是提案与审查方式，不能改成不同后端来比较。实际 prompt 由 [run_workspace_codesign.py](../scripts/run_workspace_codesign.py) 的 `round_prompt` 和 [run_workspace_agentic3d.py](../scripts/run_workspace_agentic3d.py) 的 `_prompt` 生成。

## 全部已登记任务与版本归类

当前目录包含 14 个 local task 和 9 个 global task；“存在 YAML”不等于“本次四 case 已资格”。下列链接给出完整机器可读修改边界；同目录 `prompt.md` 或 `instruction.md` 为对应原始任务说明。

| Local task 目录 | 内容 | 在最新家族中的定位 |
| --- | --- | --- |
| [netlist_partition](../local_tasks/netlist_partition/task.yaml) | Mt KaHyPar 配置及有限 partition patch，cut/balance | 兼容任务，不是本批 TP GNN loss |
| [tp_gnn_learning_objective](../local_tasks/tp_gnn_learning_objective/task.yaml) | 标准 loss hook | 历史输入合同 |
| [tp_gnn_learning_objective_reference_20260923](../local_tasks/tp_gnn_learning_objective_reference_20260923/task.yaml) | reference loss hook | 可显式绑定新版输入，待 native authority |
| [tp_gnn_ta3d_learning_objective](../local_tasks/tp_gnn_ta3d_learning_objective/task.yaml) | TA3D 风格 loss/decoder 合同 | 单独选择的历史研究变体 |
| [tp_gnn_pin_cut_learning_objective](../local_tasks/tp_gnn_pin_cut_learning_objective/task.yaml) | signed critical/noncritical pin cut | 需专用 pin 连接证据，不继承本批资格 |
| [placement_init_preconditioner](../local_tasks/placement_init_preconditioner/task.yaml) | 原始初始化/预条件 task | 历史输入合同 |
| [placement_init_preconditioner_reference_20260923](../local_tasks/placement_init_preconditioner_reference_20260923/task.yaml) | reference 初始化/预条件 hook | 可显式绑定新版，待 native authority |
| [placement_parameters](../local_tasks/placement_parameters/task.yaml) | 原始动态布局控制 | 历史 Adam 合同，不能混入 Nesterov |
| [placement_parameters_reference_20260923](../local_tasks/placement_parameters_reference_20260923/task.yaml) | Nesterov 动态控制 | 可显式绑定新版，待 native authority |
| [global_routing_parameters](../local_tasks/global_routing_parameters/task.yaml) | bounded routing 配置 | 兼容入口，非三项源码策略 |
| [global_routing_net_ordering](../local_tasks/global_routing_net_ordering/task.yaml) | net order function | 最新家族排除 |
| [global_routing_ripup_policy](../local_tasks/global_routing_ripup_policy/task.yaml) | 真实 edge ripup | 新版接口已适配，待原生 hook 资格 |
| [global_routing_maze_cost](../local_tasks/global_routing_maze_cost/task.yaml) | maze edge cost | 新版接口已适配，待原生 hook 资格 |
| [global_routing_iteration_controller](../local_tasks/global_routing_iteration_controller/task.yaml) | reroute iteration policy | 新版接口已适配，待原生 hook 资格 |

| Global task 目录 | 目标和轮数模板 | 本批可用性 |
| --- | --- | --- |
| [lol_full_flow_ppa](../global_tasks/lol_full_flow_ppa/task.yaml) | 用户定义 PPA，12 finals | 通用模板，非已资格的固定目标 |
| [lol_partition_placement_timing](../global_tasks/lol_partition_placement_timing/task.yaml) | 历史 two OPT STA，5 finals | 保留，不移植 authority |
| [lol_partition_placement_timing_one_pass](../global_tasks/lol_partition_placement_timing_one_pass/task.yaml) | 旧家族 one OPT，5 finals | 保留 |
| [lol_partition_placement_timing_one_pass_10](../global_tasks/lol_partition_placement_timing_one_pass_10/task.yaml) | 旧家族 one OPT，10 finals | 保留 |
| [lol_partition_placement_timing_20261004](../global_tasks/lol_partition_placement_timing_20261004/task.yaml) | 新版 one OPT，5 finals | 接口已注册，需自身 authority |
| [lol_partition_placement_timing_20261004_10](../global_tasks/lol_partition_placement_timing_20261004_10/task.yaml) | 新版 one OPT，10 finals | 四 case authority 已齐 |
| [lol_partition_placement_cts_thermal](../global_tasks/lol_partition_placement_cts_thermal/task.yaml) | 历史 two OPT 后 CTS cell power thermal，5 finals | 非新版默认 |
| [lol_partition_placement_cts_thermal_one_pass](../global_tasks/lol_partition_placement_cts_thermal_one_pass/task.yaml) | 旧家族 one OPT 后 CTS thermal，5 finals | 不可冒用本次独立诊断 |
| [lol_partition_placement_thermal](../global_tasks/lol_partition_placement_thermal/task.yaml) | 完整 routed HotSpot，5 finals | 涉及 DRT，本次禁止执行 |

旧 thermal 奖励为 reference mean K / candidate mean K，覆盖物理 silicon tiles；峰温是诊断，不能拿本次峰温表直接重算其分数。旧 task 具有各自 source、evaluator、resource 和 scope 字段，表中摘要不是统一覆盖它们的配置。

## 日志分析与计数要求

先看 `run_config`、status、history/summary，再核 `results/reference`、`results/round-*`、submission/verifier 回执，最后定位物理 `run.json`、stage manifest 和 native log。`RUN_SUMMARY.md`、`SUMMARY.md` 和 events 是阅读索引，不是权威评分输入；原始证据不应为了说明而原地改写。新日志中的 INPUT、CURRENT_EXECUTION、OUTPUT 以及 copied upstream 区段用于区分读取旧数据和执行新算法。

必须分别报告模型 role turns、provider/API 实测、生成候选、verifier 拒绝、build 容器、eval 容器、known/unknown native attempts、final、成功评价、registry 发布与独立改善。没有 API 逐调用证据就保持 unknown；进程墙钟不是纯 API 延迟，container dispatch 不是 native solver 已启动。候选普通失败消耗其既定槽，框架失败先核执行边界，不能直接补轮。

模型供应商属于实验配置，不等于评分合同。局部框架有 DeepSeek 配置，但当前 workspace Codex driver 的 `MODEL` 仍固定 `gpt-5.6-sol`，不能只换 API 地址就宣称全局 baseline 已切到 DeepSeek。此前用户提供的凭据不写入本报告、源码或日志；本次未发模型请求，也未更改既有认证。

三个 reference 局部任务的 evolution config 默认 50 iterations、单 evaluation 并行；实际 dispatch 的预算、缓存与模型以运行配置为准。旧四 case 的 680 槽历史实验属于另一批，不是本次审计获准自动重开的预算。恢复必须匹配任务、prompt、输入、评分、实现和已消费槽，不能借“整理仓库”重启旧队列。
