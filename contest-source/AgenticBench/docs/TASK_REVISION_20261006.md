# 2026-10-06 切分时序目标与布局源码任务拆分

后续用户决定：切分任务基线已切回 global-2way，WNS/TNS 的无 OPT 评分不变；当前入口为 `tp_gnn_post_placement_timing`。见 [最新切换说明](PARTITION_GLOBAL_2WAY_TIMING.md)。下文保留早前 bin 版本的实施与测试记录，不代表 bin 仍是当前统一 reference。

## 当前结论

主要接口、prompt、system prompt、评分与只读证据检查已实现于本地。尚未完成新的原生 reference 资格、服务器发布或 GPT-5.6 六 case 候选评价，不能称为“六 case 已跑通”。旧结果、旧任务 ID、旧评分及运行中的任务保持不变。

## 新任务

| 任务 | 模型修改范围 | 执行终点与评分 |
| --- | --- | --- |
| `tp_gnn_bin_timing` | `compute_tp_gnn_loss`，18 参数 timing-context-v2；其余训练/decoder 固定 | partition → placement → PDN → HBT → 一次统一两 die STA；无 OPT。WNS/TNS 亏损成本加权倒数。 |
| `placement_initialization` | `initialize_position`；仅 BasicPlace.py | 原生 Nesterov global placement；PreconditionOp 原生不变。 |
| `placement_preconditioner` | `PreconditionOp`；仅 PlaceObj.py | 原生 Nesterov global placement；初始化原生不变。 |

新配置与 prompt 分别在 `local_tasks/<任务>/`，system prompt 位于 `config.yaml`。通过同一 ask/tell 接口和 OpenEvolve 完整程序模板使用；反馈白名单移除 cut fitness，保留真实测得的 timing/partition/embedding 诊断。基础设施路径、版本和资格材料由宿主检查，不送入模型的优化问题正文。

两个源码任务不相互引用搜索结果，也不共同改变两部件。所有新任务保留旧任务作为历史，不复用旧 authority 充当新资格。四个 routing 任务及动态 Nesterov 任务本次未修改。

## 切分 reference 与评分

默认采用 `tp_gnn_spatial_bin.json`：4×4 空间 bin，宏先独立 embedding 聚类、满足宏数量/面积约束，再以宏实际几何覆盖分摊每个 bin 的固定负载；标准单元仅在对应 bin 中聚类与均衡。当前仍保留严格每 bin top 面积占比 0.5±0.03、全局约束和宏 count/area 0.5±0.05，没有偷偷放宽无解 bin。

记 `dW=max(0,-WNS)`、`dT=max(0,-TNS)`，单位 ns；参考亏损大于 1e-12 时成本为候选亏损/参考亏损；两者均不大于 1e-12 时成本为 1；仅参考接近零时成本为 `1+候选亏损/1e-12`。

`fitness = 1 / max(0.6*WNS_cost + 0.4*TNS_cost, 1e-12)`。

原生同 case reference 分数为 1；timing-weighted cut 不再进入 fitness。比较对象必须使用同一个 bin decoder、训练配置、HBT RC 模型和 upstream 输入。只有完整通过训练、分区、布局、PDN、HBT 及 STA 证据检查的结果才可评分；缺失 timing 不能转换为 0。

STA 使用现有 `post_placement_sta`，输入固定为 `50_hbt/checkpoint.json`，输出 `55_post_placement_sta`。它估计 die-local placement RC 并以 passive HBT π 网络连接两 die；理想 pre-CTS 时钟，无 timing optimization。验证器核对阶段 manifest、方法报告绑定、HBT checkpoint、SHA/大小、约束覆盖及 setup 路径，拒绝含 OPT 或后续物理阶段的 run。结果不等于 signoff。

## 布局任务

两任务只在各自源码组件内演化。参考为空源码 patch 的真实原生部件；另一组件必须逐字保持原生。模型可获得 die-wide utilization、target density、aspect ratio；不声称已经提供局部密度图。预条件器说明梯度除以正尺度的作用，并要求保持固定节点、region mask、dtype/device 和有效密度力。

两任务固定 4000 Nesterov global-placement iterations、512 bins、target density 0.6、initial density weight 0.00003、learning rate 0.01；宏 halo 从 PDK 传入，ng45 与 ASAP7 不共用写死的 10 µm。模型不能修改 die assignment、其他优化步骤或 stop criteria。

`fitness = 0.9*H_ref/max(H,1e-12) + 0.1*max(O_ref,0.01)/max(O,0.01)`。

H 为 native weighted HPWL，O 为 worst-die overflow；0.01 是既有停止容差，不另加任意 0.03 准入门。只适用于两个新拆分任务，不改写历史 score。GP 结果不等于合法化或 STA。

## 验证与未完成项

- 针对新任务、反馈、历史布局兼容、prompt 和 post-placement STA：161 passed、12 skipped；其中部分 OpenEvolve 集成检查因可选依赖不可用跳过。
- 六 case × 三新任务，共 18 个真实 CLI plan/dry-run 通过。没有模型调用、EDA 容器或候选 final。
- 三机只读发布核验通过：ta2、ta3、czys 仍一致使用 `fleet-221047e457db-7c0f27cc64b7`。本次新代码尚未部署；没有把旧 release 状态冒充新修改已同步。
- 更广的本地 AgenticBench 历史套件未全绿：2027 passed、136 failed、80 errors、31 skipped、109 subtests passed。已观察到历史 reference 文件不存在、旧 GRT 接口缺失等；尚未完成逐项根因分类，不据此声称全套回归通过。
- 三个新任务缺失 reference 时，仅精确初始程序可进行 FRESH 资格运行；其他候选拒绝。ADOPT 核初始程序、任务身份、上游 manifest 和实现哈希，并重新提取原始指标，不只相信成功回执。
- 上轮严格 bin 方案已有五 case 失败；Gemmini 尚未确认终态。宏固定负载超出可达平衡区间、单 cell bin 的整数不可分性，不能靠修改 loss 或反复重试解决。没有把这些失败升级为新 reference。
- 已请求用户选择：合并不可行的相邻 bin，或保持 bin 不变并采用最接近可达的局部均衡。这会改变科学约束，当前尚未实施。

## 后续小规模测试计划

case：bp_fe/ng45、bp/ng45、ariane133/ng45、swerv_wrapper/ng45、bp_fe/ASAP7、gemmini/ASAP7。每 case 对三个新任务各生成并评价一个 GPT-5.6 候选，共 18 个候选；原生 reference 单独计数，不占这 18 个。待无解 bin 策略确认、完整版本准备与资格检查通过后，使用 ta2、ta3、czys 的独立资源槽并行。正式模型必须验证为用户指定的 GPT-5.6，不静默替换其他名称。

同步遵循 agenticbench-fleet-sync 的 committed-release 与跨主机一致性门；新 reference 是单独的任务资格，不能改写现有 reference_20261004 的 producer。当前不激活未完成资格的版本，不停止或重启旧健康实验。
