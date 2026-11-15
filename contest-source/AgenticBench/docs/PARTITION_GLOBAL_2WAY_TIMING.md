# 切分任务：global 2-way 基线与布局后 STA

2026-10-06：按用户要求，将当前切分任务基线切回 global-2way；**只切换 decoder 基线，不回退时序 fitness**。

当前核心入口为 [`tp_gnn_post_placement_timing`](../local_tasks/tp_gnn_post_placement_timing/task.yaml)，替代清单中历史 cut-score 入口。八个核心局部任务的总数不变。默认 recipe 是 `Place-3D/config/partition/tp_gnn_fast.json`，当前 `TPGNNConfig.decoder` 为 `global-2way`；运行前核配置，结束后独立核 hash-bound native report 中的实际 decoder，不能仅靠 task 名称或默认值认定成功。

## 保持的任务合同

- 只演化 `compute_tp_gnn_loss`，沿用 18 参数 timing-context-v2、训练预算、种子、图/特征及固定 timing weights。
- global 2-way 在全设计上做面积加权两类聚类并执行既有可行性修复；无 per-bin 均衡要求。全局 top 面积比例 47%–53%，有宏时宏数量与宏面积比例 45%–55%。不改变全局约束。
- 候选从同一 input/floorplan 和固定 timing input 运行 partition → placement → PDN → HBT，再做统一两 die STA；无 OPT、CTS、GRT、DRT、thermal。
- `score = 1 / max(0.6 * normalized_WNS_deficit + 0.4 * normalized_TNS_deficit, 1e-12)`。亏损为 `max(0,-slack)`，原有零 reference 规则不变。cut、loss、embedding 等是诊断，不是 fitness。
- 布局和 HBT 方法、理想时钟、placement RC 与 passive HBT π 模型不变。pre-CTS STA 不是 signoff。

## 身份与历史隔离

旧 `tp_gnn_bin_timing` 的配置、prompt、初始 candidate 和已生成结果保留；bin-based 算法及接口也保留，仍可显式选择，但不在当前核心清单中充当切分基线。历史 `tp_gnn_learning_objective_reference_20260923` 保持 cut-score 语义，不伪装成新时序任务。

新 task ID 与 decoder 绑定。不能采用旧 bin/cut reference 冒充新 task 的资格；同 decoder 也必须继续核任务 ID、初始程序、实现哈希、输入 manifest 和真实 STA 证据。缺少 reference 时，仍只有精确初始程序可以建立 FRESH reference，演化候选不得因缺少参考而自动得到 1 分。改变全局任务/布线任务或两项布局任务的参考合同不在本次范围内。

## 模型内容与反馈

[Prompt](../local_tasks/tp_gnn_post_placement_timing/prompt.md) 与 [system prompt](../local_tasks/tp_gnn_post_placement_timing/config.yaml) 均描述 global 2-way 和最终 WNS/TNS；不再要求宏先固定或局部分 bin 均衡。模型继续只需 ask/tell；反馈保留原始联合 WNS/TNS、归一化亏损成本、全局可行性及原有真实诊断，移除新任务不适用的 bin-area 指标。

本次为代码、定义与离线回归修改；没有启动模型/EDA、重新派发实验、同步服务器或覆盖 reference。新的六 case 专用 reference 尚需独立资格运行，不能把本次改配置称为六 case 已重测。

## 离线验证

### 当前 OpenEvolve 反馈精简

当前 global-2way 时序任务通过统一的反馈投影给 ask/tell、OpenEvolve 原生适配器和历史示例提供相同内容。原始 evaluator 回执保留，历史 cut-score/bin 任务的模型投影不变。

- Fitness：只以 `combined_score` 选择；显示联合 WNS/TNS 及两项归一化成本，不重复组合成本和固定权重。
- Validity：运行成功、assignment 完整性、约束通过与三项均衡误差；失败原因保留并移除内部路径。
- Partition：final cut-net count、timing-weighted cut 及其比率。
- Decoder/repair：raw/balanced/feasible 的 weighted cut；两个转换的 signed delta 与 relative change（分母为零时不提供）；修复移动 supernode 数和面积比例。
- Training：final loss、实际 epoch 数、embedding variance、分正负类的配对 cosine median 及样本数。
- Status：错误原因。整体接受/拒绝/失败状态和错误类型仍由 ask/tell 的外层反馈提供；不由 score 猜测。

修复面积比例按 balanced→feasible 换层实例的原始面积求和（包含宏），除以设计全部实例面积；它不是几何位移，也不是初始面积均衡阶段的移动量。正负 cosine 在训练结束后用 final embeddings 计算，分别选最后一个实际完成训练 batch 中最多 4096 个等距索引配对；不增加随机采样、模型前向或优化器步骤，不代表全图分布。旧报告没有这些新测量时保持缺失，不用普通随机 pair cosine 冒充正负统计。loss 跨不同公式/尺度不能直接比较。

不再逐候选向模型发送 instance/top/bottom counts、total net weight、成功的 graph-completeness/OPT=0 常量、optimizer steps、普通训练耗时、cut-pin/重复 plain-cut 分解与随机 pair cosine。原始回执仍可人工审计；异常训练通过失败状态/原因解释。初始候选 patch 随诊断代码的行号变化重新生成，loss 正文、global-2way decoder、WNS/TNS fitness 和预算均未变。本次未同步服务器、运行 EDA/API 或更新 reference authority。

本次反馈精简定向回归：185 passed、2 subtests passed。覆盖真实 OpenEvolve ask/tell 与原生适配器投影一致性、原始回执不变、失败原因/缺失值、weighted-cut 零分母、修复移动面积、配对采样上限和真实 CPU 训练结果/预算不变；无模型 API 或 EDA。未宣称完整测试套件通过。

以下为此前基线切换的历史验证记录：新增针对 global 2-way 切换的 15 项检查全部通过，覆盖默认 recipe/候选一致性、core 清单、prompt/system/ask-tell、decoder 双向拒绝、同合同原生 reference 准入、WNS/TNS 权重、错误 bin 配置在 native 前拒绝，以及 artifact 篡改/OPT 混入拒绝。

扩展回归：356 passed、20 skipped、2 subtests passed、1 failed。跳过项依赖未安装的 OpenEvolve；唯一失败是旧 `tp_gnn_bin_timing` 保存的初始 patch 与当前 TP-GNN 源码生成 patch 的 hunk 行号不同（正文一致）。该旧文件本次未改动，也未将旧 reference 重标为新 reference；新任务的初始 patch 已对当前源码重新 materialize 并通过一致性检查。这里不声称全套回归或真实物理实验通过。
