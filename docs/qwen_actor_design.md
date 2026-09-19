# Qwen3.5-0.8B 共享 Actor：设计与实施方案

状态：核心实现已完成（MLP/Qwen 切换、MAPPO、SFT、采集及批推理），仅完成 CPU 替身与非训练回归检查；真实模型/GPU 训练尚未验证。当前接口和交付边界见 [使用说明](qwen_actor_usage.md)。可选旧 Swift 标量检查点转换器未纳入本次实现。
本地没有 GPU：本阶段仅允许源码开发、静态检查和使用小型替身模型的 CPU 单元测试，不下载大模型、不启动实际训练。

## 1. 目标与设计决定

在现有 MAPPO 中增加 `actor_type="qwen"`，保留默认 `actor_type="mlp"`。
多机任务继续固定 10 架。两个分支使用同样的环境、动作含义、团队奖励、GAE 和集中式 MLP critic。

Qwen 分支使用一份 Qwen3.5-0.8B 文本骨干及一份动作 MLP 输出头。
每个仿真控制步将 10 架无人机各自的局部观测转换成 10 条独立 prompt，组成 batch，一次调用骨干获得 `[10, action_dim]`。
不将 10 架的观测拼成一条 prompt，也不创建 10 份模型。不同 batch 行之间不能互相读取观测。

输出头预测连续动作分布的均值，不生成文本。共享 `log_std[action_dim]` 仍由 MAPPO 管理。
动作采样、tanh 变换及 log-prob 计算沿用现有算法，critic 继续处理原始数值联合观测。

首版采用「冻结骨干前部，训练最后两个文本 decoder block、动作头及 log_std」。
这与现有 action-head 示例一致，且避免同时引入 LoRA、量化或新的训练框架。
精确冻结边界由实际模型结构确认，不凭参数名模糊匹配；不冻结整个 actor 的计算图。

批处理能减少调用开销并提高硬件利用率，但计算量仍随 batch 大小增长；不能据此承诺达到 30 Hz 实时控制或优于 MLP。
PyBullet 可以按仿真时间离线推进，模型延迟不改变 `CTRL_TIMESTEP`。

## 2. 已核对的源码依据

当前仓库：

- `gym_pybullet_drones/learning/mappo.py`：actor 为两层 128 单元 Tanh MLP，输出维度已经使用 `action_space.shape[1]`；critic 将全体观测展平后输出团队价值。
- `gym_pybullet_drones/envs/MultiHoverAviary.py`：固定 10 架；KIN 观测末尾追加目标相对位移。
- `gym_pybullet_drones/envs/BaseRLAviary.py`：KIN 顺序为位置、姿态、线速度、角速度，再接最近半秒的动作历史；VEL 为四维动作。
- `gym_pybullet_drones/examples/learn.py`：MAPPO 默认 VEL，30 Hz 控制频率，rollout 512 个联合步，优化 batch 默认 256 个联合步。

本地参考工程：`C:/Users/mateogic/Desktop/llm-muav/ms-swift`。

- `examples/train/seq_cls/qwen3_5_action_head/README.md`、`train/train.sh`：使用 `seq_cls`、`num_labels=1`、`problem_type=regression`；训练最后两层和 `score`。
- 同目录 `infer/infer.py`：使用 `TransformersEngine.infer()` 返回字符串，适合演示，不适合 MAPPO 的可微训练路径。
- 同目录 `data/generate_dataset.py`：任务是整数加法，其数据和回归标签不能直接用作无人机控制数据。
- `swift/model/patcher.py` 中 `_patch_sequence_classification()`：将 `lm_head` 替换为 Identity，构造四层 Linear、三层 SiLU 的 `score`。
- 同文件 `transformers_seq_cls_forward()`：先逐 token 应用 score，再取最后一个非 padding token；提供 labels 时执行监督损失。

官方配置确认 Qwen3.5-0.8B 文本 hidden size 为 1024、decoder 层数为 24，包含线性注意力与完整注意力混合层；仓库入口为多模态条件生成模型。因此不能假设普通 CausalLM 的类名、隐藏状态位置或缓存逻辑可直接套用。
来源：[官方 config.json](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json)、[官方模型页](https://huggingface.co/Qwen/Qwen3.5-0.8B)。实施时固定模型 revision 和依赖版本，而不是依赖持续变化的 main。

## 3. 动作维度：改输出头，不改任务含义

统一以 `D = env.action_space.shape[-1]` 为准；智能体数量 N=10 属于 batch 维，不是输出头宽度。

| ActionType | D | 每架动作含义 |
| --- | ---: | --- |
| VEL（默认） | 4 | 方向 x/y/z 三分量、速度幅值控制量 |
| RPM | 4 | 四个电机相对悬停转速的控制量 |
| PID | 3 | 目标位置 x/y/z，沿用当前环境实现 |
| ONE_D_RPM | 1 | 四电机共用的转速控制量 |
| ONE_D_PID | 1 | 垂直位置增量控制量 |

VEL 的实际映射是：

```text
direction = a[:3] / ||a[:3]||，零向量时为零
target_velocity = SPEED_LIMIT * abs(a[3]) * direction
```

因此不能把默认输出改成三维并直接解释成 vx/vy/vz，也不能输出一个数再复制给所有动作维度。
本次设计不改变 VEL 的映射或速度上限；如以后采用笛卡尔三维速度动作，应作为独立环境变更，对两个 actor 同时适用并重新训练。

现有 MLP actor：`obs_dim → 128 → 128 → D`，最后一层本来就是动态 D，无需为 Qwen 改成固定 4 或 40。
需要修改的是参考工程的标量 action MLP 末层：

```text
Qwen pooled hidden [B, H]
  → Linear(H,H) → SiLU
  → Linear(H,H) → SiLU
  → Linear(H,H) → SiLU
  → Linear(H,D)
  → mu [B,D]，无末层 tanh
```

H 从文本配置读取，当前模型为 1024。最后一层采用小幅初始化（例如现有 MLP 的 0.01 输出增益）、零偏置。
使用原标量检查点时，只允许显式跳过尺寸不匹配的最终输出层并重新初始化；输出缺失/多余参数报告。
不复制一维输出权重去制造四维控制，不通过 `ignore_mismatched_sizes` 静默忽略所有问题。
加法任务训练过的中间层没有已知控制优势；默认从官方预训练骨干初始化新动作头，原动作头检查点作为显式可选初始化来源。

## 4. 文本骨干加载与池化

在独立 Qwen 适配器中封装模型版本相关操作；MAPPO 不依赖 ms-swift 的全局 monkey patch 或另一仓库的绝对导入路径。
复用参考实现的结构和冻结策略，不把 `swift sft`、MSE labels 或 `TransformersEngine` 带入在线 PPO。

适配器负责：

1. 用固定版本 Transformers 支持的加载器加载本地 Qwen 检查点；核实权重映射后访问文本 decoder。不可直接假定 `AutoModelForCausalLM` 支持完整 checkpoint。
2. 仅执行文本路径，避免 vision 前向和词表 logits `[B,L,V]`；最终运行对象只保留需要的文本骨干与动作头。保留输入 embedding，即使原始 LM head 与其共享权重。
3. 文本 forward 返回最后一层 hidden `[B,L,H]`。不要开启所有层的 `output_hidden_states` 只为读取最后一层。
4. 通过 attention mask 取最后一个有效 token 的 hidden，再运行动作头。这与参考代码的逐 token MLP 后池化等价（该头无跨 token 运算），但避免对所有 L 个 token 重复执行四层 MLP。
5. `use_cache=False`，不使用 `generate()`，不保留跨步 KV cache 或线性注意力状态；每次完整 prompt 是独立样本。

首版固定右侧 padding，以 `lengths = attention_mask.sum(-1)`、`indices = lengths - 1` 池化；拒绝空序列。
使用 tokenizer 的有效 pad ID，必要时显式配置并保存；attention mask 不通过比较 token ID 推导。
若模型调用需要 `position_ids`，由有效 token 的累积位置生成，padding 位置置零；适配器必须验证批推理与逐样本结果一致。
混合注意力的 mask 和位置参数使用该固定版本的官方实现，不手写注意力层。

## 5. 动态 prompt 与 1024 上限

将限制定义为「含特殊 token、实际送入模型的非 padding 输入最多 1024 tokens」。为覆盖用户按字符理解长度的情况，首版同时要求 prompt 文本不超过 1024 个字符。
真实 tokenizer 的 token 数是最终依据，不能凭字符数估算满足 token 限制。

默认观测布局（VEL、30 Hz）：

```text
obs[0:3]    position，m
obs[3:6]    roll/pitch/yaw，rad
obs[6:9]    linear velocity，m/s
obs[9:12]   angular velocity，rad/s
obs[12:-3] action history，15 × 4，按时间从旧到新
obs[-3:]   target - position，m
obs_dim = 12 + 15*4 + 3 = 75
```

prompt 包含完整局部观测，不引入其他无人机状态、团队距离、critic 输出、未来信息或当前动作标签。
使用短英文固定模板、简短字段名和固定小数格式；无需聊天模板、思考链、输出 JSON 或生成回复提示。
这样也与参考训练脚本 `use_chat_template=false` 一致。序列结尾固定为 `Action features:`，池化其最后有效 token。

模板示意（实际 history 必须填满全部行，不能输入省略号）：

```text
Fly to your goal quickly and hover. World xyz; angles rad; p/g m; v m/s; w rad/s.
Action vel: xyz direction and speed magnitude, each in [-1,1]. History oldest first.
p=[0.000,0.000,0.113]; rpy=[0.000,0.000,0.000]
v=[0.000,0.000,0.000]; w=[0.000,0.000,0.000]
g=[0.000,0.000,1.000]
h=[完整的15行4列历史动作，以分号分行]
Action features:
```

实施规则：

- `ObservationPromptEncoder` 接受结构化 `ObservationSpec`：字段顺序、D、历史长度、动作类型、频率和格式版本。
- 数值固定三位小数，负零规范成 `0.000`，拒绝 NaN/Inf；量化是 Qwen 的观测预处理，记录在检查点中。不能声称其与原始 float32 MLP 输入完全等价。
- 每个动作模式有准确且固定的动作说明；不在 VEL 与 RPM 之间复用错误含义的文字。
- 批量构造字符串、一次 tokenizer 调用；允许 Python 遍历生成文本，禁止逐无人机循环调用骨干。
- `padding="longest"`，`truncation=False`；逐行检查字符数与 token 数，超过任一限制时抛出含样本索引的明确错误。
- 不截断目标/历史，不通过动态减少精度或删字段悄悄改变策略输入。非默认频率导致历史超长时，要求显式定义并版本化更紧凑 schema 后重新验证。
- tokenizer、模板、字段顺序、数值精度和特殊 token 设置在整个 rollout 与其所有 PPO epochs 中保持一致。

默认完整模板是否覆盖全部允许状态范围，需要真实 tokenizer 离线验证；本文不虚报已得到精确 token 数。越界检查保证任何送入模型的输入都满足 1024 上限。

## 6. 共享 actor 接口与批处理

提取统一接口，避免在 MAPPO 损失里到处判断 actor 类型：

```python
actor = build_actor(actor_config, observation_spec, action_dim)
inputs = actor.encode(local_observations)    # MLP: 数值；Qwen: token IDs/mask
mu = actor(inputs)                          # [B, D]；恰好一次 actor forward
distribution = Normal(mu.float(), log_std.clamp(-5, 2).exp())
```

`MLPActor` 保留现有网络；`QwenActor` 包含唯一一份文本骨干和动作头。
二者都接受展平的局部 batch。`predict()` 在边界记录前导形状、reshape、调用 actor 一次，再还原形状；单架输入 `[O]` 也可用。

| 场景 | 原始输入 | Actor batch | 均值输出 |
| --- | --- | --- | --- |
| 一个控制步 | `[10,O]` | 10 条独立样本 | `[10,D]` |
| 更新 M 个联合步 | `[M,10,O]` | `[M*10,...]` | reshape 为 `[M,10,D]` |
| 单架部署 | `[O]` | `[1,...]` | `[D]` |

一个控制步的 Qwen batch 固定 10，一次文本 decoder forward。训练不是整个 rollout 只前向一次：PPO 每个 epoch 都必须在新权重上重新计算 log-prob。
critic 每个联合步只估值一次，不因为有 10 架重复拟合 10 份相同的团队目标。

## 7. MAPPO 数学语义与梯度

继续采用：

```text
u_i ~ Normal(mu_i, exp(log_std))
a_i = tanh(u_i)
log pi_i = sum_d [log Normal(u_i,d) - log(1 - tanh(u_i,d)^2)]
ratio_i = exp(new_log_pi_i - old_log_pi_i)
```

沿用当前数值稳定的 tanh Jacobian 实现。动作维求和、智能体维保留；禁止将所有无人机概率相乘后只裁剪一个联合 ratio。
actor loss 对时间与智能体取平均；团队 advantage 广播至各架。critic 仍输出共同回报的标量，保留现有 GAE、失败不 bootstrap、超时用重置前观测 bootstrap 等逻辑。

训练时不传监督 `labels`，不依赖 seq_cls 自动分类/回归判断，也不把 token 对数概率当作动作概率。
若单独做可选行为克隆预热，才使用 `[B,D]` 连续标签，并明确 `problem_type="regression"`；D>1 也不是分类。

防止 PPO 分布漂移：

- 在 rollout、评估、优化阶段均关闭 actor dropout。可让 actor 保持 `eval()`，更新时仍正常启用 autograd；`eval()` 不等于 `no_grad()`。
- rollout 使用 `no_grad()`；更新不能通过推理引擎、`.detach()` 或覆盖整个 actor 的 `no_grad()` 切断梯度。
- 冻结前 22 层和 embedding，仅最后两层、score、log_std 可训练；最终 norm 首版保持冻结。冻结参数不需要训练用梯度，但末两层前向必须保留计算图。
- 禁止跨 PPO 更新缓存可训练层的 hidden 或均值；缓存 token IDs 是安全的。
- 在更新前用相同存储 token 重算 log-prob，应与 old log-prob 一致到合理浮点误差，ratio 约为 1。
- BF16 骨干（GPU 支持时）与 FP32 动作头/分布计算可组合。可训练的末两层保留 FP32 主权重，Adam 梯度、动量与检查点均使用 FP32；每次前向通过可微 dtype 转换生成临时 BF16 计算权重，反传回 FP32 主权重，不缓存跨步副本。避免低学习率更新被直接写入 BF16 参数时舍入丢失。显式转换 pooled hidden，保证概率、优势和损失用 FP32。

优化器使用 trainable 参数组：文本末两层、动作头/log_std、critic，可分别配置学习率；冻结权重不进入优化器。
actor 和 critic 分别裁剪梯度。候选起点为骨干 `1e-5`、头 `1e-4`、critic `3e-4`，仅作将来 GPU 实验起点，不作为已验证最优值。

## 8. Rollout 存储与更新显存

保留原数值观测给 critic。Qwen rollout 另存每条实际使用的 token IDs 和长度，padding mask 可在组 batch 时重建。
保存 raw action、old log-prob、old value、advantage、return；不保存完整骨干 hidden states。
token IDs 以 CPU int32 存储，送模型时转 long。512×10×1024 的固定上限存储约 20 MiB，实际可按变长序列存储减少开销。

当前 256 联合步等于 Qwen 的 2560 条序列，不能直接作为单次 GPU forward 默认值。
保留 `batch_size` 的「联合步」含义，并增加 `update_microbatch_steps`。Qwen 初始取 1，即每次更新前向仍为 10 条序列。

对一个优化 batch 的 M 个联合步：

1. 随机采样联合步，保持每步 10 架顺序；GAE/advantage 先按完整 rollout 计算。
2. 划分为 m 个联合步的小批；其 Qwen 输入为 `[m*10,L]`，每个小批一次骨干前向。
3. 每个小批 actor、critic、entropy 的均值损失乘 `m/M` 后 backward。
4. 所有小批累计完成，分别梯度裁剪，再一次 optimizer step；不在每个小批更新参数。

尾批也按真实 M 加权，不能用固定累积次数平均。显存仍不足以容纳 10 条序列时，报告无法满足一次控制步整批推理的硬件条件，不静默退回逐无人机调用。
仅冻结前层不会消除其推理成本；梯度检查点仅在将来的 GPU 性能验证后启用，需确认混合注意力和冻结前缀下梯度正确。

## 9. 配置、依赖与检查点

新增 actor 配置并随 checkpoint 保存：

```text
actor_type: mlp | qwen（默认 mlp）
model_path / model_revision / tokenizer_revision
trainable_last_n_layers: 2
max_prompt_tokens: 1024
max_prompt_chars: 1024
prompt_schema_version: kin_v1
number_precision: 3
padding_side: right
backbone_dtype / attention_backend
actor_backbone_lr / actor_head_lr
update_microbatch_steps
```

无人机数量继续由环境固定为 10，不重新引入 `--num_drones`。
`learn.run()` / CLI 增加 actor 类型与模型路径入口；单机 SB3 PPO 对 Qwen 选项明确报错，避免悄悄忽略。
`play.py` 依据检查点 actor 类型恢复，不要求使用者再次选择模型架构。

Qwen 依赖作为可选组，延迟导入 Transformers；MLP 分支不应需要安装或下载 Qwen。
记录实际验证过的 Transformers、Torch、tokenizers 版本。当前只完成源码级方案核对，不编造一个未经验证的最低兼容版本。
CPU 开发配置使用本地小型替身，不做真实模型 CPU 训练；未来 GPU 路径使用明确单设备，不依赖 `device_map="auto"` 自动分片进行 PPO 训练。

MLP 保持现有 `.pt` 保存/加载兼容；无 `actor_type` 的旧 checkpoint 按 MLP 加载。
Qwen 使用 checkpoint 目录，避免每次 best/final 保存都复制完整冻结骨干：

```text
best_model/
  manifest.json       # 格式版本、完整配置、ObservationSpec、基座 revision/hash
  tokenizer/          # 固定 tokenizer 文件
  trainable.pt        # 文本末层、score、log_std、critic
  optimizer.pt        # 优化器、步数、随机状态（仅恢复训练需要）
```

Qwen 的 manifest 必须指定不可变基座版本；本地模型需可验证文件清单/hash。
恢复先加载同一基座，再严格覆盖全部可训练权重。只保存 score 会丢失末两层训练结果，属于错误实现。
保存数值观测布局、动作模式、目标、频率和 prompt 版本；加载时检查 D、O、N 和环境一致，不能只凭维度相同接受 RPM/VEL 互换。
回放不恢复 optimizer，恢复训练才加载其状态并校验参数组一致。
即使保存随机状态，本方案也不保存 PyBullet 中途状态；恢复训练从新 episode 开始，不宣称逐步等同于未中断训练。

## 10. 文件实施清单与顺序

| 文件 | 责任 |
| --- | --- |
| `learning/actors.py`（新增） | 统一 actor 接口、MLPActor 与 factory；Qwen 延迟导入 |
| `learning/qwen_actor.py`（新增） | 骨干加载适配器、mask 池化、四层动作头、冻结边界 |
| `learning/observation_prompt.py`（新增） | ObservationSpec、固定 schema、批编码、长度校验 |
| `learning/actor_checkpoint.py`（新增） | SFT、MAPPO、推理共享的 actor 保存与加载 |
| `examples/qwen_action/collect.py`（新增） | 采集无人机专家轨迹，输出结构化观测与连续动作标签 |
| `examples/qwen_action/sft.py`（新增） | 本地 PyTorch 连续动作监督微调入口，不调用 swift sft |
| `examples/qwen_action/infer.py`（新增） | 批量观测推理入口，直接调用同一 QwenActor |
| `learning/mappo.py` | 注入可切换 actor、token rollout、小批梯度累积、检查点分支 |
| `examples/learn.py` | 构造配置和 ObservationSpec、保存路径、日志 |
| `examples/play.py` | 从检查点恢复 actor 与 tokenizer |
| `pyproject.toml`、`README.md` | 可选依赖和经过验证后的用法 |
| `tests/test_qwen_actor.py`（新增） | 不依赖真实模型的小型替身测试 |

实施阶段：

1. 先提取 MLPActor，验证现有 MLP 输出、动作概率和 checkpoint 不变，不改环境奖励。
2. 实现纯函数式 prompt 编码与严格长度检查，完成默认观测布局、完整历史和四种维度场景的测试。
3. 加入可注入 tokenizer/backbone 的 QwenActor，先用微型替身验证批处理、池化和梯度；真实 loader 与算法隔离。
4. 接入 rollout 的 token 保存、PPO 小批累积和版本化检查点，清除 Qwen 路径上的字符串推理/MSE/词表输出残留。
5. 增加本地专家数据采集、SFT、批推理入口，使用同一 actor 和检查点组件，完成 SFT → MAPPO → 推理的格式与权重交接检查。
6. 完成 CPU 静态与替身检查，提交开发结果；真实加载、延迟、显存和控制成功率列为待 GPU 验收事项。

## 11. 验收标准

本地可执行的检查不使用真实 0.8B 模型，也不启动实际训练：

- 编译/导入测试；MLP 分支在未安装可选 Qwen 依赖时可导入和实例化。
- 用调用计数替身确认每个控制步骨干只被调用一次，输入 batch=10，输出 `[10,4]`。
- 修改一架的观测只改变这一行 prompt 与均值；不使用 BatchNorm 或跨 batch 归一化。
- 验证 D=1/3/4；不允许 squeeze 丢掉 batch 或 action 维；默认 VEL 的输出确为四维。
- 变长输入、右 padding、不同 batch 组合的池化正确；批推理与逐条推理在关闭随机层时等价。
- 所有历史字段、目标误差和单位正确；NaN/Inf、超过字符/token 限制均明确失败；无静默 truncation。
- 用极小模型和固定张量执行一次合成损失 backward（不运行环境训练）：冻结层无梯度，末两层、动作头和 log_std 有有限梯度。
- 同权重、同 token 的 old/new log-prob 一致；稳定 tanh log-prob、逐智能体 ratio 和终止/超时语义保留。
- 合成确定性损失下，完整 batch 与加权小批累积的梯度一致；熵采样需固定噪声或在该对照测试关闭。
- 检查点往返、缺失基座、动作模式不匹配、输出层维度不匹配均有对应验证。

未来 GPU 验收单独进行：真实模型加载及参数审计、真实 tokenizer 的最坏长度测试、单步 batch=10 前向/反向、显存峰值、每步延迟、短程 PPO 数值检查，最后才开展多随机种子完整训练与 MLP 基线比较。
报告团队最终成功率、最差目标误差、首次全队连续悬停时间、失败率及仿真吞吐；不能只凭 reward 上升判定满足任务。

## 12. 交付边界

本方案可将语言模型作为连续控制 MAPPO 的共享 actor，并保留 MLP 切换路径。
是否学得更快、能否收敛、是否达到实时控制，需要后续 GPU 实验确认。
本地 action-head 示例证明的是标量监督回归接口；它不是已经训练好的无人机策略，也不能替代 MAPPO 的动作分布、PPO 更新和真实控制验收。

## 13. 从 ms-swift 迁入当前仓库并独立运行

### 13.1 迁移范围与依赖边界

最终部署和训练只需要当前仓库、Qwen 基座权重及明确的第三方 Python 依赖，不需要旁边存在 ms-swift 源码，不安装 `ms-swift`，不修改 `PYTHONPATH` 指向外部仓库。
ms-swift 仅作为一次性的实现参考和可选旧检查点来源。
独立于 ms-swift 不等于重新实现 Transformer：Qwen 文本模型和 tokenizer 仍由固定版本的 Transformers 提供，优化使用 PyTorch。

核心代码确实较少，但示例脚本不是完整训练器。迁移对应关系如下：

| 外部文件/能力 | 当前仓库中的处理 |
| --- | --- |
| `swift/model/patcher.py` 的四层 score MLP | 提取为本地动作头，末层由 1 改为 D；不复制整个 patcher |
| 同文件的末有效 token 池化 | 按本文 mask 池化重写，先池化再执行 MLP |
| 同文件替换 lm_head 和绑定 forward | 改为显式 QwenActor 包装，直接访问文本 hidden，不复制通用 monkey patch |
| `train/train.sh` 的冻结规则与训练参数 | 迁入本地配置；最后两层动态定位并验证，替换 `swift sft` 为本地 SFT 入口 |
| `infer/infer.py` | 替换 Swift Engine、InferRequest 和字符串解析，调用本地 actor 批前向 |
| `data/generate_dataset.py` | 不迁入加法数据生成逻辑；为无人机任务增加专家轨迹采集 |
| RL 训练 | 此示例没有 RL 实现；复用当前仓库 MAPPO，不复制 Swift 的语言生成 RL 训练器 |

直接复制或改编源码时记录来源文件、版本和修改说明，保留原代码对应的署名与许可文件；本地参考仓库的 LICENSE 为 Apache-2.0。实施时同时核对该版本的 NOTICE，不把外部工程的历史路径变成运行依赖。

### 13.2 三条路径共享一份实现

```text
结构化局部观测 → ObservationPromptEncoder → QwenActor → mu [B,D]
                                            │
                      SFT：tanh(mu) 与专家动作计算监督损失
                      MAPPO：Normal(mu,std) → tanh → PPO 动作概率损失
                      推理：tanh(mu)，或显式选择随机采样
```

骨干加载、冻结规则、动作头、prompt、动作维度、检查点均只实现一次。
SFT 入口只负责数据、监督损失和优化循环；MAPPO 保留环境采样、GAE 和策略更新；推理入口只负责加载及输出数值。
不能分别维护「SFT 的 score」与「RL 的 action_head」两套参数命名和前向逻辑。

本地入口如下；已实现入口和示例配置见使用说明，训练命令留待 GPU 环境执行：

```text
python -m gym_pybullet_drones.examples.qwen_action.collect
python -m gym_pybullet_drones.examples.qwen_action.sft --config <配置文件>
python -m gym_pybullet_drones.examples.learn --multiagent true --actor_type qwen --actor_init <SFT检查点>
python -m gym_pybullet_drones.examples.qwen_action.infer --checkpoint <检查点> --observations <观测文件>
python -m gym_pybullet_drones.examples.play --multiagent true --model_path <MAPPO检查点>
```

`actor_init` 是初始化 actor，`resume` 才是恢复完整训练；两者互斥且不能混用。MLP 仍为默认分支，10 架的固定配置保持不变。

### 13.3 无人机 SFT 数据契约

这里的 SFT 是连续动作的监督微调/行为克隆，不是预测回答 token 的语言模型 SFT。
每条记录保存动作执行前的本机原始观测与专家动作；prompt 由共享编码器动态构造，不同时维护一份可能过期的手写 prompt。

```text
dataset_manifest.json:
  schema_version, observation_spec, action_type, action_dim,
  ctrl_freq, physics, expert_version, episode_split_seed

每条样本:
  episode_id, step, agent_index,
  observation: float[O],
  action: float[D]  # 实际传给 env.step 的 [-1,1] 控制量
```

agent_index 只用于追踪样本，不自动加入 actor prompt。
固定 10 架按同一时刻批量采集，再拆成各架样本；训练/验证按完整 episode 划分，避免同一轨迹的相邻时刻或不同无人机泄漏到验证集。
初始状态、目标、扰动的覆盖范围需要在采集配置中明确，并记录失败轨迹，不以默认固定场景的重复记录代替多样数据。

默认 VEL 专家的最小实现可用目标误差与速度反馈生成世界系期望速度，并限制其模长至 SPEED_LIMIT：

```text
v_des = limit_norm(kp * goal_delta - kd * velocity, SPEED_LIMIT)
action[:3] = v_des / ||v_des||，速度为零时设为零
action[3] = ||v_des|| / SPEED_LIMIT，位于 [0,1]
```

这是数据采集用的候选专家，需未来仿真验证，不能称为已验证最优控制器。
优先实现 VEL 数据采集；RPM/PID 等模式只有在提供相应标签生成器时才开放采集，不能把真实电机 RPM 数值当成 [-1,1] 的 VEL 标签。
必须先保存 obs_t、计算 a_t，再 step；动作历史来自实际已执行动作，禁止把 a_t 提前拼进 obs_t。

### 13.4 本地 SFT 训练器

使用 PyTorch Dataset/DataLoader、共享 encoder/actor 和简单优化循环，首版不依赖 Swift Trainer，也不同时引入通用分布式训练框架。
动态 padding、长度限制、冻结范围与 MAPPO 完全一致。训练样本可跨 episode 混合组成 batch；每条依然只含本机观测。

监督目标定义为有界动作空间上的 MSE：

```text
predicted_action = tanh(actor(encoded_observation))
loss_sft = mean((predicted_action - expert_action)^2)
```

actor 返回的仍是无界 mu，tanh 在损失外部仅应用一次。不能在 SFT 中把输出层训练成直接动作值、到 RL 中又错误解释为经过 tanh 前的均值。
采用有界动作 MSE 可处理标签 ±1，不需要对这些标签取发散的 atanh。
首版不通过 SFT 拟合探索方差，log_std 保持预设初值，在 MAPPO 阶段再训练。
记录动作 MSE、各维误差和验证集误差；监督误差小不等价于闭环控制成功。

SFT 优化器只包含文本末两层和动作头；使用独立的骨干/动作头学习率，小批累积按实际样本数加权，支持保存最佳验证模型和最终模型。
用 `eval()` 关闭随机层仍允许反向传播，与 MAPPO 约束相同。训练入口不得调用 `generate()` 或传入语言 token labels。

### 13.5 SFT → MAPPO → 推理的检查点衔接

细化第 9 节的格式：抽取 `actor_checkpoint.py`，让 SFT 与 MAPPO 保存相同的 actor 部分。
manifest 增加 `stage=sft|mappo` 和 `format_version`；公共内容为文本可训练层、score、log_std、基座标识、tokenizer 和 ObservationSpec。
SFT 检查点不创建无意义的 critic；MAPPO 检查点额外保存 critic、MAPPO 配置与步数。optimizer 状态按阶段分别保存。

初始化 MAPPO 时：

1. 加载 SFT actor，校验基座、D、动作类型、prompt/schema、量化精度和控制频率。
2. 完整恢复已训练文本末层和动作头，保留 SFT 阶段默认 log_std。
3. 新建集中式 critic 和 MAPPO 优化器，不加载 SFT 优化器状态；从新 rollout 开始。
4. 同一观测下，初始化前后 `tanh(mu)` 应一致；初次 PPO 更新前 ratio 应接近 1。

若从 MAPPO 检查点恢复训练，则严格恢复 actor、critic、优化器和计数；若仅推理，公共 actor 加载器不需要构造 critic。
Qwen 权重文件独立于外部 ms-swift 目录：下载/导出到显式模型缓存目录并记录来源，不在 manifest 写入必须存在的兄弟仓库路径。

旧 Swift 标量检查点只通过一次性转换入口导入，不让日常 SFT/RL/推理保留 Swift 加载分支。
转换器只支持已审计的权重格式和参数名映射，直接读取配置与权重；列出哪些层被载入、哪些因维度不同被重新初始化。对于未知包装或不匹配基座明确拒绝，不能任意忽略键。
原加法标量头到 VEL 四维头是部分初始化，不是恢复训练；不继承旧优化器或宣称保持原输出。

### 13.6 独立性与迁移验收

在未安装 ms-swift、外部仓库不在 sys.path 的环境中完成：

- 静态扫描新增运行代码不存在 `import swift`、`from swift`、`swift sft`、`TransformersEngine`、硬编码兄弟仓库路径或 sys.path 注入。
- 三个入口可 `--help`，公共组件可导入；只有 Qwen 分支才需要 Transformers 可选依赖。
- 用微型替身 actor 和合成数据验证监督损失、冻结梯度、SFT 保存/加载、MAPPO 初始化后的确定性输出一致。
- 用微型替身确认独立推理与 MAPPO.predict 使用同一动作变换；10 条输入仅一次骨干调用。
- 数据拆分按 episode，obs/action 时序及 VEL 标签映射有单元测试。
- 不保留无用途的加法脚本、Swift shell 启动器、语言生成 RL 配置或重复输出头。

上述本地验收不启动真实训练、不下载或加载 0.8B 权重。真实 tokenizer、模型加载以及 SFT/RL 闭环性能留给 GPU 环境验证。
本节保留迁移设计依据；当前代码采用独立重写，来源记录、已交付内容和待 GPU 验证项见使用说明。
