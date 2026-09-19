# 本地 Qwen actor 使用说明

已实现 MLP/Qwen 切换、完整观测 prompt、共享批量 actor、MAPPO token 缓存与梯度累积、专家数据采集、连续动作 SFT、独立推理和检查点衔接。不需要安装 ms-swift 或保留外部仓库。

## 验证边界

本地只做微型 CPU 替身测试与非训练环境检查，没有下载或加载 Qwen3.5-0.8B，没有实际 SFT/RL 训练。真实模型加载、tokenizer 长度覆盖、BF16、GPU 显存和闭环效果仍需 GPU 环境验收；不能保证实时 30 Hz。

可选适配器按 Transformers **5.3.0** 源码实现并固定版本，未声称这个组合已通过真实模型运行验证。检查点记录实际 Torch、Transformers、tokenizers、safetensors 版本。MLP 不导入 Transformers，兼容原来的 `.pt` 检查点。

本次验证：41 项 CPU 替身及非训练回归测试通过，2 项实际训练测试排除；编译和差异检查通过。覆盖 BF16 计算副本/FP32 梯度、小学习率更新累积、SFT 初始化学习率覆盖、检查点往返、MLP 随机状态恢复、换卡恢复的 CUDA RNG 调用及推理预处理。CUDA RNG 测试替换了显卡接口，不代表真实 GPU 验证。

```powershell
python -m pytest tests/test_qwen_actor.py tests/test_mappo.py -k "not training_and_playback and not ppo_final_evaluation" -q --basetemp=tmp/qwen-tests
```

## 未来 GPU 环境操作

以下为已实现的入口，不代表本机已运行训练。首先将官方 Qwen3.5-0.8B 权重、配置及 tokenizer 放入显式本地目录，例如 `D:/models/Qwen3.5-0.8B`。加载器只读本地文件，不自动下载。使用 safetensors；本地配置和所有权重分片的 SHA256 固定基座身份。

```powershell
python -m pip install -e ".[qwen]"
python -m gym_pybullet_drones.examples.qwen_action.collect --output results/qwen_data --episodes 20
python -m gym_pybullet_drones.examples.qwen_action.sft --config docs/qwen_sft.example.json
python -m gym_pybullet_drones.examples.learn --multiagent true --actor_type qwen --actor_init results/qwen_sft/best_model --device cuda --gui false --plot false
```

采集仅支持 VEL 候选 PD 专家。数据记录实际执行动作前的观测，按 episode 拆分验证集，保留失败轨迹；随机初始位置和目标范围写入数据 manifest。SFT 是 `MSE(tanh(mu), action)`，不是文本回答训练；较小的监督误差不等于闭环成功。

从基座直接进行 MAPPO，或恢复完整 MAPPO 训练：

```powershell
python -m gym_pybullet_drones.examples.learn --multiagent true --actor_type qwen --model_path D:/models/Qwen3.5-0.8B --backbone_dtype bfloat16 --device cuda --update_microbatch_steps 1 --gui false --plot false
python -m gym_pybullet_drones.examples.learn --multiagent true --resume results/<run>/final_model --device cuda --gui false --plot false
```

`actor_init` 只恢复 actor，创建新 critic/优化器；`resume` 恢复 MAPPO 参数、优化器、计数和 RNG，两者互斥。恢复从新 episode 开始，不恢复 PyBullet 中途状态。resume 恢复原环境配置，禁止覆盖 `act` 和目标。SFT 的 `actor_init` 同样只是初始化，不恢复 SFT 优化器。

旧 MLP `.pt` 检查点仍可加载，但未保存 RNG 的旧文件无法恢复原随机序列。

新检查点仅保存训练显卡的 CUDA RNG，恢复时应用到 `--device` 指定的显卡。旧版保存全部显卡 RNG 的检查点未记录训练显卡，需沿用原逻辑显卡编号恢复。

SFT 使用 `actor_init` 时，骨干、prompt 和观测语义沿用检查点；新优化器采用本次 JSON `actor` 中的 `actor_backbone_lr`、`actor_head_lr`（省略时分别为 `1e-5`、`1e-4`），并将实际学习率写入新检查点，不继承旧训练的学习率。

默认动作仍为 VEL 四维，10 架无人机共享一套骨干和动作头，每个控制步将 10 条 prompt 整批前向一次。训练微批单位是**联合环境步**；默认 Qwen 每个微批 1 步，即 10 条序列。骨干最后两层、动作头及 MAPPO `log_std` 可训练；前三者之外的骨干参数冻结。默认骨干、头、critic 学习率分别为 `1e-5`、`1e-4`、`3e-4`，可通过 Python 配置对象调整。

`backbone_dtype=bfloat16` 指骨干计算精度，不是可训练参数的存储精度：末两层保留 FP32 主权重，前向通过 `torch.func.functional_call` 使用可微 BF16 临时副本。SFT/MAPPO 的 Adam 直接更新 FP32 参数，动量及保存权重也是 FP32，不需要额外主权重同步或两套优化器。冻结骨干仍保持原加载精度，动作头/动作分布保持 FP32。旧 BF16 权重可载入 FP32 参数继续训练，但此前因舍入已丢失的更新无法恢复；真实 GPU 算子兼容性仍待验证。

默认 MLP 用法不变：

```powershell
python -m gym_pybullet_drones.examples.learn --multiagent true --actor_type mlp --gui false --plot false
```

## 推理与文件

```powershell
python -m gym_pybullet_drones.examples.qwen_action.infer --checkpoint results/qwen_sft/best_model --observations observations.json --device cuda
python -m gym_pybullet_drones.examples.play --multiagent true --model_path results/<run>/best_model --device cuda --gui false --plot false
```

`observations.json` 是数值数组 `[B,O]`，默认 VEL 的 O=75，包括全部 15 行历史及目标差值。推理输出数值 `[B,D]`，不生成语言。回放从 checkpoint 恢复环境，并检查动作类型、控制频率、观测布局和智能体数量。

Qwen 检查点目录包含 `manifest.json`、`tokenizer/`、`trainable.pt` 和训练状态 `optimizer.pt`。仅保存可训练骨干层和动作头；冻结基座仍需单独保留。模型目录移动后，训练/推理用 `--model_path`、回放用 `--base_model_path` 指定新位置，文件 hash 必须相同。仅推理不加载 optimizer。

prompt 固定三位小数（负零规范化），完整保留字段及历史，超过 1024 字符或含特殊 token 的 1024 tokens 就报错，不截断。高控制频率可能超过上限，需要显式升级 schema，而不是隐式丢历史。

## 来源与未纳入部分

参考 ms-swift 提交 `2596a34422a01622e36cd467f3836c88d139936d` 的 `swift/model/patcher.py` 及 `examples/train/seq_cls/qwen3_5_action_head` 的结构设计：四层 Linear、三层 SiLU、最后两个 decoder block 可训练。参考仓库署名为 ModelScope Contributors，许可证 Apache-2.0。这里根据设计文档独立重写，不复制 patcher、Trainer、推理引擎或示例源码；没有引入其运行时依赖。

模型适配结构依据 [Transformers v5.3.0 Qwen3.5 源码](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py)。首次加载条件生成包装后仅保留 `model.language_model`，不执行视觉分支或词表头；首次加载仍需容纳完整 checkpoint 的 CPU 内存。

本次未提供可选的旧 Swift 标量检查点转换器：不接受未审计包装或静默跳过尺寸错误，也不将加法回归权重当作无人机策略。默认从官方基座构造 D 维新头，SFT/RL/推理均使用本仓库格式。需要旧权重迁移时应先确定具体导出格式再增加严格的一次性转换器。
