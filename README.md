> [!TIP]
> 如需开展**符号动力学与约束**方向的研究，也可使用 [`safe-control-gym`](https://github.com/learnsyslab/safe-control-gym)。
>
> 如需 GPU 加速的**可微 JAX 仿真**，也可使用 [`crazyflow`](https://github.com/learnsyslab/crazyflow)。
>
> 如需部署到真实设备并使用 **PX4/ArduPilot + ROS2 + JetPack**，请使用 [`aerial-autonomy-stack`](https://github.com/JacopoPan/aerial-autonomy-stack)。

# gym-pybullet-drones

这是对原始 `gym-pybullet-drones` 仓库的精简重构，兼容 [`gymnasium`](https://github.com/Farama-Foundation/Gymnasium)、[`stable-baselines3` 2.0](https://github.com/DLR-RM/stable-baselines3/pull/1327) 和 [`betaflight`](https://github.com/betaflight/betaflight) SITL。

> **新闻**：`gym-pybullet-drones` 入选 [GitHub 2026 Maintainer Spotlight](https://maintainermonth.github.com/academia/gym-pybullet-drones-maintainer-spotlight)。

> **说明**：如需访问原始 IROS 2021 代码，请执行 `git checkout [paper|master]`。

<img src="gym_pybullet_drones/assets/helix.gif" alt="编队飞行" width="325"> <img src="gym_pybullet_drones/assets/helix.png" alt="控制信息" width="425">

## 安装

已在 Intel x64/Ubuntu 24.04 和 Apple Silicon/macOS 26 上测试。

```sh
git clone https://github.com/learnsyslab/gym-pybullet-drones.git
cd gym-pybullet-drones/

conda create -n drones python=3.12
conda activate drones

# Python 3.10 以上版本没有 `pybullet` 的预编译二进制包
# Ubuntu 上请安装 `gcc`，让 `pip3 install` 编译 `pybullet`
sudo apt install build-essential
# macOS 上请使用以下命令构建并安装 `pybullet`
CFLAGS="-Dfdopen=fdopen" pip install pybullet --no-cache-dir

pip3 install -e .

# 登录 SwanLab；训练指标会自动上传到此平台
swanlab login

# 用 `conda list` 查看已安装包；用 `conda deactivate` 退出环境；用 `conda remove -n drones --all` 删除环境
```

## 使用

### 控制示例

```sh
cd gym_pybullet_drones/examples/
python3 pid.py
python3 pid_velocity.py
python3 mrac.py
```

### 下洗气流示例

```sh
cd gym_pybullet_drones/examples/
python3 downwash.py
```

### 强化学习示例（PPO 和 MAPPO）

```sh
cd gym_pybullet_drones/examples/

# 单智能体：单架无人机在 z == 1.0 悬停
python learn.py
LATEST_MODEL=$(ls -t results | head -n 1) && python play.py --model_path "results/${LATEST_MODEL}/best_model.zip"

# 多智能体：在脚本中修改训练参数后，以 MLP 启动 MAPPO
bash train_mappo_mlp.sh

# 回放：EVAL_FREQ 大于 0 时使用 best_model；设为 0 时将其替换为 final_model
python play.py --multiagent true --model_path "../../results/<RUN_NAME>/save-时间戳/best_model.pt"

# 多智能体：安装可选依赖，在脚本中修改参数后，以 Qwen3.5 启动 MAPPO
python -m pip install -e "../..[qwen]"
bash train_mappo_qwen.sh
```

MAPPO 使用共享 actor，它只能看到自身的运动学状态、动作历史和目标相对位移。训练时，critic 可见全部无人机的观测；执行时只需要 actor。实现采用按智能体裁剪的策略比率、共享的平均团队奖励、GAE、裁剪价值损失和有界的 tanh 高斯动作。时间限制会从最后一个观测进行自举估计，失败终止不会。这是 [MAPPO](https://github.com/marlbenchmark/on-policy) 集中训练、分散执行方法的前馈实现。

MAPPO 任务始终仿真 10 架无人机，默认通过现有 PID 控制器使用三维速度动作。使用 `--act rpm` 直接控制电机，或使用 `--act one_d_rpm` 仅进行垂直飞行。默认目标为初始位置加上 `1 / (i + 1)` 米的垂直位移。自定义 `(10, 3)` 目标可通过 `target_positions` 传给 `learn.run()` 或 `MultiHoverAviary`。

稠密奖励鼓励更早到达和较低速度；悬停奖励要求位置误差不超过 5 cm、速度不超过 0.1 m/s、角速度不超过 0.2 rad/s、横滚/俯仰角不超过 0.1 rad。全部无人机必须连续一秒满足这些条件才算成功，并持续悬停至 8 秒时间限制结束。高度低于 2 cm、横滚/俯仰超过 0.4 rad，或距目标超过 3 m 都会使回合因失败而终止。启用评估时，多智能体会在 `evaluations.npz` 中记录团队奖励、最终是否成功、最差最终距离和连续最终悬停时间。单智能体评估仅记录回合奖励，不填充悬停指标。

`best_model.pt` 和 `final_model.pt` 包含网络、优化器状态以及回放所需的环境设置。最优模型按评估回报选择。请在 `train_mappo_mlp.sh` 或 `train_mappo_qwen.sh` 中选择 actor、设备（如 `cuda:0`、`cuda:1` 或 `cpu`）和训练超参数，并用 `RUN_NAME` 标记实验。输出路径为仓库根目录下的 `results/<RUN_NAME>/save-时间戳/`。将 `EVAL_FREQ` 设为 `0` 可跳过评估；此时只保存 `final_model`，不生成 `best_model` 和 `evaluations.npz`。训练会自动向 SwanLab 上传训练损失和评估指标；在脚本中通过 `SWANLAB_PROJECT`、`SWANLAB_WORKSPACE` 和 `SWANLAB_MODE` 设置目标项目、工作空间与上传模式。MAPPO 的时间步计数单位是联合仿真步，而非单个智能体动作。评估使用独立环境和固定的确定性回合；应使用不同随机种子重复训练以评估鲁棒性。已有的多无人机 PPO 检查点与新的目标感知观测不兼容；单无人机 PPO 仍使用 `.zip` 检查点。

<img src="gym_pybullet_drones/assets/rl.gif" alt="强化学习示例" width="375"> <img src="gym_pybullet_drones/assets/marl.gif" alt="多智能体强化学习示例" width="375">

### 运行全部测试

```sh
# 在仓库根目录执行
cd gym-pybullet-drones/
pytest tests/
```

### Betaflight SITL 示例（仅 Ubuntu）

```sh
# 一次性设置：在仓库根目录中为每架无人机构建一个 SITL 可执行文件（例如 2 架）；如有需要，先执行 `apt install curl`
cd gym-pybullet-drones/
./gym_pybullet_drones/assets/clone_bfs.sh 2

# 运行示例
cd gym_pybullet_drones/examples/
python3 beta.py --num_drones 2
# --num_drones 必须不大于传给 clone_bfs.sh 的数量
```

## 引用

如使用本项目，请引用我们的 [IROS 2021 论文](https://arxiv.org/abs/2103.02142)（[原始代码](https://github.com/learnsyslab/gym-pybullet-drones/tree/paper)）：

```bibtex
@INPROCEEDINGS{panerati2021learning,
      title={Learning to Fly---a Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control}, 
      author={Jacopo Panerati and Hehui Zheng and SiQi Zhou and James Xu and Amanda Prorok and Angela P. Schoellig},
      booktitle={2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
      year={2021},
      volume={},
      number={},
      pages={7512-7519},
      doi={10.1109/IROS51168.2021.9635857}
}
```

## 可选的 Qwen 共享 actor

MAPPO 保留默认 MLP actor，并可选支持共享的 Qwen3.5-0.8B 连续动作 actor、本地行为克隆和批量推理，无需 ms-swift。参见[安装、命令与验证边界](docs/qwen_actor_usage.md)。目前仅完成微型 CPU 替身测试，真实模型和 GPU 训练尚未验证。

## 参考资料

- Erwin Coumans and Yunfei Bai (2023) [*PyBullet Quickstart Guide*](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit?tab=t.0#heading=h.2ye70wns7io3)
- Carlos Luis and Jerome Le Ny (2016) [*Design of a Trajectory Tracking Controller for a Nanoquadcopter*](https://arxiv.org/pdf/1608.05786.pdf)
- Nathan Michael, Daniel Mellinger, Quentin Lindsey, Vijay Kumar (2010) [*The GRASP Multiple Micro-UAV Testbed*](https://ieeexplore.ieee.org/document/5569026)
- Benoit Landry (2014) [*Planning and Control for Quadrotor Flight through Cluttered Environments*](http://groups.csail.mit.edu/robotics-center/public_papers/Landry15)
- Julian Forster (2015) [*System Identification of the Crazyflie 2.0 Nano Quadrocopter*](https://www.research-collection.ethz.ch/handle/20.500.11850/214143)
- Antonin Raffin, Ashley Hill, Maximilian Ernestus, Adam Gleave, Anssi Kanervisto, and Noah Dormann (2019) [*Stable Baselines3*](https://github.com/DLR-RM/stable-baselines3)
- Guanya Shi, Xichen Shi, Michael O’Connell, Rose Yu, Kamyar Azizzadenesheli, Animashree Anandkumar, Yisong Yue, and Soon-Jo Chung (2019)
  [*Neural Lander: Stable Drone Landing Control Using Learned Dynamics*](https://arxiv.org/pdf/1811.08027.pdf)
- C. Karen Liu and Dan Negrut (2020) [*The Role of Physics-Based Simulators in Robotics*](https://www.annualreviews.org/doi/pdf/10.1146/annurev-control-072220-093055)
- Yunlong Song, Selim Naji, Elia Kaufmann, Antonio Loquercio, and Davide Scaramuzza (2020) [*Flightmare: A Flexible Quadrotor Simulator*](https://arxiv.org/pdf/2009.00563.pdf)

---

> UTIAS / [Learning Systems and Robotics Lab](https://github.com/learnsyslab) / [Vector Institute](https://github.com/VectorInstitute) / 剑桥大学 [Prorok Lab](https://github.com/proroklab)

<!--
## 待办事项

- [ ] 在 `BaseAviary._dynamics()` 中实现缓冲区，以加入电机延迟和更高级的 ESC 建模
- [ ] 修改 `BaseAviary._updateAndStoreKinematicInformation()`、`BaseAviary._getDroneStateVector()` 和相关子类的 `.computeObs()`，用四元数替换 `rpy`（并用机体系角速度替换 `ang_vel`）

## 故障排查

- 在配有 NVIDIA 显卡的 Ubuntu 上，若出现 “Failed to create and OpenGL context”，请启动 `nvidia-settings`，在 “PRIME Profiles” 中选择 “NVIDIA (Performance Mode)”，重启后重试。
-->
