> [!TIP]
> For research work with **symbolic dynamics and constraints**, also try [`safe-control-gym`](https://github.com/learnsyslab/safe-control-gym)
>
> For GPU-accelerated, **differentiable, JAX-based simulation**, also try [`crazyflow`](https://github.com/learnsyslab/crazyflow)
>
> For real-world deployment of **PX4/ArduPilot + ROS2 + JetPack**, use [`aerial-autonomy-stack`](https://github.com/JacopoPan/aerial-autonomy-stack)

# gym-pybullet-drones

This is a minimalist refactoring of the original `gym-pybullet-drones` repository, designed for compatibility with [`gymnasium`](https://github.com/Farama-Foundation/Gymnasium), [`stable-baselines3` 2.0](https://github.com/DLR-RM/stable-baselines3/pull/1327), and [`betaflight`](https://github.com/betaflight/betaflight) SITL.

> **NEWS**: `gym-pybullet-drones` was featured in [GitHub's Maintainer Spotlight 2026](https://maintainermonth.github.com/academia/gym-pybullet-drones-maintainer-spotlight)

> **NOTE**: if you want to access the original IROS 2021 codebase, please `git checkout [paper|master]`

<img src="gym_pybullet_drones/assets/helix.gif" alt="formation flight" width="325"> <img src="gym_pybullet_drones/assets/helix.png" alt="control info" width="425">

## Installation

Tested on Intel x64/Ubuntu 24.04 and Apple Silicon/macOS 26.

```sh
git clone https://github.com/learnsyslab/gym-pybullet-drones.git
cd gym-pybullet-drones/

conda create -n drones python=3.12
conda activate drones

# Beyond Python 3.10, `pybullet` has no pre-built wheel
# On Ubuntu, install `gcc` to let `pip3 install` build `pybullet`
sudo apt install build-essential
# On macOS, build and install `pybullet` with
CFLAGS="-Dfdopen=fdopen" pip install pybullet --no-cache-dir

pip3 install -e .

# check installed packages with `conda list`, deactivate with `conda deactivate`, remove with `conda remove -n drones --all`
```

## Use

### Control examples

```sh
cd gym_pybullet_drones/examples/
python3 pid.py
python3 pid_velocity.py
python3 mrac.py
```

### Downwash effect example

```sh
cd gym_pybullet_drones/examples/
python3 downwash.py
```

### Reinforcement learning examples (PPO and MAPPO)

```sh
cd gym_pybullet_drones/examples/

# single agent, task: single drone hover at z == 1.0
python learn.py
LATEST_MODEL=$(ls -t results | head -n 1) && python play.py --model_path "results/${LATEST_MODEL}/best_model.zip"

# multi-agent: decentralized actors with a centralized team critic (MAPPO)
python learn.py --multiagent true --num_drones 2 --total_timesteps 1000000 --gui false --plot false
LATEST_MODEL=$(ls -t results | head -n 1) && python play.py --multiagent true --model_path "results/${LATEST_MODEL}/best_model.pt"
```

MAPPO uses a shared actor that sees only its own kinematics, action history,
and target displacement. The critic sees all drones' observations during
training; execution needs only the actor. It uses per-agent clipped policy
ratios, a shared mean team reward, GAE, clipped value loss, and bounded tanh
Gaussian actions. Time limits bootstrap from the final observation; failures
do not. This is a feed-forward implementation of the centralized-training,
decentralized-execution approach in [MAPPO](https://github.com/marlbenchmark/on-policy).

Multi-drone training defaults to 3D velocity actions through the existing PID
controller. Use `--act rpm` for direct motor control or `--act one_d_rpm` for
vertical-only flight. The default targets are the initial positions plus
`1 / (i + 1)` meters vertically. Custom `(num_drones, 3)` targets can be passed
as `target_positions` to `learn.run()` or `MultiHoverAviary`.

The dense reward favors early arrival and low speed; a hover bonus requires
position error <= 5 cm, speed <= 0.1 m/s, angular speed <= 0.2 rad/s, and
roll/pitch <= 0.1 rad. All drones must meet these conditions continuously for
one second to count as successful, and must continue hovering until the
8-second time limit. Altitude below 2 cm, roll/pitch beyond 0.4 rad, or distance
from the assigned target beyond 3 m ends an episode as failure.
Multi-agent evaluation records team reward, final success,
worst final distance, and consecutive final hover time in `evaluations.npz`.
Single-agent evaluation records episode reward without placeholder hover metrics.

`best_model.pt` and `final_model.pt` include network/optimizer states and the
environment settings used for replay. The best model is selected by evaluation
return. `--seed`, `--rollout_steps`, `--batch_size`, `--epochs`, `--eval_freq`, and
`--device` configure training. MAPPO timesteps count joint simulation steps,
not individual agent actions. Evaluation uses a separate environment and a
fixed deterministic episode; repeat training with different seeds to assess
robustness. Existing multi-drone PPO checkpoints are incompatible with the
new target-aware observations. Single-drone PPO still uses `.zip` checkpoints.

<img src="gym_pybullet_drones/assets/rl.gif" alt="rl example" width="375"> <img src="gym_pybullet_drones/assets/marl.gif" alt="marl example" width="375">

### Run all tests

```sh
# from the repo's top folder
cd gym-pybullet-drones/
pytest tests/
```

### Betaflight SITL example (Ubuntu only)

```sh
# one-time setup: from the repo's top folder, build one SITL executable per drone (e.g. 2), if needed, `apt install curl`
cd gym-pybullet-drones/
./gym_pybullet_drones/assets/clone_bfs.sh 2

# run the example
cd gym_pybullet_drones/examples/
python3 beta.py --num_drones 2
# --num_drones must be <= the number passed to clone_bfs.sh
```

## Citation

If you wish, please cite our [IROS 2021 paper](https://arxiv.org/abs/2103.02142) ([and original codebase](https://github.com/learnsyslab/gym-pybullet-drones/tree/paper)) as

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

## References

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

-----
> UTIAS / [Learning Systems and Robotics Lab](https://github.com/learnsyslab) / [Vector Institute](https://github.com/VectorInstitute) / University of Cambridge's [Prorok Lab](https://github.com/proroklab)

<!--
## TODOs

- [ ] Add motor delay, advanced ESC modeling by implementing a buffer in `BaseAviary._dynamics()`
- [ ] Replace `rpy` with quaternions (and `ang_vel` with body rates) by editing `BaseAviary._updateAndStoreKinematicInformation()`, `BaseAviary._getDroneStateVector()`, and the `.computeObs()` methods of relevant subclasses

## Troubleshooting

- On Ubuntu, with an NVIDIA card, if you receive a "Failed to create and OpenGL context" message, launch `nvidia-settings` and under "PRIME Profiles" select "NVIDIA (Performance Mode)", reboot and try again.
-->
