# Offline Reinforcement Learning with OOD State Correction and OOD Action Suppression

Implementation of the SCAS algorithm.

## Environment
Paper results were collected with [MuJoCo 210](https://mujoco.org/) (and [mujoco-py 2.1.2.14](https://github.com/openai/mujoco-py)) in [OpenAI gym 0.23.1](https://github.com/openai/gym) with the [D4RL datasets](https://github.com/Farama-Foundation/D4RL). Networks are trained using [PyTorch 1.11.0](https://github.com/pytorch/pytorch) and [Python 3.7](https://www.python.org/).

## Usage

### Pretrained Models

We have uploaded pretrained dynamics models in SCAS_dynamics/ to facilitate experiment reproduction. 

You can also pretrain dynamics models by running:
```
./run_pretrain.sh
```

### Offline RL


The SCAS algorithm can be trained by running:
```
./run_experiments.sh
```

### Logging

This codebase uses tensorboard. You can view saved runs with:

```
tensorboard --logdir <run_dir>
```