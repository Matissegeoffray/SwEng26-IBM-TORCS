# TORCS RL – ML Skeleton

Train / evaluate an RL agent on TORCS with Stable-Baselines3.

## Layout

```
gym_torcs/
├── gym_torcs.py          ← TorcsEnv
└── ml/
    ├── config.yaml
    ├── wrappers.py       ← Gymnasium wrapper
    ├── utils.py          ← factories + helpers
    ├── train.py
    ├── evaluate.py       ← eval + demo (--episodes 1)
    ├── requirements.txt
    └── README.md
```

## Setup

```bash
cd gym_torcs/ml
pip install -r requirements.txt
```

## Train

```bash
python train.py
python train.py --timesteps 50000 --algorithm SAC
```

Logs → `logs/` (TensorBoard), checkpoints → `checkpoints/`.

## Evaluate

```bash
python evaluate.py                          # 10 episodes → results.json
python evaluate.py --episodes 1             # quick demo
python evaluate.py --model checkpoints/torcs_rl_50000_steps.zip --episodes 20
```

## Adding algorithms

Add to `_ALGO_MAP` in `utils.py` and `_ALGO_LOAD` in `evaluate.py`, then set `train.algorithm` in `config.yaml`.
