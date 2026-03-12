"""Evaluate a saved model"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from utils import (
    load_config,
    make_normalized_env,
    extract_metrics,
    load_normalizer,
)

_ALGO_LOAD: dict[str, type] = {"PPO": PPO, "SAC": SAC}


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate TORCS RL agent")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    ec = cfg["eval"]
    _script_dir = pathlib.Path(__file__).resolve().parent
    model_path = (
        pathlib.Path(args.model)
        if args.model
        else _script_dir / pathlib.Path(ec["model_path"])
    )

    num_episodes = args.episodes if args.episodes is not None else ec["num_episodes"]
    results_path = args.output or ec.get("results_path", "results.json")
    algo_name = cfg["train"]["algorithm"].upper()

    print(f"[eval] model={model_path}  algo={algo_name}  episodes={num_episodes}")

    env = make_normalized_env(cfg)
    # Load normalizer stats and set eval mode
    checkpoint_dir = cfg["train"].get("checkpoint_dir", "checkpoints")
    norm_path = _script_dir / checkpoint_dir / "obs_normalizer.pkl"
    load_normalizer(env, norm_path)
    if isinstance(env, VecNormalize):
        env.training = False
        env.norm_reward = False

    model = _ALGO_LOAD.get(algo_name, PPO).load(model_path, env=env)

    is_vec = isinstance(env, VecEnv)
    episode_metrics: list[dict] = []
    for ep in range(1, num_episodes + 1):
        res = env.reset()
        obs = res[0] if isinstance(res, tuple) else res
        done, total_reward, length = False, 0.0, 0
        info: dict = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if is_vec:
                obs, rew, done_arr, infos = env.step(action)
                total_reward += float(rew[0])
                length += 1
                done = bool(done_arr[0])
                info = infos[0]
            else:
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                length += 1
                done = terminated or truncated
        m = extract_metrics(info, length, total_reward)
        episode_metrics.append(m)
        print(f"  Ep {ep}/{num_episodes}: reward={m['total_reward']:.2f}  len={length}")

    env.close()

    rewards = [m["total_reward"] for m in episode_metrics]
    lengths = [m["episode_length"] for m in episode_metrics]
    summary = {
        "num_episodes": num_episodes,
        "avg_reward": round(statistics.mean(rewards), 4),
        "avg_length": round(statistics.mean(lengths), 2),
        "std_reward": round(statistics.stdev(rewards), 4) if len(rewards) > 1 else 0.0,
        "episodes": episode_metrics,
    }

    out = pathlib.Path(results_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n[eval] Results → {out}")
    print(
        f"[eval] avg_reward={summary['avg_reward']:.2f}  avg_length={summary['avg_length']:.1f}"
    )


if __name__ == "__main__":
    main()
