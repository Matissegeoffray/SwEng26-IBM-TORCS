#!/usr/bin/env python3
"""Train an RL agent on TORCS.
Usage: python train.py [--timesteps N] [--algorithm SAC]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import warnings

import numpy as np

from utils import (
    load_config,
    make_normalized_env,
    make_algorithm,
    make_checkpoint_cb,
    save_normalizer,
    request_env_relaunch,
)

from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecEnv, VecNormalize


def evaluate_model(model, env, n_episodes=1):
    """Evaluate *model* and return the mean reward"""
    is_vec = isinstance(env, VecEnv)
    rewards: list[float] = []
    prev_training = None
    prev_norm_reward = None
    if isinstance(env, VecNormalize):
        prev_training = env.training
        prev_norm_reward = env.norm_reward
        env.training = False
        env.norm_reward = False

    for ep in range(n_episodes):
        res = env.reset()
        obs = res[0] if isinstance(res, tuple) else res
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if is_vec:
                obs, rew, done_arr, _ = env.step(action)
                total_reward += float(rew[0])
                done = bool(done_arr[0])
            else:
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated

        rewards.append(total_reward)
        print(f"[eval] Episode {ep + 1}: {total_reward:.2f}")

    # Restore normalizer training mode
    if prev_training is not None:
        env.training = prev_training
        env.norm_reward = prev_norm_reward

    mean_reward = float(np.mean(rewards))
    print(f"[eval] Mean reward over {n_episodes} episodes: {mean_reward:.2f}")
    return mean_reward


def main() -> None:
    p = argparse.ArgumentParser(description="Train RL agent on TORCS")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--algorithm", type=str, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)

    if args.timesteps is not None:
        cfg["train"]["total_timesteps"] = args.timesteps
    if args.algorithm is not None:
        cfg["train"]["algorithm"] = args.algorithm

    tc = cfg["train"]
    seed = tc.get("seed")
    if seed is None:
        warnings.warn(
            "No seed set in config — training will not be reproducible.",
            stacklevel=2,
        )
    else:
        random.seed(seed)
        np.random.seed(seed)
        set_random_seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
        except ImportError:
            pass

    print(
        f"[train] algo={tc['algorithm']}  "
        f"steps/block={tc['total_timesteps']}  "
        f"seed={seed}"
    )

    # --- Environments ---
    env = make_normalized_env(cfg)

    # --- Model ---
    model = make_algorithm(cfg, env)

    # --- Callbacks ---
    cb_checkpoint = make_checkpoint_cb(cfg)
    cb = CallbackList([cb_checkpoint])

    # --- Output directory ---
    _script_dir = pathlib.Path(__file__).resolve().parent
    out = _script_dir / tc["checkpoint_dir"]
    out.mkdir(parents=True, exist_ok=True)

    # Persist best_mean_reward across restarts so we never overwrite
    # a genuinely good model with a worse one after restarting training.
    meta_path = out / "training_meta.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
        best_mean_reward = meta.get("best_mean_reward", -float("inf"))
        print(f"[train] Loaded previous best_mean_reward = {best_mean_reward:.2f}")
    else:
        best_mean_reward = -float("inf")

    max_blocks = tc.get("max_blocks")
    block = 1

    try:
        while max_blocks is None or block <= max_blocks:
            print(f"\n[train] Starting block {block}")

            # Relaunch TORCS once per block (except the very first — it's
            # already running from initialisation).
            if block > 1:
                request_env_relaunch(env)

            model.learn(
                total_timesteps=tc["total_timesteps"],
                callback=cb,
                reset_num_timesteps=False,
            )

            print(f"[train] Finished block {block}")

            # ---- Manual Evaluation ----
            mean_reward = evaluate_model(
                model, env, n_episodes=tc.get("n_eval_episodes", 1)
            )

            # Always save latest so we never lose progress
            latest_path = out / "latest_model.zip"
            model.save(latest_path)
            latest_replay_path = out / "latest_replay_buffer.pkl"
            if hasattr(model, "save_replay_buffer"):
                model.save_replay_buffer(latest_replay_path)
            latest_norm_path = out / "latest_normalizer.pkl"
            save_normalizer(env, latest_norm_path)

            # ---- Save best if improved ----
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward

                best_path = out / "best_model.zip"
                model.save(best_path)

                replay_buf_path = out / "best_model_replay_buffer.pkl"
                if hasattr(model, "save_replay_buffer"):
                    model.save_replay_buffer(replay_buf_path)

                norm_path = out / "obs_normalizer.pkl"
                save_normalizer(env, norm_path)

                meta_path = out / "training_meta.json"
                with open(meta_path, "w") as f:
                    json.dump({"best_mean_reward": best_mean_reward, "block": block}, f)

                print(f"[train] NEW BEST → {mean_reward:.2f}")
                print(f"[train] Saved → {best_path}")
            else:
                print(
                    f"[train] No improvement this block "
                    f"(best={best_mean_reward:.2f}).  Continuing from latest."
                )

            block += 1

    except KeyboardInterrupt:

        print("\n[train] Interrupted by user.")

    finally:

        print("[train] Closing environment.")
        env.close()


if __name__ == "__main__":
    main()
