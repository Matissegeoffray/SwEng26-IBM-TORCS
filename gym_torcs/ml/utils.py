"""Shared helpers: config, algorithm factory, env builder, metrics."""

from __future__ import annotations

import pathlib
import re
from typing import Any

import yaml
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from wrappers import TorcsGymnasiumWrapper

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_DEFAULT_CFG = _SCRIPT_DIR / "config.yaml"


class RetainedCheckpointCallback(CheckpointCallback):
    """Checkpoint callback that keeps only the most recent step files."""

    def __init__(self, *args, max_step_checkpoints: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_step_checkpoints = max(1, int(max_step_checkpoints))

    def _on_step(self) -> bool:
        should_prune = self.save_freq > 0 and self.n_calls % self.save_freq == 0
        continue_training = super()._on_step()
        if should_prune:
            self._prune_step_checkpoints()
        return continue_training

    def _on_training_start(self) -> None:
        self._prune_step_checkpoints()

    def _prune_step_checkpoints(self) -> None:
        checkpoint_dir = pathlib.Path(self.save_path)
        filename_re = re.compile(rf"^{re.escape(self.name_prefix)}_(\d+)_steps$")

        checkpoints: list[tuple[int, pathlib.Path]] = []
        for path in checkpoint_dir.glob(f"{self.name_prefix}_*_steps.zip"):
            match = filename_re.match(path.stem)
            if match:
                checkpoints.append((int(match.group(1)), path))

        checkpoints.sort(key=lambda item: item[0], reverse=True)
        stale = checkpoints[self.max_step_checkpoints :]
        for _, path in stale:
            try:
                path.unlink()
                print(f"[utils] Deleted old checkpoint -> {path}")
            except OSError as exc:
                print(f"[utils] WARNING: Failed to delete {path}: {exc}")


def _resolve_path(rel: str) -> pathlib.Path:
    """Resolve a path relative to the ml/ directory, not cwd."""
    return _SCRIPT_DIR / rel


def load_config(path: str | pathlib.Path | None = None) -> dict[str, Any]:
    p = pathlib.Path(path) if path else _DEFAULT_CFG
    with open(p, "r") as fh:
        return yaml.safe_load(fh)


def make_env(cfg: dict[str, Any]) -> TorcsGymnasiumWrapper:
    ec = cfg.get("env", {})
    env = TorcsGymnasiumWrapper(
        vision=ec.get("vision", False),
        throttle=ec.get("throttle", False),
        gear_change=ec.get("gear_change", False),
        obs_set=ec.get("obs_set", "full"),
    )
    env.relaunch_every = ec.get("relaunch_every", 3)
    return env


def make_normalized_env(cfg: dict[str, Any]):
    """Create a TORCS env, optionally wrapped in DummyVecEnv + VecNormalize."""
    tc = cfg.get("train", {})
    if not tc.get("normalize_obs", True) and not tc.get("normalize_reward", True):
        return Monitor(make_env(cfg))
    vec_env = DummyVecEnv([lambda _cfg=cfg: Monitor(make_env(_cfg))])
    return VecNormalize(
        vec_env,
        norm_obs=tc.get("normalize_obs", True),
        norm_reward=tc.get("normalize_reward", True),
        clip_obs=10.0,
    )


def save_normalizer(env, path) -> None:
    """Save VecNormalize statistics to *path* (no-op for plain envs)."""
    if isinstance(env, VecNormalize):
        env.save(str(path))
        print(f"[utils] Saved normalizer stats -> {path}")


def load_normalizer(env, path) -> None:
    """Restore VecNormalize statistics from *path* into *env*."""
    p = pathlib.Path(path)
    if isinstance(env, VecNormalize) and p.exists():
        loaded = VecNormalize.load(str(p), env.venv)
        env.obs_rms = loaded.obs_rms
        env.ret_rms = loaded.ret_rms
        print(f"[utils] Loaded normalizer stats <- {path}")


def request_env_relaunch(env) -> None:
    """Schedule a TORCS relaunch to happen on the next env.reset()"""
    inner = env.venv if isinstance(env, VecNormalize) else env
    if hasattr(inner, "envs"):
        for e in inner.envs:
            if hasattr(e, "request_relaunch"):
                e.request_relaunch()
                print("[utils] TORCS relaunch scheduled for next block.")
                return
    if hasattr(env, "request_relaunch"):
        env.request_relaunch()
        print("[utils] TORCS relaunch scheduled for next block.")


# Algorithm factory — add new algos here
_ALGO_MAP: dict[str, type] = {"PPO": PPO, "SAC": SAC}


# New make_algorithm to continue from best_model if it exists
def make_algorithm(cfg: dict[str, Any], env):
    tc = cfg.get("train", {})
    name = tc.get("algorithm", "PPO").upper()

    if name not in _ALGO_MAP:
        raise ValueError(f"Unknown algorithm '{name}'. Available: {list(_ALGO_MAP)}")

    AlgoClass = _ALGO_MAP[name]

    checkpoint_dir = _resolve_path(tc.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    latest_model_path = checkpoint_dir / "best_model.zip"
    replay_buffer_path = checkpoint_dir / "best_model_replay_buffer.pkl"

    # Build keyword args from config
    net_arch = tc.get("net_arch", [256, 256])
    algo_kwargs = dict(
        policy="MlpPolicy",
        env=env,
        seed=tc.get("seed"),
        verbose=1,
        tensorboard_log=str(_resolve_path(tc.get("log_dir", "logs"))),
        learning_rate=tc.get("learning_rate", 3e-4),
        gamma=tc.get("gamma", 0.99),
        batch_size=tc.get("batch_size", 256),
        policy_kwargs=dict(net_arch=net_arch),
    )
    # SAC-specific args
    if name == "SAC":
        algo_kwargs["buffer_size"] = tc.get("buffer_size", 100_000)
        algo_kwargs["learning_starts"] = tc.get("learning_starts", 2000)
        algo_kwargs["tau"] = tc.get("tau", 0.005)
        algo_kwargs["ent_coef"] = tc.get("ent_coef", "auto")
        algo_kwargs["target_entropy"] = tc.get("target_entropy", "auto")
        algo_kwargs["gradient_steps"] = tc.get("gradient_steps", 1)

    if latest_model_path.exists():
        print(f"[train] Resuming from {latest_model_path}")
        model = AlgoClass.load(
            latest_model_path,
            env=env,
            learning_rate=tc.get("learning_rate", 3e-4),
            gamma=tc.get("gamma", 0.99),
            batch_size=tc.get("batch_size", 256),
        )
        if name == "SAC":
            model.tau = tc.get("tau", 0.005)
            model.gradient_steps = tc.get("gradient_steps", 1)
            te = tc.get("target_entropy", "auto")
            if te == "auto":
                model.target_entropy = -float(env.action_space.shape[0])  # -2.0
            else:
                model.target_entropy = float(te)

        # Restore replay buffer so we don't lose all past experience
        if replay_buffer_path.exists() and hasattr(model, "load_replay_buffer"):
            print(f"[train] Loading replay buffer from {replay_buffer_path}")
            model.load_replay_buffer(replay_buffer_path)
            print(f"[train] Replay buffer size: {model.replay_buffer.size()}")
        elif hasattr(model, "replay_buffer"):
            print(
                "[train] WARNING: No replay buffer found — starting with empty buffer!"
            )

        # Restore observation normalizer if available
        norm_path = checkpoint_dir / "obs_normalizer.pkl"
        load_normalizer(env, norm_path)
    else:
        print("[train] Starting new training run")
        model = AlgoClass(**algo_kwargs)

    return model


def make_checkpoint_cb(cfg: dict[str, Any]) -> CheckpointCallback:
    tc = cfg.get("train", {})
    return RetainedCheckpointCallback(
        save_freq=tc.get("checkpoint_freq", 10_000),
        save_path=str(_resolve_path(tc.get("checkpoint_dir", "checkpoints"))),
        name_prefix="torcs_rl",
        max_step_checkpoints=tc.get("checkpoint_keep_last", 10),
    )


def extract_metrics(info: dict, episode_length: int, total_reward: float) -> dict:
    return {
        "episode_length": episode_length,
        "total_reward": round(total_reward, 4),
        "terminal_reason": info.get("terminal_reason", None),
        "reward_total_last": info.get("reward_total", None),
        "raw_progress_last": info.get("raw_progress", None),
        "on_track_factor_last": info.get("on_track_factor", None),
        "r_dist_progress_last": info.get("r_dist_progress", None),
        "r_progress_last": info.get("r_progress", None),
        "r_speed_bonus_last": info.get("r_speed_bonus", None),
        "r_track_prox_last": info.get("r_track_prox", None),
        "p_heading_last": info.get("p_heading", None),
        "p_low_speed_last": info.get("p_low_speed", None),
        "p_offtrack_last": info.get("p_offtrack", None),
        "p_time_last": info.get("p_time", None),
        "p_reverse_last": info.get("p_reverse", None),
        "p_steer_last": info.get("p_steer", None),
        "p_lateral_last": info.get("p_lateral", None),
        "p_collision_last": info.get("p_collision", None),
        "p_terminal_last": info.get("p_terminal", None),
        "track_pos_last": info.get("track_pos", None),
        "speed_x_last": info.get("speed_x", None),
        "dist_delta_last": info.get("dist_delta", None),
        "damage_delta_last": info.get("damage_delta", None),
        "lap_time": info.get("lap_time", None),  # TODO: populate when available
        "off_track": info.get("terminal_reason") == "off_track",
        "damage": info.get("damage", info.get("damage_delta", None)),
    }
