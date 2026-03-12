"""Gymnasium wrapper around TorcsEnv"""

from __future__ import annotations

import sys
import pathlib
import traceback

import gymnasium as gym
import numpy as np
from gymnasium import spaces

_PARENT_DIR = str(pathlib.Path(__file__).resolve().parent.parent)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)


class TorcsGymnasiumWrapper(gym.Env):
    """Makes TorcsEnv compatible with Gymnasium / Stable-Baselines3."""

    metadata: dict = {"render_modes": []}
    _OBS_FIELDS: dict = {
        "focus": (0, 5),
        "speed": (5, 8),
        "speedX": (5, 6),
        "speedY": (6, 7),
        "speedZ": (7, 8),
        "opponents": (8, 44),
        "rpm": (44, 45),
        "track": (45, 64),
        "wheel": (64, 68),
        "angle": (68, 69),
        "trackPos": (69, 70),
    }

    def __init__(
        self, vision=False, throttle=False, gear_change=False, obs_set: str = "full"
    ):
        super().__init__()

        try:
            from gym_torcs import TorcsEnv
        except Exception as exc:
            print(
                "\n[ERROR] Cannot import TorcsEnv. Is gym_torcs.py in the parent folder?"
            )
            raise SystemExit(1) from exc

        """try:
            self._env = TorcsEnv(vision=vision, throttle=throttle, gear_change=gear_change)
        except Exception as exc:
            print("\n[ERROR] TORCS failed to launch.")
            print("  • Is 'torcs' installed and on $PATH?")
            print("  • Is $DISPLAY set?")
            raise SystemExit(1) from exc"""

        self._env = TorcsEnv(vision=vision, throttle=throttle, gear_change=gear_change)

        self._vision = vision
        self._inspected = False
        self._episode_count = 0
        self.relaunch_every: int = 20

        self.obs_set: str = obs_set
        self._last_steer: float = 0.0

        self.action_space = self._env.action_space
        if not hasattr(self.action_space, "low") or not hasattr(
            self.action_space, "high"
        ):
            raise TypeError(
                f"action_space must have low/high bounds, "
                f"got {type(self.action_space)}"
            )
        print(
            f"[TorcsWrapper] action_space bounds: "
            f"low={self.action_space.low}  high={self.action_space.high}"
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._estimate_obs_dim(),),
            dtype=np.float32,
        )

    # Gymnasium API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._episode_count += 1
        relaunch = (
            self.relaunch_every > 0 and (self._episode_count % self.relaunch_every) == 0
        )

        max_reset_attempts = 3
        for attempt in range(1, max_reset_attempts + 1):
            try:
                raw = self._env.reset(relaunch=relaunch)
                break  # success
            except Exception as exc:
                print(
                    f"[TorcsWrapper] reset attempt {attempt}/{max_reset_attempts} "
                    f"failed: {exc}"
                )
                traceback.print_exc()

                if attempt == max_reset_attempts:
                    print("[TorcsWrapper] All reset attempts exhausted. Exiting.")
                    raise SystemExit(1) from exc

                # Kill, relaunch TORCS, then retry
                try:
                    if hasattr(self._env, "reset_torcs"):
                        self._env.reset_torcs()
                    else:
                        self._env.end()
                except Exception:
                    traceback.print_exc()

                # Next iteration will try reset(relaunch=False) since TORCS
                # was just relaunched by reset_torcs / launch_torcs.
                relaunch = False

        obs = self._flatten(raw)
        obs = self._select_obs(obs)
        self._last_steer = 0.0
        self._inspect_once(obs, "reset")
        return obs, {}

    def step(self, action):
        action = self._clip(action)
        self._last_steer = float(action[0])  # steer is first action dim
        raw, reward, done, info = self._env.step(action)
        obs = self._flatten(raw)
        obs = self._select_obs(obs)  # reduce observations
        self._inspect_once(obs, "step")

        if info.get("terminal_reason") == "server_shutdown":
            return obs, 0.0, False, True, info

        return obs, float(reward), bool(done), False, info

    def close(self):
        try:
            self._env.end()
        except Exception:
            pass

    def request_relaunch(self) -> None:
        self._env._force_relaunch_next_reset = True

    # Observation flattening

    @staticmethod
    def _flatten(obs) -> np.ndarray:
        if isinstance(obs, np.ndarray):
            return obs.ravel().astype(np.float32)
        if hasattr(obs, "_fields"):
            parts = [
                np.atleast_1d(np.asarray(getattr(obs, f), dtype=np.float32)).ravel()
                for f in obs._fields
            ]
        elif isinstance(obs, dict):
            parts = [
                np.atleast_1d(np.asarray(v, dtype=np.float32)).ravel()
                for v in obs.values()
            ]
        elif isinstance(obs, (list, tuple)):
            parts = [
                np.atleast_1d(np.asarray(x, dtype=np.float32)).ravel() for x in obs
            ]
        else:
            parts = [np.atleast_1d(np.asarray(obs, dtype=np.float32)).ravel()]
        return np.concatenate(parts).astype(np.float32)

    def _clip(self, action) -> np.ndarray:
        return np.clip(
            np.asarray(action, dtype=np.float32),
            self.action_space.low,
            self.action_space.high,
        )

    # Helpers

    def _estimate_obs_dim(self) -> int:
        if self._vision:
            return 70 + 64 * 64 * 3

        obs_set = getattr(self, "obs_set", "full")
        if obs_set == "basic":
            # track(19) + speedX(1) + speedY(1) + angle(1) + trackPos(1) + steer(1)
            return 24

        if obs_set == "no_opponents":
            # focus(5)+speed(3)+rpm(1)+track(19)+wheel(4)+angle(1)+trackPos(1)+steer(1)
            return 35
        return 70

    def _inspect_once(self, obs: np.ndarray, tag: str) -> None:
        if self._inspected:
            return
        self._inspected = True
        print(
            f"\n[TorcsWrapper] obs inspection ({tag}): "
            f"shape={obs.shape}  dtype={obs.dtype}  "
            f"min={obs.min():.4f}  max={obs.max():.4f}  "
            f"action_space={self.action_space}\n"  # noqa: E501
        )
        expected = self.observation_space.shape[0]
        actual = obs.shape[0]
        if actual != expected:
            raise ValueError(
                f"[TorcsWrapper] obs dim mismatch! "
                f"expected={expected} actual={actual}. "
                f"Fix _estimate_obs_dim() or obs_set config."
            )

    def _obs_field(self, obs: np.ndarray, name: str) -> np.ndarray:
        """Slice a named field from the flat obs vector via *_OBS_FIELDS*."""
        lo, hi = self._OBS_FIELDS[name]
        return obs[lo:hi]

    def _select_obs(self, obs: np.ndarray) -> np.ndarray:
        """Optionally drop observation groups (non-vision only)."""
        if self._vision:
            return obs

        obs_set = getattr(self, "obs_set", "full")
        if obs_set == "full":
            return obs

        steer = np.array([self._last_steer], dtype=np.float32)

        if obs_set == "basic":
            return np.concatenate(
                [
                    self._obs_field(obs, "track"),
                    self._obs_field(obs, "speedX"),
                    self._obs_field(obs, "speedY"),
                    self._obs_field(obs, "angle"),
                    self._obs_field(obs, "trackPos"),
                    steer,
                ]
            ).astype(np.float32)

        if obs_set == "no_opponents":
            return np.concatenate(
                [
                    self._obs_field(obs, "focus"),
                    self._obs_field(obs, "speed"),
                    self._obs_field(obs, "rpm"),
                    self._obs_field(obs, "track"),
                    self._obs_field(obs, "wheel"),
                    self._obs_field(obs, "angle"),
                    self._obs_field(obs, "trackPos"),
                    steer,
                ]
            ).astype(np.float32)

        return obs
