import gymnasium as gym  # stablebaselines expects gymnasium instead of gym
from gymnasium import spaces
import numpy as np

# from os import path
import snakeoil3_gym as snakeoil3
import copy
import collections as col
import os


class TorcsEnv:
    terminal_judge_start = 200
    termination_limit_progress = 5
    default_speed = 50
    centerline_deadband = 0.15
    offcenter_weight = 0.35
    curve_clearance_threshold = 0.72
    curve_speed_weight = 0.75
    low_progress_patience = 60
    low_progress_curve_grace = 0.35
    corner_progress_min = 1.0

    initial_reset = True

    def __init__(self, vision=False, throttle=False, gear_change=False):
        self.vision = vision
        self.throttle = throttle
        self.gear_change = gear_change
        self.last_u = None
        self.last_steer = 0
        self.initial_run = True
        self._force_relaunch_next_reset = False
        self._low_progress_steps = 0

        ##print("launch torcs")
        snakeoil3.launch_torcs(self.vision)

        n_act = 1 + (1 if self.throttle else 0) + (1 if self.gear_change else 0)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_act,), dtype=np.float32
        )

        if vision is False:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(70,), dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(70 + 64 * 64 * 3,), dtype=np.float32
            )

    def step(self, u):
        client = getattr(self, "client", None)
        if client is None or getattr(client, "so", None) is None:
            # Server already shut down (race finished). End episode so SB3 calls reset().
            self._force_relaunch_next_reset = True
            return self.get_obs(), 0.0, True, {"terminal_reason": "server_shutdown"}

        client = self.client

        this_action = self.agent_to_torcs(u)

        # Apply Action
        action_torcs = client.R.d

        # Steering
        action_torcs["steer"] = this_action["steer"]  # in [-1, 1]

        #  Simple Autnmatic Throttle Control by Snakeoil
        if self.throttle is False:
            target_speed = self.default_speed
            if client.S.d["speedX"] < target_speed - (client.R.d["steer"] * 50):
                client.R.d["accel"] += 0.01
            else:
                client.R.d["accel"] -= 0.01

            if client.R.d["accel"] > 0.2:
                client.R.d["accel"] = 0.2

            if client.S.d["speedX"] < 10:
                client.R.d["accel"] += 1 / (client.S.d["speedX"] + 0.1)

            # Re-clamp after the low-speed boost
            if client.R.d["accel"] > 0.2:
                client.R.d["accel"] = 0.2

            # Traction Control System
            if (client.S.d["wheelSpinVel"][2] + client.S.d["wheelSpinVel"][3]) - (
                client.S.d["wheelSpinVel"][0] + client.S.d["wheelSpinVel"][1]
            ) > 5:
                action_torcs["accel"] -= 0.2
        else:
            raw_accel = this_action["accel"]  # in [-1, 1]
            # Asymmetric split: 85 % of the range is throttle, 15 % is brake.
            #   [-1, -0.7) → brake   [0 … 1]
            #   [-0.7,  1] → throttle [0 … 1]
            if raw_accel >= -0.7:
                action_torcs["accel"] = (raw_accel + 0.7) / 1.7  # [-0.7,1] → [0,1]
                action_torcs["brake"] = 0
            else:
                action_torcs["accel"] = 0
                action_torcs["brake"] = (-raw_accel - 0.7) / 0.3  # [-1,-0.7) → [0,1]

        #  Automatic Gear Change by Snakeoil
        if self.gear_change is True:
            action_torcs["gear"] = this_action["gear"]
        else:
            #  Automatic Gear Change by Snakeoil is possible
            action_torcs["gear"] = 1

            if client.S.d["speedX"] > 50:
                action_torcs["gear"] = 2
            if client.S.d["speedX"] > 80:
                action_torcs["gear"] = 3
            if client.S.d["speedX"] > 110:
                action_torcs["gear"] = 4
            if client.S.d["speedX"] > 140:
                action_torcs["gear"] = 5
            if client.S.d["speedX"] > 170:
                action_torcs["gear"] = 6

        # Save the privious full-obs from torcs for the reward calculation
        obs_pre = copy.deepcopy(client.S.d)

        # One-Step Dynamics Update #################################
        # Apply the Agent's action into torcs
        client.respond_to_server()
        # Get the response of TORCS
        client.get_servers_input()

        if getattr(client, "so", None) is None:
            # We just received ***shutdown*** inside get_servers_input()
            self._force_relaunch_next_reset = True
            return self.get_obs(), 0.0, True, {"terminal_reason": "server_shutdown"}

        # Get the current full-observation from torcs
        obs = client.S.d

        # Make an obsevation from a raw observation vector from TORCS
        self.observation = self.make_observaton(obs)

        # Reward setting Here #######################################
        angle = float(obs["angle"])
        speed_x = float(obs["speedX"])
        track_pos = float(obs["trackPos"])
        track = np.asarray(obs["track"], dtype=np.float32)
        track_pos_c = float(np.clip(track_pos, -1.0, 1.0))
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        abs_track_pos = abs(track_pos)

        forward = speed_x * cos_a
        side = abs(speed_x * sin_a)
        center_error = max(0.0, abs(track_pos_c) - self.centerline_deadband)
        offcenter = speed_x * center_error

        reward = ((forward * 1.8) - 0.8 * side - self.offcenter_weight * offcenter) / 100.0
        reward -= 0.01  # small step cost to discourage stalling

        # Steering smoothness penalty
        current_steer = float(this_action["steer"])
        steer_change = abs(current_steer - self.last_steer)
        reward -= 0.04 * steer_change
        self.last_steer = current_steer

        # Penalize carrying too much speed into sharp bends (anticipatory corner signal).
        track_norm = np.clip(track / 200.0, 0.0, 1.0)
        forward_clearance = (
            float(track_norm[9]) if track_norm.size > 9 else float(track_norm.mean())
        )
        curve_risk = float(
            np.clip(
                (self.curve_clearance_threshold - forward_clearance)
                / self.curve_clearance_threshold,
                0.0,
                1.0,
            )
        )
        p_curve_speed = (
            self.curve_speed_weight * curve_risk * (max(speed_x, 0.0) / 100.0) ** 2
        )
        reward -= p_curve_speed

        # --- Collision penalty ---
        p_collision = 0.0
        damage_delta = float(obs["damage"] - obs_pre["damage"])
        if damage_delta > 0:
            p_collision = 1.0
            reward -= p_collision

        p_brake = 0.0
        if self.throttle:
            brake = float(action_torcs.get("brake", 0.0))
            # Don't penalize braking much when the forward track sensor indicates a corner.
            straight_factor = float(
                np.clip((forward_clearance - 0.50) / 0.50, 0.0, 1.0)
            )
            p_brake = 0.05 * brake * straight_factor
            reward -= p_brake

        # Termination judgement
        raw_progress = forward

        episode_terminate = False
        terminal_reason = None
        p_terminal = 0.0
        # Strict off-track rule: terminate as soon as car leaves track bounds.
        if abs_track_pos > 1.0:
            p_terminal = 8.0
            reward -= p_terminal
            episode_terminate = True
            terminal_reason = "off_track"
            client.R.d["meta"] = True

        if not episode_terminate and self.terminal_judge_start < self.time_step:
            # Avoid false terminations while cautiously navigating tight corners.
            progress_floor = self.termination_limit_progress
            if curve_risk >= self.low_progress_curve_grace:
                progress_floor = self.corner_progress_min

            if raw_progress < progress_floor:
                self._low_progress_steps += 1
            else:
                self._low_progress_steps = 0

            if self._low_progress_steps >= self.low_progress_patience:
                episode_terminate = True
                terminal_reason = "low_progress"
                client.R.d["meta"] = True

        if not episode_terminate and cos_a < 0:
            # Episode is terminated if the agent runs backward.
            episode_terminate = True
            terminal_reason = "backward"
            client.R.d["meta"] = True

        if client.R.d["meta"] is True:  # Send a reset signal
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1

        info = {
            "terminal_reason": terminal_reason,
            "reward_total": float(reward),
            "forward": float(forward),
            "side": float(side),
            "offcenter": float(offcenter),
            "forward_clearance": float(forward_clearance),
            "curve_risk": float(curve_risk),
            "p_collision": float(p_collision),
            "p_curve_speed": float(p_curve_speed),
            "p_brake": float(p_brake),
            "p_terminal": float(p_terminal),
            "raw_progress": float(raw_progress),
            "speed_x": speed_x,
            "track_pos": track_pos,
            "track_pos_clipped": track_pos_c,
            "damage_delta": damage_delta,
            "low_progress_steps": int(self._low_progress_steps),
            "episode_terminate": bool(episode_terminate),
        }
        return self.get_obs(), reward, client.R.d["meta"], info

    def reset(self, relaunch=False):
        self.time_step = 0
        self._low_progress_steps = 0

        # Force relaunch if the previous episode ended because TORCS shut down
        if getattr(self, "_force_relaunch_next_reset", False):
            snakeoil3.launch_torcs(self.vision)
            self._force_relaunch_next_reset = False
            self.initial_reset = True  # avoid sending meta on a dead client

        if self.initial_reset is not True:
            self.client.R.d["meta"] = True
            self.client.respond_to_server()

            if relaunch is True:
                snakeoil3.launch_torcs(self.vision)
                print("### TORCS is RELAUNCHED ###")

        self.client = snakeoil3.Client(p=3001, vision=self.vision)
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        self.observation = self.make_observaton(obs)

        self.last_u = None

        self.initial_reset = False
        return self.get_obs()

    def reset_torcs(self):
        """Kill/relaunch TORCS and force the next reset() to behave like a fresh start."""
        try:
            # Close UDP socket cleanly if present
            if hasattr(self, "client") and self.client is not None:
                try:
                    self.client.shutdown()
                except Exception:
                    pass
        except Exception:
            pass

        # launch_torcs() already taskkills wtorcs.exe in your snakeoil3_gym.py
        snakeoil3.launch_torcs(self.vision)

        # Make reset() do the full handshake again
        self.initial_reset = True

    """def end(self):
        cwd = os.getcwd()
        os.chdir('torcs')
        os.system('taskkill /f /im wtorcs.exe')
        os.chdir(cwd)"""

    def end(self):
        # No need to chdir; taskkill works from anywhere
        try:
            os.system("taskkill /f /im wtorcs.exe")
        except Exception:
            pass

    def get_obs(self):
        return self.observation

    def agent_to_torcs(self, u):
        a = np.asarray(u, dtype=np.float32).ravel()
        idx = 0
        torcs_action = {"steer": float(a[idx])}
        idx += 1

        if self.throttle:
            torcs_action["accel"] = float(a[idx])
            idx += 1

        if self.gear_change:
            gear_raw = float(a[idx])  # [-1, 1]
            gear = int(np.clip(np.round(((gear_raw + 1.0) / 2.0) * 5.0) + 1, 1, 6))
            torcs_action["gear"] = gear

        return torcs_action

    def obs_vision_to_image_rgb(self, obs_image_vec):
        arr = np.asarray(obs_image_vec, dtype=np.uint8)
        arr = arr[:12288]
        return arr.reshape(64 * 64, 3)

    def make_observaton(self, raw_obs):
        if self.vision is False:
            names = [
                "focus",
                "speedX",
                "speedY",
                "speedZ",
                "opponents",
                "rpm",
                "track",
                "wheelSpinVel",
                "angle",
                "trackPos",
            ]
            Observation = col.namedtuple("Observaion", names)
            return Observation(
                focus=np.array(raw_obs["focus"], dtype=np.float32) / 200.0,
                speedX=np.array(raw_obs["speedX"], dtype=np.float32)
                / self.default_speed,
                speedY=np.array(raw_obs["speedY"], dtype=np.float32)
                / self.default_speed,
                speedZ=np.array(raw_obs["speedZ"], dtype=np.float32)
                / self.default_speed,
                opponents=np.array(raw_obs["opponents"], dtype=np.float32) / 200.0,
                rpm=np.array(raw_obs["rpm"], dtype=np.float32) / 10000.0,
                track=np.array(raw_obs["track"], dtype=np.float32) / 200.0,
                wheelSpinVel=np.array(raw_obs["wheelSpinVel"], dtype=np.float32)
                / 100.0,
                angle=np.array(raw_obs["angle"], dtype=np.float32) / np.pi,
                trackPos=np.array(raw_obs["trackPos"], dtype=np.float32),
            )
        else:
            names = [
                "focus",
                "speedX",
                "speedY",
                "speedZ",
                "opponents",
                "rpm",
                "track",
                "wheelSpinVel",
                "angle",
                "trackPos",
                "img",
            ]
            Observation = col.namedtuple("Observaion", names)

            # Get RGB from observation
            image_rgb = self.obs_vision_to_image_rgb(raw_obs["img"])

            return Observation(
                focus=np.array(raw_obs["focus"], dtype=np.float32) / 200.0,
                speedX=np.array(raw_obs["speedX"], dtype=np.float32)
                / self.default_speed,
                speedY=np.array(raw_obs["speedY"], dtype=np.float32)
                / self.default_speed,
                speedZ=np.array(raw_obs["speedZ"], dtype=np.float32)
                / self.default_speed,
                opponents=np.array(raw_obs["opponents"], dtype=np.float32) / 200.0,
                rpm=np.array(raw_obs["rpm"], dtype=np.float32) / 10000.0,
                track=np.array(raw_obs["track"], dtype=np.float32) / 200.0,
                wheelSpinVel=np.array(raw_obs["wheelSpinVel"], dtype=np.float32)
                / 100.0,
                angle=np.array(raw_obs["angle"], dtype=np.float32) / np.pi,
                trackPos=np.array(raw_obs["trackPos"], dtype=np.float32),
                img=image_rgb,
            )
