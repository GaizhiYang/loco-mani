# Copyright (c) 2025-2026, Junjie Zhu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Play an RSL-RL checkpoint with keyboard-controlled commands."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import math
import os
import sys
import time
from typing import Any

import gymnasium as gym
import torch
from packaging import version

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Play an RSL-RL agent with keyboard command control.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playback.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=None,
    help="Ignored. Keyboard playback always forces num_envs=1.",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--command_name",
    type=str,
    default="both",
    choices=("base_velocity", "ee_pose", "both"),
    help="Command term controlled by the keyboard.",
)
parser.add_argument(
    "--command_rate",
    type=float,
    default=0.8,
    help="Base velocity command change per second while a key is held.",
)
parser.add_argument(
    "--ee_pos_rate",
    type=float,
    default=0.15,
    help="End-effector position change in meters per second while a key is held.",
)
parser.add_argument(
    "--ee_rot_rate",
    type=float,
    default=0.5,
    help="End-effector orientation change in radians per second while a key is held.",
)
parser.add_argument(
    "--gripper_rate",
    type=float,
    default=0.05,
    help="Gripper opening change in meters per second while a key is held.",
)
parser.add_argument(
    "--print_command_interval",
    type=float,
    default=0.5,
    help="Seconds between command value prints. Set to 0 to disable.",
)
parser.add_argument(
    "--box_usd",
    type=str,
    default=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../resource/object/box_usd/d1_piper_l.usd")
    ),
    help="Path to the USD box asset.",
)
parser.add_argument(
    "--box_distance",
    type=float,
    default=0.75,
    help="Distance from the robot root to the box center along the robot's forward direction.",
)
parser.add_argument(
    "--disable_box",
    action="store_true",
    help="Do not load the box USD asset.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

installed_version = metadata.version("rsl-rl-lib")

import carb
import omni.appwindow
import isaaclab.sim as sim_utils
from pxr import PhysxSchema, Usd, UsdPhysics

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectRLEnv,
    DirectRLEnvCfg,
    DirectMARLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import LeggedManip_Lab.tasks  # noqa: F401


def spawn_box_visual(
    prim_path: str,
    cfg: Any,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn the box as visual-only geometry.

    The generated USD contains URDF physics schemas and a fixed root joint.
    The box is only a visual prop in this playback script, so remove those
    schemas before Isaac Sim parses the physics scene.
    """
    prim = sim_utils.create_prim(
        prim_path=prim_path,
        usd_path=cfg.usd_path,
        translation=translation,
        orientation=orientation,
        scale=cfg.scale,
    )
    stage = sim_utils.get_current_stage()
    descendants = list(Usd.PrimRange(prim))
    for child in reversed(descendants):
        if "joint" in child.GetName().lower() or "joint" in child.GetTypeName().lower():
            stage.RemovePrim(child.GetPath())
            continue
        for schema in (
            UsdPhysics.RigidBodyAPI,
            UsdPhysics.CollisionAPI,
            UsdPhysics.MassAPI,
            UsdPhysics.ArticulationRootAPI,
            PhysxSchema.PhysxRigidBodyAPI,
            PhysxSchema.PhysxCollisionAPI,
            PhysxSchema.PhysxArticulationAPI,
        ):
            if child.HasAPI(schema):
                child.RemoveAPI(schema)
    return prim


def add_box_to_scene_cfg(env_cfg: Any, usd_path: str, distance: float) -> None:
    """Add the box to the scene config before Isaac Lab creates the simulation."""
    if distance < 0.0:
        raise ValueError("--box_distance must be non-negative.")
    if not os.path.isfile(usd_path):
        raise FileNotFoundError(f"Box USD file does not exist: {usd_path}")

    robot_init = env_cfg.scene.robot.init_state
    robot_pos = robot_init.pos
    robot_rot = robot_init.rot
    # Quaternion is in (w, x, y, z), and only the initial yaw is needed here.
    yaw = math.atan2(
        2.0 * (robot_rot[0] * robot_rot[3] + robot_rot[1] * robot_rot[2]),
        1.0 - 2.0 * (robot_rot[2] ** 2 + robot_rot[3] ** 2),
    )
    box_position = (
        float(robot_pos[0] + distance * math.cos(yaw)),
        float(robot_pos[1] + distance * math.sin(yaw)),
        0.0,
    )
    box_orientation = (
        float(math.cos(yaw / 2.0)),
        0.0,
        0.0,
        float(math.sin(yaw / 2.0)),
    )

    box_spawn_cfg = sim_utils.UsdFileCfg(usd_path=usd_path)
    box_spawn_cfg.func = spawn_box_visual
    env_cfg.scene.box = AssetBaseCfg(
        prim_path="/World/KeyboardBox",
        spawn=box_spawn_cfg,
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=box_position,
            rot=box_orientation,
        ),
    )
    print(
        f"[INFO] Loaded box: {usd_path}\n"
        f"[INFO] Box position: ({box_position[0]:+.3f}, {box_position[1]:+.3f}, {box_position[2]:+.3f}), "
        f"distance={distance:.3f} m in front of robot."
    )


class KeyboardCommandController:
    """Keyboard controller for base velocity and end-effector pose commands."""

    _BASE_KEY_TO_DELTA = {
        "UP": (1.0, 0.0, 0.0),
        "DOWN": (-1.0, 0.0, 0.0),
        "LEFT": (0.0, 1.0, 0.0),
        "RIGHT": (0.0, -1.0, 0.0),
        "NUMPAD_8": (1.0, 0.0, 0.0),
        "NUMPAD_2": (-1.0, 0.0, 0.0),
        "NUMPAD_4": (0.0, 1.0, 0.0),
        "NUMPAD_6": (0.0, -1.0, 0.0),
        "NUMPAD_7": (0.0, 0.0, 1.0),
        "NUMPAD_9": (0.0, 0.0, -1.0),
    }
    _EE_POS_KEY_TO_DELTA = {
        "W": (1.0, 0.0, 0.0),
        "S": (-1.0, 0.0, 0.0),
        "A": (0.0, 1.0, 0.0),
        "D": (0.0, -1.0, 0.0),
        "Q": (0.0, 0.0, 1.0),
        "E": (0.0, 0.0, -1.0),
    }
    _EE_ROT_KEY_TO_DELTA = {
        "Z": (1.0, 0.0, 0.0),
        "X": (-1.0, 0.0, 0.0),
        "T": (0.0, 1.0, 0.0),
        "G": (0.0, -1.0, 0.0),
        "C": (0.0, 0.0, 1.0),
        "V": (0.0, 0.0, -1.0),
    }
    _GRIPPER_KEY_TO_DELTA = {
        "R": 1.0,
        "F": -1.0,
    }

    def __init__(
        self,
        env: Any,
        command_name: str,
        command_rate: float,
        ee_pos_rate: float,
        ee_rot_rate: float,
        gripper_rate: float,
        print_interval: float,
    ):
        if command_rate < 0.0 or ee_pos_rate < 0.0 or ee_rot_rate < 0.0 or gripper_rate < 0.0:
            raise ValueError("Command rates must be non-negative.")
        if not hasattr(env, "command_manager"):
            raise RuntimeError(
                "The selected task has no command_manager. Use a manager-based task."
            )
        active_terms = env.command_manager.active_terms
        requested = {"base_velocity", "ee_pose"} if command_name == "both" else {command_name}
        missing = requested.difference(active_terms)
        if missing:
            raise RuntimeError(
                f"Command term(s) {', '.join(sorted(missing))!r} are not active. "
                f"Active command terms: {', '.join(active_terms) or '<none>'}"
            )

        self._base_term = env.command_manager.get_term("base_velocity") if "base_velocity" in requested else None
        self._ee_term = env.command_manager.get_term("ee_pose") if "ee_pose" in requested else None
        if self._base_term is not None and self._base_term.command.shape[1] < 3:
            raise RuntimeError("The base_velocity command must have at least 3 values.")
        if self._ee_term is not None and self._ee_term.command.shape[1] < 7:
            raise RuntimeError("The ee_pose command must have 7 values: xyz + quaternion.")

        self._command_rate = command_rate
        self._ee_pos_rate = ee_pos_rate
        self._ee_rot_rate = ee_rot_rate
        self._gripper_rate = gripper_rate
        self._print_interval = max(0.0, print_interval)
        self._pressed_keys: set[str] = set()
        self._last_update_time = time.monotonic()
        self._last_print_time = self._last_update_time
        self._last_printed_command: tuple[float, ...] | None = None

        if self._base_term is not None:
            self._base_lower, self._base_upper = self._get_velocity_limits()
            self._base_command = torch.zeros_like(self._base_term.command)
        else:
            self._base_lower = self._base_upper = self._base_command = None
        if self._ee_term is not None:
            self._ee_lower, self._ee_upper = self._get_pose_limits()
            self._ee_command = self._ee_term.command.clone()
            self._ee_euler = torch.stack(euler_xyz_from_quat(self._ee_command[:, 3:7]), dim=-1)
        else:
            self._ee_lower = self._ee_upper = self._ee_command = self._ee_euler = None

        self._robot = env.scene["robot"]
        self._gripper_joint_ids, self._gripper_joint_names = self._robot.find_joints(
            ["gripper", "gripper_joint1", "gripper_joint2"], preserve_order=True
        )
        if len(self._gripper_joint_ids) != 3:
            raise RuntimeError(
                "Expected gripper joints 'gripper', 'gripper_joint1', and 'gripper_joint2'; "
                f"found: {', '.join(self._gripper_joint_names) or '<none>'}."
            )
        self._gripper_joint_ids = torch.tensor(self._gripper_joint_ids, device=self._robot.device, dtype=torch.long)
        self._gripper_lower = self._robot.data.soft_joint_pos_limits[0, self._gripper_joint_ids, 0].clone()
        self._gripper_upper = self._robot.data.soft_joint_pos_limits[0, self._gripper_joint_ids, 1].clone()
        self._gripper_opening = torch.zeros(1, device=self._robot.device)

        appwindow = omni.appwindow.get_default_app_window()
        self._keyboard = appwindow.get_keyboard()
        self._input = carb.input.acquire_input_interface()
        self._keyboard_subscription = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_keyboard_event
        )

        print("[INFO] Keyboard command control enabled.")
        if self._base_term is not None:
            print("[INFO] Base velocity: Arrow keys/Numpad control x/y/yaw.")
        if self._ee_term is not None:
            print("[INFO] EE position: W/S=x, A/D=y, Q/E=z.")
            print("[INFO] EE orientation: Z/X=roll, T/G=pitch, C/V=yaw.")
        print("[INFO] Gripper: R=open, F=close.")
        print("[INFO] SPACE: reset keyboard command to the current target.")
        print("[INFO] Press ESC to close the simulator.")
        if self._base_term is not None:
            print(
                f"[INFO] Base limits: x={self._base_lower[0]:.3f}, "
                f"y={self._base_lower[1]:.3f}, yaw={self._base_lower[2]:.3f}"
            )

    def _get_ranges(self):
        ranges = getattr(self._base_term.cfg if self._base_term is not None else self._ee_term.cfg, "limit_ranges", None)
        if ranges is None:
            ranges = (self._base_term or self._ee_term).cfg.ranges
        return ranges

    def _get_velocity_limits(self) -> tuple[torch.Tensor, torch.Tensor]:
        ranges = getattr(self._base_term.cfg, "limit_ranges", None) or self._base_term.cfg.ranges
        names = ("lin_vel_x", "lin_vel_y", "ang_vel_z")
        bounds = []
        for name in names:
            value = getattr(ranges, name, None)
            if value is None or len(value) != 2:
                raise RuntimeError(f"Cannot read command range '{name}' from {type(ranges).__name__}.")
            bounds.append((float(value[0]), float(value[1])))
        lower = torch.tensor([item[0] for item in bounds], device=self._base_term.command.device)
        upper = torch.tensor([item[1] for item in bounds], device=self._base_term.command.device)
        return lower, upper

    def _get_pose_limits(self) -> tuple[torch.Tensor, torch.Tensor]:
        ranges = getattr(self._ee_term.cfg, "limit_ranges", None) or self._ee_term.cfg.ranges
        names = ("pos_x", "pos_y", "pos_z", "roll", "pitch", "yaw")
        bounds = []
        for name in names:
            value = getattr(ranges, name, None)
            if value is None or len(value) != 2:
                raise RuntimeError(f"Cannot read EE pose range '{name}' from {type(ranges).__name__}.")
            bounds.append((float(value[0]), float(value[1])))
        lower = torch.tensor([item[0] for item in bounds], device=self._ee_term.command.device)
        upper = torch.tensor([item[1] for item in bounds], device=self._ee_term.command.device)
        return lower, upper

    @staticmethod
    def _keyboard_key_name(input_value: Any) -> str:
        """Normalize Carb enum and string keyboard input values to one key name."""
        key = getattr(input_value, "name", input_value)
        key = str(key).strip().upper()
        # Isaac Sim 5.1 can provide either ``KeyboardInput.W`` or the plain
        # string ``W`` depending on the focused UI widget.
        if "." in key:
            key = key.rsplit(".", 1)[-1]
        if key.startswith("KEY_"):
            key = key[4:]
        return key

    def _on_keyboard_event(self, event, *args):
        key = self._keyboard_key_name(event.input)
        if event.type in (carb.input.KeyboardEventType.KEY_PRESS, carb.input.KeyboardEventType.KEY_REPEAT):
            if key == "SPACE":
                self.reset()
                self._pressed_keys.clear()
            elif key == "ESCAPE":
                simulation_app.close()
            elif (
                key in self._BASE_KEY_TO_DELTA
                or key in self._EE_POS_KEY_TO_DELTA
                or key in self._EE_ROT_KEY_TO_DELTA
                or key in self._GRIPPER_KEY_TO_DELTA
            ):
                self._pressed_keys.add(key)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._pressed_keys.discard(key)
        return True

    def reset(self):
        """Reset keyboard targets to a safe command state."""
        if self._base_command is not None:
            self._base_command.zero_()
        if self._ee_term is not None and self._ee_command is not None and self._ee_euler is not None:
            self._ee_command.copy_(self._ee_term.command)
            self._ee_euler = torch.stack(euler_xyz_from_quat(self._ee_command[:, 3:7]), dim=-1)
        self._gripper_opening.copy_(self._robot.data.joint_pos[0, self._gripper_joint_ids[0]])

    def update(self):
        now = time.monotonic()
        elapsed = min(now - self._last_update_time, 0.1)
        self._last_update_time = now

        if self._base_term is not None and self._base_command is not None and self._pressed_keys and self._command_rate > 0.0:
            delta = torch.zeros(3, device=self._base_command.device)
            for key in self._pressed_keys:
                if key in self._BASE_KEY_TO_DELTA:
                    delta += torch.tensor(self._BASE_KEY_TO_DELTA[key], device=self._base_command.device)
            self._base_command[0, :3] += delta * (self._command_rate * elapsed)
            self._base_command[0, :3].clamp_(self._base_lower, self._base_upper)

        if self._ee_term is not None and self._ee_command is not None and self._ee_euler is not None:
            pos_delta = torch.zeros(3, device=self._ee_command.device)
            rot_delta = torch.zeros(3, device=self._ee_command.device)
            for key in self._pressed_keys:
                if key in self._EE_POS_KEY_TO_DELTA:
                    pos_delta += torch.tensor(self._EE_POS_KEY_TO_DELTA[key], device=self._ee_command.device)
                if key in self._EE_ROT_KEY_TO_DELTA:
                    rot_delta += torch.tensor(self._EE_ROT_KEY_TO_DELTA[key], device=self._ee_command.device)
            self._ee_command[0, :3] += pos_delta * (self._ee_pos_rate * elapsed)
            self._ee_euler[0] += rot_delta * (self._ee_rot_rate * elapsed)
            self._ee_command[0, :3].clamp_(self._ee_lower[:3], self._ee_upper[:3])
            self._ee_euler[0].clamp_(self._ee_lower[3:], self._ee_upper[3:])
            self._ee_command[:, 3:7] = quat_from_euler_xyz(
                self._ee_euler[:, 0], self._ee_euler[:, 1], self._ee_euler[:, 2]
            )

        gripper_delta = sum(self._GRIPPER_KEY_TO_DELTA.get(key, 0.0) for key in self._pressed_keys)
        if gripper_delta and self._gripper_rate > 0.0:
            self._gripper_opening += gripper_delta * (self._gripper_rate * elapsed)
            self._gripper_opening.clamp_(self._gripper_lower[0], self._gripper_upper[0])

    def apply(self):
        # Isaac Lab resamples commands inside env.step(), so apply after every step as well.
        if self._base_term is not None:
            self._base_term.command[:, :3] = self._base_command[:, :3]
        if self._ee_term is not None:
            self._ee_term.command.copy_(self._ee_command)
        # The gripper is intentionally outside the policy action vector. Set
        # its three coupled prismatic-joint targets directly each frame.
        gripper_target = torch.stack(
            (
                self._gripper_opening[0],
                0.5 * self._gripper_opening[0],
                -0.5 * self._gripper_opening[0],
            )
        ).unsqueeze(0)
        gripper_target.clamp_(self._gripper_lower, self._gripper_upper)
        self._robot.set_joint_position_target(gripper_target, joint_ids=self._gripper_joint_ids)

        if self._print_interval > 0.0:
            now = time.monotonic()
            values = []
            if self._base_command is not None:
                values.extend(float(value) for value in self._base_command[0, :3].detach().cpu())
            if self._ee_command is not None:
                values.extend(float(value) for value in self._ee_command[0, :3].detach().cpu())
                values.extend(float(value) for value in self._ee_euler[0].detach().cpu())
            values.append(float(self._gripper_opening[0].detach().cpu()))
            command = tuple(values)
            if (
                now - self._last_print_time >= self._print_interval
                and command != self._last_printed_command
            ):
                print(f"[COMMAND] {command}")
                self._last_print_time = now
                self._last_printed_command = command

    def close(self):
        if self._keyboard_subscription is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_subscription)
            self._keyboard_subscription = None


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play an RSL-RL checkpoint with one environment and keyboard commands."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = 1
    print("[INFO] Forcing env_cfg.scene.num_envs=1 for keyboard playback.")
    if not args_cli.disable_box:
        # Keep the robot at its configured initial pose so the box remains in its
        # initial forward direction instead of being separated from a random reset.
        if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "reset_base"):
            env_cfg.events.reset_base = None
        add_box_to_scene_cfg(env_cfg, args_cli.box_usd, args_cli.box_distance)

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] A pre-trained checkpoint is unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play_by_key"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during keyboard playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    controller = KeyboardCommandController(
        env.unwrapped,
        command_name=args_cli.command_name,
        command_rate=args_cli.command_rate,
        ee_pos_rate=args_cli.ee_pos_rate,
        ee_rot_rate=args_cli.ee_rot_rate,
        gripper_rate=args_cli.gripper_rate,
        print_interval=args_cli.print_command_interval,
    )

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if version.parse(installed_version) >= version.parse("4.0.0"):
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
    else:
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    controller.reset()
    controller.apply()
    obs = env.get_observations()
    dt = env.unwrapped.step_dt
    timestep = 0

    try:
        while simulation_app.is_running():
            start_time = time.time()
            controller.update()
            controller.apply()
            # Refresh observations so the policy sees the command just written above.
            obs = env.get_observations()
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                if version.parse(installed_version) >= version.parse("4.0.0"):
                    policy.reset(dones)
                else:
                    policy_nn.reset(dones)

            if args_cli.video:
                timestep += 1
                if timestep == args_cli.video_length:
                    break

            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        controller.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
