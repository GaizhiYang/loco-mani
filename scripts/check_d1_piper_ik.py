"""Regression checks for D1-Piper balance rewards and script-equivalent arm IK.

Run with the Isaac Lab Python environment:
    python scripts/check_d1_piper_ik.py --headless
"""

import argparse
import ast
from copy import deepcopy
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import unittest
from types import SimpleNamespace

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.controllers import DifferentialIKController
from isaaclab.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_from_euler_xyz,
    quat_inv,
    subtract_frame_transforms,
)

import LeggedManip_Lab.tasks  # noqa: F401
from LeggedManip_Lab.assets.d1_piper_l.d1_piper_articulation_cfg import D1_PIPER_CFG
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.d1_piper_l.agents.rsl_rl_ppo_cfg import (
    D1PiperIKPPORunnerCfg,
    D1PiperWBCPPORunnerCfg,
)
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.d1_piper_l.d1_piper_common_env_cfg import (
    D1_LEG_JOINT_NAMES,
    D1_LEG_POS_JOINT_NAMES,
    D1_WHEEL_NAMES,
)
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.d1_piper_l.ik_env_cfg import (
    D1PiperIKEnvCfg,
    D1PiperIKEnvCfg_PLAY,
)
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp.differential_ik_actions import (
    CommandDifferentialIKGravityCompensationAction,
)
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.mdp.rewards import stand_nice, stand_still


class TestStandingRewards(unittest.TestCase):
    def test_rolling_then_stopping_does_not_create_posture_cost(self):
        # Three otherwise identical stopped robots have different wheel travel.
        positions = torch.zeros(3, 16)
        positions[:, [3, 7, 11, 15]] = torch.tensor([[0.0], [20.0], [1000.0]])
        data = SimpleNamespace(
            joint_pos=positions,
            default_joint_pos=torch.zeros_like(positions),
            root_lin_vel_b=torch.zeros(3, 3),
            root_ang_vel_b=torch.zeros(3, 3),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(3, 1),
        )
        command = torch.zeros(3, 3)
        env = SimpleNamespace(
            scene={"robot": SimpleNamespace(data=data)},
            command_manager=SimpleNamespace(get_command=lambda _: command),
        )
        cfg = SimpleNamespace(name="robot", joint_ids=[i for i in range(16) if i % 4 != 3])
        for reward in (stand_still, stand_nice):
            with self.subTest(reward=reward.__name__):
                torch.testing.assert_close(reward(env, "base_velocity", asset_cfg=cfg), torch.zeros(3))
                # Leg posture still matters, regardless of wheel travel.
                positions[:, 0] = 0.2
                costs = reward(env, "base_velocity", asset_cfg=cfg)
                self.assertTrue(torch.all(costs > 0))
                torch.testing.assert_close(costs, costs[0].expand_as(costs))
                command[:, 0] = 0.5
                torch.testing.assert_close(reward(env, "base_velocity", asset_cfg=cfg), torch.zeros(3))
                positions[:, 0] = 0.0
                command.zero_()

    def test_ik_ppo_changes_do_not_mutate_wbc(self):
        ik = D1PiperIKPPORunnerCfg()
        wbc = D1PiperWBCPPORunnerCfg()
        self.assertEqual(ik.actor.distribution_cfg.init_std, 0.5)
        self.assertEqual(ik.algorithm.entropy_coef, 0.001)
        self.assertEqual(ik.clip_actions, 3.0)
        self.assertEqual(wbc.actor.distribution_cfg.init_std, 1.0)
        self.assertEqual(wbc.algorithm.entropy_coef, 0.005)


class TestArmTargetSlew(unittest.TestCase):
    def test_large_goals_reversals_and_partial_reset(self):
        # Exercise the real limiter without depending on the robot's dynamics.
        action = object.__new__(CommandDifferentialIKGravityCompensationAction)
        action._debug_vis_handle = None
        action._joint_ids = list(range(6))
        action._asset = SimpleNamespace(data=SimpleNamespace(
            joint_pos_limits=torch.tensor([-1.0, 1.0]).repeat(2, 6, 1)
        ))
        action._joint_pos_des = torch.full((2, 6), -0.8)
        action._target_needs_init = torch.ones(2, dtype=torch.bool)
        action._max_joint_target_step = 0.5 * 0.005
        measured = torch.full((2, 6), 0.3)
        goal = torch.full((2, 6), 100.0)
        action._update_joint_target(goal, measured)
        torch.testing.assert_close(action._joint_pos_des, measured + 0.0025)
        for _ in range(500):
            before = action._joint_pos_des.clone()
            action._update_joint_target(goal, measured)
            self.assertLessEqual((action._joint_pos_des - before).abs().max().item(), 0.002501)
            self.assertTrue(torch.all(action._joint_pos_des <= 1.0))
        torch.testing.assert_close(action._joint_pos_des, torch.ones(2, 6))

        # Only one environment resets; the other keeps its own smooth target.
        action._target_needs_init[0] = True
        measured[0] = -0.4
        action._update_joint_target(-goal, measured)
        torch.testing.assert_close(action._joint_pos_des[0], measured[0] - 0.0025)
        torch.testing.assert_close(action._joint_pos_des[1], torch.full((6,), 0.9975))


class TestIKConfiguration(unittest.TestCase):
    def test_matches_script_and_preserves_shared_assets(self):
        # Parse defaults without launching a second SimulationApp by importing
        # the executable tutorial. This catches later script/task drift.
        tutorial = Path(__file__).parent / "tutorials/05_controllers/run_d1_piper_diff_ik.py"
        tree = ast.parse(tutorial.read_text())
        script_ranges = next(
            ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "GOAL_POSE_RANGES" for target in node.targets)
        )
        defaults = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
                for keyword in node.keywords:
                    if keyword.arg == "default":
                        defaults[ast.literal_eval(node.args[0])] = ast.literal_eval(keyword.value)
        shared_pose = deepcopy(D1_PIPER_CFG.init_state.joint_pos)
        for cfg in (D1PiperIKEnvCfg(), D1PiperIKEnvCfg_PLAY()):
            self.assertIsNone(cfg.actions.joint_pos.arm_control_start_iteration)
            self.assertEqual(cfg.sim.dt, defaults["--physics_dt"])
            self.assertEqual(cfg.actions.joint_pos.max_joint_target_velocity, defaults["--max_joint_velocity"])
            self.assertEqual(cfg.actions.joint_pos.controller.ik_params["lambda_val"], defaults["--lambda_val"])
            self.assertEqual(
                [cfg.scene.robot.init_state.joint_pos[f"joint{i}"] for i in range(1, 7)],
                defaults["--initial_arm_pose"],
            )
            self.assertEqual(cfg.commands.ee_pose.root_name, "piper_base_link")
            self.assertEqual(cfg.commands.ee_pose.resampling_time_range, (defaults["--goal_duration"],) * 2)
            self.assertFalse(cfg.commands.ee_pose.curriculum_enabled)
            self.assertIsNone(cfg.curriculum.pos_cmd_levels)
            self.assertFalse(cfg.scene.robot.spawn.articulation_props.fix_root_link)
            self.assertTrue(cfg.scene.replicate_physics)
            for name, bounds in script_ranges.items():
                self.assertEqual(getattr(cfg.commands.ee_pose.ranges, name), bounds)
                self.assertEqual(getattr(cfg.commands.ee_pose.limit_ranges, name), bounds)
            self.assertEqual(
                cfg.events.randomize_actuator_gains.params["asset_cfg"].joint_names, D1_LEG_JOINT_NAMES
            )
        self.assertEqual(D1_PIPER_CFG.init_state.joint_pos, shared_pose)
        self.assertIs(D1_PIPER_CFG.spawn.func, sim_utils.spawn_from_usd)


class TestLiveIKEnvironment(unittest.TestCase):
    def test_script_controller_on_replicated_floating_robots(self):
        cfg = D1PiperIKEnvCfg()
        cfg.seed = 42
        cfg.scene.num_envs = 8
        cfg.sim.device = args.device
        cfg.commands.ee_pose.debug_vis = False
        cfg.commands.base_velocity.debug_vis = False
        env = gym.make("D1-PIPER-IK", cfg=cfg)
        try:
            base = env.unwrapped
            env.reset()
            self.assertEqual(base.action_manager.total_action_dim, 16)
            robot = base.scene["robot"]
            arm = base.action_manager.get_term("joint_pos")
            self.assertFalse(robot.is_fixed_base)
            self.assertIsNone(arm._arm_control_start_step)
            self.assertTrue(arm._is_ik_phase_active())
            self.assertAlmostEqual(arm._max_joint_target_step, 0.005)
            initial_arm = torch.tensor([0.0, 1.0, -1.5, 0.0, 0.5, 0.0], device=base.device).repeat(8, 1)
            torch.testing.assert_close(robot.data.joint_pos[:, arm._joint_ids], initial_arm)
            helper_ids, _ = robot.find_bodies(["end_effector", "gripper_link"])
            masses = robot.root_physx_view.get_masses()[:, helper_ids]
            torch.testing.assert_close(masses, torch.full_like(masses, 1e-4))
            inertias = robot.root_physx_view.get_inertias()[:, helper_ids].reshape(8, 2, 3, 3)
            torch.testing.assert_close(
                inertias, torch.eye(3, device=inertias.device).expand_as(inertias) * 1e-8, atol=1e-10, rtol=1e-4
            )
            for name in ("stand_still", "stand_nice"):
                selected = base.reward_manager.get_term_cfg(name).params["asset_cfg"].joint_ids
                names = {base.scene["robot"].joint_names[i] for i in selected}
                self.assertEqual(names, set(D1_LEG_POS_JOINT_NAMES))
                self.assertTrue(names.isdisjoint(D1_WHEEL_NAMES))

            # Even callers bypassing the RSL wrapper get physical target bounds.
            base.action_manager.process_action(torch.full((8, 16), 1000.0, device=base.device))
            for leg in ("fl", "fr", "rl", "rr"):
                pos = base.action_manager.get_term(f"joint_pos_{leg}")
                vel = base.action_manager.get_term(f"joint_vel_{leg}")
                torch.testing.assert_close(pos.processed_actions, pos._offset + 0.5)
                torch.testing.assert_close(vel.processed_actions, torch.full_like(vel.processed_actions, 15.0))

            # Match the script's root-frame DLS solution on rotated floating
            # bases. The training action solves the same problem in Piper axes.
            root_pose = robot.data.root_pose_w.clone()
            angles = torch.tensor([0.15, -0.12, 0.6], device=base.device).repeat(8, 1)
            root_pose[:, 3:7] = quat_from_euler_xyz(*angles.unbind(-1))
            robot.write_root_pose_to_sim(root_pose)
            base.sim.forward()
            base.scene.update(base.physics_dt)
            command = base.command_manager.get_command("ee_pose")
            reference = robot.data.body_pose_w[:, arm._reference_body_idx]
            target_pos_w, target_quat_w = combine_frame_transforms(
                reference[:, :3], reference[:, 3:7], command[:, :3], command[:, 3:7]
            )
            target_pos_b, target_quat_b = subtract_frame_transforms(
                root_pose[:, :3], root_pose[:, 3:7], target_pos_w, target_quat_w
            )
            ee_pose = robot.data.body_pose_w[:, arm._body_idx]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose[:, :3], root_pose[:, 3:7], ee_pose[:, :3], ee_pose[:, 3:7]
            )
            world_to_root = matrix_from_quat(quat_inv(root_pose[:, 3:7]))
            jacobian_w = arm.jacobian_w.clone()
            jacobian_b = torch.cat(
                (torch.bmm(world_to_root, jacobian_w[:, :3]), torch.bmm(world_to_root, jacobian_w[:, 3:])),
                dim=1,
            )
            script_controller = DifferentialIKController(cfg.actions.joint_pos.controller, 8, base.device)
            script_controller.set_command(torch.cat((target_pos_b, target_quat_b), dim=-1))
            q = robot.data.joint_pos[:, arm._joint_ids].clone()
            script_goal = script_controller.compute(ee_pos_b, ee_quat_b, jacobian_b, q)
            arm._ik_controller.set_command(command)
            training_goal = arm._ik_controller.compute(
                *arm._compute_reference_frame_pose(), arm._compute_reference_frame_jacobian(), q
            )
            torch.testing.assert_close(training_goal, script_goal, atol=2e-4, rtol=2e-4)
            arm.apply_actions()
            limits = robot.data.joint_pos_limits[:, arm._joint_ids]
            script_goal = script_goal.clamp(limits[..., 0], limits[..., 1])
            expected_target = q + (script_goal - q).clamp(-0.005, 0.005)
            torch.testing.assert_close(arm._joint_pos_des, expected_target, atol=2e-5, rtol=2e-5)
            gravity = robot.root_physx_view.get_gravity_compensation_forces()
            gravity_offset = 0 if gravity.shape[1] == robot.num_joints else 6
            torch.testing.assert_close(arm._gravity_torque, gravity[:, arm._joint_ids_tensor + gravity_offset])
            self.assertTrue(torch.isfinite(arm._gravity_torque).all())

            # Fresh targets for every environment; sampling is in Piper axes.
            term = base.command_manager.get_term("ee_pose")
            previous_command = command.clone()
            term._resample_command(torch.arange(8, device=base.device))
            self.assertFalse(torch.equal(previous_command, term.command))
            self.assertEqual(torch.unique(term.command, dim=0).shape[0], 8)
            for axis, name in enumerate(("pos_x", "pos_y", "pos_z")):
                low, high = getattr(cfg.commands.ee_pose.ranges, name)
                self.assertTrue(torch.all((term.command[:, axis] >= low) & (term.command[:, axis] <= high)))
            torch.testing.assert_close(term.command[:, 3:].norm(dim=-1), torch.ones(8, device=base.device))

            env.reset()
            zeros = torch.zeros((8, 16), device=base.device)
            with torch.inference_mode():
                env.step(zeros)
                self.assertFalse(torch.allclose(arm._joint_pos_des, initial_arm))
                for _ in range(100):
                    before = arm._joint_pos_des.clone()
                    was_initialized = ~arm._target_needs_init.clone()
                    obs, reward, terminated, truncated, _ = env.step(zeros)
                    self.assertTrue(torch.isfinite(reward).all())
                    for values in obs.values():
                        self.assertTrue(torch.isfinite(values).all())
                    continuing = was_initialized & ~terminated & ~truncated
                    if continuing.any():
                        delta = (arm._joint_pos_des[continuing] - before[continuing]).abs()
                        self.assertLessEqual(delta.max().item(), 1.0 * base.step_dt + 1e-6)
                # Partial reset only clears the selected target history.
                before = arm._joint_pos_des.clone()
                arm.reset(torch.tensor([0, 2], device=base.device))
                self.assertTrue(arm._target_needs_init[[0, 2]].all())
                torch.testing.assert_close(arm._joint_pos_des[[1, 3]], before[[1, 3]])
                env.reset()
                self.assertTrue(arm._target_needs_init.all())
                env.step(zeros)
                self.assertFalse(arm._target_needs_init.any())
        finally:
            env.close()


if __name__ == "__main__":
    try:
        result = unittest.main(argv=[__file__], verbosity=2, exit=False).result
        if not result.wasSuccessful():
            app.app.post_uncancellable_quit(1)
    finally:
        app.close(wait_for_replicator=False)
