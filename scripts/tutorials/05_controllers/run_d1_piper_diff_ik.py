"""Differential IK debug example for the fixed-base D1-Piper-L robot.

This script follows Isaac Lab's ``run_diff_ik.py`` control loop.  The
articulation root is fixed only in this debug configuration; the original
``D1_PIPER_CFG`` is not modified.  Task-space quantities are expressed in the
articulation-root frame (``base_link``), and the PhysX Jacobian is passed to
``DifferentialIKController`` without an additional frame rotation.

Recommended first test::

    ./isaaclab.sh -p scripts/tutorials/05_controllers/run_d1_piper_diff_ik.py \
        --num_envs 1 --command_type position --ik_method dls --lambda_val 0.05 \
        --gravity_compensation

For full pose IK, use a bent arm configuration because the neutral all-zero
Piper pose is close to a pose-Jacobian singularity::

    ./isaaclab.sh -p scripts/tutorials/05_controllers/run_d1_piper_diff_ik.py \
        --num_envs 1 --command_type pose --ik_method dls --lambda_val 0.05 \
        --initial_arm_pose 0 0.5 -0.8 0 0.3 0
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from copy import deepcopy

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="D1-Piper differential inverse-kinematics debug example.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument(
    "--command_type",
    type=str,
    choices=("position", "pose"),
    default="pose",
    help="Differential IK command type.",
)
parser.add_argument(
    "--relative_mode",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Interpret each command as a delta in the current root frame.",
)
parser.add_argument(
    "--ik_method",
    type=str,
    choices=("pinv", "svd", "trans", "dls"),
    default="dls",
    help="Method used to invert the Jacobian.",
)
parser.add_argument("--lambda_val", type=float, default=0.05, help="DLS damping coefficient.")
parser.add_argument(
    "--gravity_compensation",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Add PhysX's joint gravity-compensation torque as feed-forward effort. "
        "Use --no-gravity_compensation for comparison."
    ),
)
parser.add_argument(
    "--goal_duration",
    type=float,
    default=3.0,
    help="Seconds to hold each goal before resetting and switching to the next goal.",
)
parser.add_argument(
    "--initial_arm_pose",
    type=float,
    nargs=6,
    default=None,
    metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
    help="Optional initial joint1...joint6 pose in radians.",
)
parser.add_argument("--print_interval", type=float, default=0.5, help="Diagnostic print interval in seconds.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs < 1:
    parser.error("--num_envs must be at least 1")
if args_cli.goal_duration <= 0.0:
    parser.error("--goal_duration must be positive")
if args_cli.lambda_val <= 0.0:
    parser.error("--lambda_val must be positive")
if args_cli.print_interval <= 0.0:
    parser.error("--print_interval must be positive")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below this line runs after Isaac Sim has been launched."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    combine_frame_transforms,
    compute_pose_error,
    quat_from_angle_axis,
    quat_mul,
    subtract_frame_transforms,
)

from LeggedManip_Lab.assets.d1_piper_l.d1_piper_articulation_cfg import D1_PIPER_CFG


@configclass
class D1PiperSceneCfg(InteractiveSceneCfg):
    """Minimal scene used for the fixed-base IK test."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot = D1_PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def make_fixed_base_robot_cfg():
    """Return a private D1-Piper config with only the root-lock changed."""

    robot_cfg = deepcopy(D1_PIPER_CFG).replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot_cfg.spawn.articulation_props.fix_root_link = True
    return robot_cfg


def reset_robot(robot, scene, initial_arm_pose, arm_joint_ids):
    """Write the reset state and return the joint positions used as targets."""

    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(torch.zeros_like(root_state[:, 7:]))

    joint_pos = robot.data.default_joint_pos.clone()
    if initial_arm_pose is not None:
        arm_pose = torch.tensor(initial_arm_pose, device=robot.device, dtype=joint_pos.dtype)
        arm_pose = arm_pose.unsqueeze(0).expand(scene.num_envs, -1)
        limits = robot.data.soft_joint_pos_limits[:, arm_joint_ids, :]
        joint_pos[:, arm_joint_ids] = torch.clamp(arm_pose, limits[..., 0], limits[..., 1])
    robot.write_joint_state_to_sim(joint_pos, robot.data.default_joint_vel.clone())
    robot.reset()
    return joint_pos


def build_goals(initial_pos, initial_quat):
    """Build small, reachable goals in the articulation-root frame."""

    # Unlike the Franka example, D1-Piper's reachable workspace depends on
    # its mounting pose.  Therefore the goals are deliberately offsets from
    # the measured reset pose rather than hard-coded world coordinates.
    offsets = torch.tensor(
        [
            (0.00, 0.00, 0.00),
            (0.03, 0.00, 0.00),
            (-0.03, 0.00, 0.00),
            (0.00, 0.03, 0.00),
            (0.00, -0.03, 0.00),
            (0.00, 0.00, 0.03),
            (0.00, 0.00, -0.03),
        ],
        device=initial_pos.device,
        dtype=initial_pos.dtype,
    )
    goal_pos = initial_pos[:, None, :] + offsets[None, :, :]
    goal_quat = initial_quat[:, None, :].expand(-1, offsets.shape[0], -1).clone()

    if args_cli.command_type == "pose":
        # Test orientation only after the small position displacements.
        angle_axis = torch.tensor(
            [(0.10, 0.0, 0.0), (-0.10, 0.0, 0.0), (0.0, 0.10, 0.0), (0.0, 0.0, 0.10)],
            device=initial_pos.device,
            dtype=initial_pos.dtype,
        )
        angle = torch.linalg.vector_norm(angle_axis, dim=-1)
        axis = angle_axis / torch.clamp(angle[:, None], min=1.0e-8)
        delta_quat = quat_from_angle_axis(angle, axis)
        goal_quat[:, 3:, :] = quat_mul(
            delta_quat[None, :, :].expand(initial_quat.shape[0], -1, -1),
            initial_quat[:, None, :].expand(-1, 4, -1),
        )
    return goal_pos, goal_quat


def set_goal(controller, goal_pos, goal_quat, current_pos, current_quat):
    """Set one goal using the controller's absolute or relative API."""

    if controller.cfg.use_relative_mode:
        if controller.cfg.command_type == "position":
            command = goal_pos - current_pos
        else:
            position_error, axis_angle_error = compute_pose_error(
                current_pos,
                current_quat,
                goal_pos,
                goal_quat,
                rot_error_type="axis_angle",
            )
            command = torch.cat((position_error, axis_angle_error), dim=-1)
        controller.set_command(command, ee_pos=current_pos, ee_quat=current_quat)
    elif controller.cfg.command_type == "position":
        controller.set_command(goal_pos, ee_quat=goal_quat)
    else:
        controller.set_command(torch.cat((goal_pos, goal_quat), dim=-1))


def get_gravity_torque(robot, arm_joint_ids):
    """Return PhysX's joint torques that balance gravity for the arm."""

    gravity_disabled = (
        robot.cfg.spawn.rigid_props is not None and robot.cfg.spawn.rigid_props.disable_gravity is True
    )
    if not args_cli.gravity_compensation or gravity_disabled:
        return torch.zeros_like(robot.data.joint_pos[:, arm_joint_ids])
    return robot.root_physx_view.get_gravity_compensation_forces()[:, arm_joint_ids]


def print_diagnostics(
    robot,
    root_anchor,
    ee_pos_b,
    ee_quat_b,
    controller,
    joint_pos,
    joint_pos_des,
    gravity_torque,
    jacobian,
    t,
):
    """Print the same frame quantities used by the official IK loop."""

    pos_error, rot_error = compute_pose_error(
        ee_pos_b,
        ee_quat_b,
        controller.ee_pos_des,
        controller.ee_quat_des,
        rot_error_type="axis_angle",
    )
    jacobian_for_condition = jacobian[0, :3] if args_cli.command_type == "position" else jacobian[0]
    singular_values = torch.linalg.svdvals(jacobian_for_condition)
    condition = singular_values[0] / torch.clamp(singular_values[-1], min=1.0e-8)
    q_delta = joint_pos_des - joint_pos

    def fmt(value):
        return "[" + ", ".join(f"{float(x): .4f}" for x in value.detach().cpu()) + "]"

    print(
        f"[IK t={t:7.2f}s] p_b={fmt(ee_pos_b[0])} -> {fmt(controller.ee_pos_des[0])}, "
        f"q_b={fmt(ee_quat_b[0])} -> {fmt(controller.ee_quat_des[0])}, "
        f"|ep|={float(torch.linalg.vector_norm(pos_error[0])):.6f} m, "
        f"|er|={float(torch.linalg.vector_norm(rot_error[0])):.6f} rad"
    )
    print(
        f"    q={fmt(joint_pos[0])}, q_des={fmt(joint_pos_des[0])}, "
        f"max|dq|={float(torch.max(torch.abs(q_delta[0]))):.6f} rad, "
        f"tau_g={fmt(gravity_torque[0])}, |tau_g|={float(torch.linalg.vector_norm(gravity_torque[0])):.4f} Nm, "
        f"jacobian_s={fmt(singular_values)}, cond={float(condition):.3e}, "
        f"root_drift={float(torch.linalg.vector_norm(robot.data.root_pose_w[0, :3] - root_anchor[0, :3])):.6f} m"
    )
    if float(condition) > 1.0e4:
        print("    [WARNING] Jacobian is close to singular; try position IK or a bent initial arm pose.")


def run_simulator(sim, scene):
    """Run the official Isaac Lab differential-IK simulation loop."""

    robot = scene["robot"]

    # Robot-specific entity selection.  preserve_order is important because
    # the controller expects [joint1, ..., joint6] in this order.
    robot_entity_cfg = SceneEntityCfg(
        "robot",
        joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        body_names=["end_effector"],
        preserve_order=True,
    )
    robot_entity_cfg.resolve(scene)
    piper_body_ids, piper_body_names = robot.find_bodies("piper_base_link")
    if len(piper_body_ids) != 1:
        raise RuntimeError(
            "Expected one piper_base_link body, found "
            f"{len(piper_body_ids)}: {piper_body_names}"
        )
    piper_body_idx = piper_body_ids[0]

    # Exactly the index rule used by Isaac Lab's official run_diff_ik.py.
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]
    if not robot.is_fixed_base:
        raise RuntimeError("D1-Piper root is not fixed. Check fix_root_link=True in the debug config.")

    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type=args_cli.command_type,
        use_relative_mode=args_cli.relative_mode,
        ik_method=args_cli.ik_method,
        ik_params={"lambda_val": args_cli.lambda_val} if args_cli.ik_method == "dls" else None,
    )
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
    ee_marker = VisualizationMarkers(marker_cfg.replace(prim_path="/Visuals/d1_piper_ee_current"))
    goal_marker = VisualizationMarkers(marker_cfg.replace(prim_path="/Visuals/d1_piper_ee_goal"))

    sim_dt = sim.get_physics_dt()
    arm_joint_ids = robot_entity_cfg.joint_ids

    # Establish the requested reset pose once before measuring the goals.
    reset_joint_pos = reset_robot(robot, scene, args_cli.initial_arm_pose, arm_joint_ids)
    robot.set_joint_position_target(reset_joint_pos)
    gravity_torque = get_gravity_torque(robot, arm_joint_ids)
    robot.set_joint_effort_target(gravity_torque, joint_ids=arm_joint_ids)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim_dt)
    root_anchor = robot.data.root_pose_w.clone()

    ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
    root_pose_w = robot.data.root_pose_w
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    piper_pose_w = robot.data.body_pose_w[:, piper_body_idx]
    ee_pos_piper, ee_quat_piper = subtract_frame_transforms(
        piper_pose_w[:, 0:3], piper_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    reset_ee_pos_b = ee_pos_b.clone()
    reset_ee_quat_b = ee_quat_b.clone()
    goal_pos, goal_quat = build_goals(ee_pos_b.clone(), ee_quat_b.clone())

    if args_cli.command_type == "pose" and args_cli.initial_arm_pose is None:
        print(
            "[WARNING] Pose IK starts from the asset neutral pose. "
            "Use --initial_arm_pose 0 0.5 -0.8 0 0.3 0 if the pose Jacobian is singular."
        )

    goal_idx = 0
    set_goal(diff_ik_controller, goal_pos[:, goal_idx], goal_quat[:, goal_idx], ee_pos_b, ee_quat_b)
    goal_steps = max(1, round(args_cli.goal_duration / sim_dt))
    count = 0
    sim_time = 0.0
    print_elapsed = 0.0
    joint_pos_des = reset_joint_pos[:, arm_joint_ids].clone()
    jacobian = None

    print(
        f"[INFO] Fixed-base D1-Piper IK started: num_envs={scene.num_envs}, "
        f"ee_body_id={robot_entity_cfg.body_ids[0]}, jacobian_id={ee_jacobi_idx}, "
        f"arm_joint_ids={arm_joint_ids}, command_type={args_cli.command_type}, "
        f"relative_mode={args_cli.relative_mode}, ik={args_cli.ik_method}, "
        f"gravity_compensation={args_cli.gravity_compensation}"
    )
    print(f"[INFO] Initial EE pose in base_link: p={ee_pos_b[0].cpu().tolist()}, q(wxyz)={ee_quat_b[0].cpu().tolist()}")
    print(
        "[INFO] Initial EE pose in piper_base_link: "
        f"p={ee_pos_piper[0].cpu().tolist()}, q(wxyz)={ee_quat_piper[0].cpu().tolist()}"
    )

    while simulation_app.is_running():
        # This reset/compute split is the same as run_diff_ik.py.  The only
        # difference is that D1 has a fixed root and six selected arm joints.
        if count % goal_steps == 0:
            count = 0
            reset_joint_pos = reset_robot(robot, scene, args_cli.initial_arm_pose, arm_joint_ids)
            robot.set_joint_position_target(reset_joint_pos)
            joint_pos_des = reset_joint_pos[:, arm_joint_ids].clone()
            gravity_torque = get_gravity_torque(robot, arm_joint_ids)
            diff_ik_controller.reset()
            set_goal(
                diff_ik_controller,
                goal_pos[:, goal_idx],
                goal_quat[:, goal_idx],
                reset_ee_pos_b,
                reset_ee_quat_b,
            )
            goal_idx = (goal_idx + 1) % goal_pos.shape[1]
        else:
            # Official Jacobian extraction: do not rotate this tensor by hand.
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_joint_ids]
            ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
            root_pose_w = robot.data.root_pose_w
            joint_pos = robot.data.joint_pos[:, arm_joint_ids]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
            gravity_torque = get_gravity_torque(robot, arm_joint_ids)

        # As in the official example, only the selected arm targets are sent
        # to the articulation each step. Other D1 joints keep their reset PD
        # targets established above.
        robot.set_joint_position_target(joint_pos_des, joint_ids=arm_joint_ids)
        # D1's DelayedPDActuator adds this effort to its normal position-PD
        # torque, yielding tau = tau_PD + tau_gravity.  This keeps the IK
        # structure intact and avoids double-computing the PD term.
        robot.set_joint_effort_target(gravity_torque, joint_ids=arm_joint_ids)
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)
        sim_time += sim_dt
        print_elapsed += sim_dt

        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        goal_pos_w, goal_quat_w = combine_frame_transforms(
            robot.data.root_pose_w[:, 0:3],
            robot.data.root_pose_w[:, 3:7],
            diff_ik_controller.ee_pos_des,
            diff_ik_controller.ee_quat_des,
        )
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(goal_pos_w, goal_quat_w)

        if print_elapsed >= args_cli.print_interval and jacobian is not None:
            print_elapsed = 0.0
            print_diagnostics(
                robot,
                root_anchor,
                ee_pos_b,
                ee_quat_b,
                diff_ik_controller,
                joint_pos,
                joint_pos_des,
                gravity_torque,
                jacobian,
                sim_time,
            )


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(1.4, -1.4, 1.0), target=(0.2, 0.0, 0.45))

    scene_cfg = D1PiperSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=2.0,
        replicate_physics=True,
        lazy_sensor_update=False,
    )
    scene_cfg.robot = make_fixed_base_robot_cfg()
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO] Scene setup complete. Running standalone fixed-base IK test...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
