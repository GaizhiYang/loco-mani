"""Differential IK debug example for the fixed-base D1-Piper-L robot.

This script follows Isaac Lab's ``run_diff_ik.py`` control loop.  The
articulation root is fixed only in this debug configuration; the original
``D1_PIPER_CFG`` is not modified.  The pose error and the PhysX Jacobian are
expressed in the articulation-root frame (``base_link``). Random goals are
sampled in ``piper_base_link`` and transformed into that IK frame. Joint targets
are limited in position and rate before the explicit PD actuator adds the
model's gravity-compensation torque.

Finite tracking check (run in the Isaac Lab Python environment)::

    python scripts/tutorials/05_controllers/run_d1_piper_diff_ik.py \
        --headless --num_envs 2 --seed 42 --max_cycles 1 --check

Each cycle contains ``--goals_per_cycle`` independently sampled goals per
environment (default: 7). Every goal switch draws new XYZ and Euler XYZ values
uniformly from ``GOAL_POSE_RANGES`` below, including on subsequent cycles.
Angles are in radians; quaternions use (w, x, y, z), as in UniformBodyPoseCommand.

The default bent pose avoids the neutral pose's singularity and joint limits.
Use ``--no-gravity_compensation`` to compare gravity feed-forward. Empty URDF
helper frames are given negligible positive masses in this scene only; the
source USD and the training asset configuration are not edited. To reproduce
the imported USD's extra helper-frame payload, use ``--no-fix_helper_masses``.

Independent position/orientation bounds do not guarantee reachable pose
combinations. Targets are not filtered by reachability; ``--check`` reports any
goal that fails to converge within ``--goal_duration`` instead of resampling it.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import traceback
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
parser.add_argument("--lambda_val", type=float, default=0.02, help="DLS damping coefficient.")
parser.add_argument("--physics_dt", type=float, default=0.005, help="Physics/PD control period in seconds.")
parser.add_argument(
    "--max_joint_velocity",
    type=float,
    default=1.0,
    help="Maximum joint target slew rate in rad/s (0 disables rate limiting).",
)
parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible random pose sampling.")
parser.add_argument("--goals_per_cycle", type=int, default=7, help="Random goals per environment in each cycle.")
parser.add_argument("--max_cycles", type=int, default=0, help="Exit after this many random goal cycles (0 runs forever).")
parser.add_argument("--check", action="store_true", help="Fail if any goal misses the end-of-hold tolerances.")
parser.add_argument("--position_tolerance", type=float, default=0.002, help="Goal position tolerance in meters.")
parser.add_argument("--orientation_tolerance", type=float, default=0.02, help="Goal orientation tolerance in radians.")
parser.add_argument(
    "--fix_helper_masses",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Remove the importer's 1 kg fallback masses from empty URDF helper frames in this scene only.",
)
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
    default=[0.0, 1.0, -1.5, 0.0, 0.5, 0.0],
    metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
    help="Initial joint1...joint6 pose in radians; defaults to a bent, nonsingular pose.",
)
parser.add_argument("--print_interval", type=float, default=0.5, help="Diagnostic print interval in seconds.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs < 1:
    parser.error("--num_envs must be at least 1")
if args_cli.goals_per_cycle < 1:
    parser.error("--goals_per_cycle must be at least 1")
if args_cli.goal_duration <= 0.0:
    parser.error("--goal_duration must be positive")
if args_cli.lambda_val <= 0.0:
    parser.error("--lambda_val must be positive")
if args_cli.print_interval <= 0.0:
    parser.error("--print_interval must be positive")
if args_cli.physics_dt <= 0 or args_cli.max_joint_velocity < 0 or args_cli.max_cycles < 0:
    parser.error("physics_dt must be positive; max_joint_velocity and max_cycles must be nonnegative")
if args_cli.position_tolerance <= 0 or args_cli.orientation_tolerance <= 0:
    parser.error("Goal tolerances must be positive")
if args_cli.check and not args_cli.max_cycles:
    parser.error("--check requires a positive --max_cycles")

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
    matrix_from_quat,
    quat_from_euler_xyz,
    quat_inv,
    quat_unique,
    subtract_frame_transforms,
)

from LeggedManip_Lab.assets.d1_piper_l.d1_piper_articulation_cfg import D1_PIPER_CFG
from LeggedManip_Lab.assets.d1_piper_l.ik_helpers import fix_d1_piper_helper_frames


# Match CommandsCfg.ee_pose.limit_ranges, not its narrower curriculum ranges.
# These are absolute end_effector poses in piper_base_link: meters and radians.
GOAL_POSE_RANGES = {
    "pos_x": (0.15, 0.50),
    "pos_y": (-0.30, 0.30),
    "pos_z": (0.10, 0.55),
    "roll": (-0.45, 0.45),
    "pitch": (0.90, 2.00),
    "yaw": (-0.60, 0.60),
}


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


def configure_helper_frame_masses(sim, scene):
    """Avoid treating empty coordinate frames as two additional 1 kg payloads.

    Keep the links so the end-effector body/Jacobian remain addressable. A
    positive epsilon avoids PhysX's automatic mass/inertia fallback for zero
    values. Author only local stage overrides before physics initialization.
    """

    if not args_cli.fix_helper_masses:
        return
    fix_d1_piper_helper_frames(
        sim.stage, [f"{env_path}/Robot" for env_path in scene.env_prim_paths], D1_PIPER_CFG.spawn.usd_path
    )


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
        limits = robot.data.joint_pos_limits[:, arm_joint_ids, :]
        joint_pos[:, arm_joint_ids] = torch.clamp(arm_pose, limits[..., 0], limits[..., 1])
    robot.write_joint_state_to_sim(joint_pos, robot.data.default_joint_vel.clone())
    robot.reset()
    return joint_pos


def sample_goal_poses(num_envs, device, dtype, generator):
    """Sample independent absolute poses in piper_base_link for all environments.

    Uniform Euler XYZ sampling follows UniformBodyPoseCommand; it is not a
    uniform distribution on SO(3). Return the original Euler samples too, since
    converting back from quaternions near pitch=pi/2 can select another branch.
    """

    pose = torch.empty((num_envs, 6), device=device, dtype=dtype)
    for component, bounds in enumerate(GOAL_POSE_RANGES.values()):
        pose[:, component].uniform_(*bounds, generator=generator)
    quat = quat_unique(quat_from_euler_xyz(pose[:, 3], pose[:, 4], pose[:, 5]))
    return pose[:, :3], quat, pose[:, 3:]


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
    gravity = robot.root_physx_view.get_gravity_compensation_forces()
    if gravity.shape[-1] != robot.num_joints:
        raise RuntimeError(f"Unexpected fixed-base gravity tensor shape: {tuple(gravity.shape)}")
    # This API already returns compensation, not the force of gravity itself.
    # Do not negate it or add another PD torque on top of the actuator's PD.
    return gravity[:, arm_joint_ids].clone()


def get_ee_state(robot, body_id):
    """Get the measured end-effector pose in the articulation-root frame."""

    ee_pose_w = robot.data.body_pose_w[:, body_id]
    root_pose_w = robot.data.root_pose_w
    return subtract_frame_transforms(
        root_pose_w[:, :3], root_pose_w[:, 3:7], ee_pose_w[:, :3], ee_pose_w[:, 3:7]
    )


def limit_joint_targets(goal, previous, limits, dt):
    """Respect hard joint limits and slew the commanded trajectory in rad/s."""

    goal = goal.clamp(min=limits[..., 0], max=limits[..., 1])
    if args_cli.max_joint_velocity:
        step = args_cli.max_joint_velocity * dt
        goal = previous + (goal - previous).clamp(min=-step, max=step)
    return goal.clamp(min=limits[..., 0], max=limits[..., 1])


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
    arm_joint_ids,
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
    applied = robot.data.applied_torque[:, arm_joint_ids]
    requested = robot.data.computed_torque[:, arm_joint_ids]
    clipped = (requested - applied).abs().amax(dim=-1)

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
        f"tau_applied={fmt(applied[0])}, torque_clipped={float(clipped[0]):.4f} Nm, "
        f"jacobian_s={fmt(singular_values)}, cond={float(condition):.3e}, "
        f"root_drift={float(torch.linalg.vector_norm(robot.data.root_pose_w[0, :3] - root_anchor[0, :3])):.6f} m"
    )
    if float(condition) > 1.0e4:
        print("    [WARNING] Jacobian is close to singular; try position IK or a bent initial arm pose.")
    if float(clipped.max()) > 0.1:
        print("    [WARNING] Actuator torque is saturated: gravity feed-forward also shares this effort limit.")


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
    arm_limits = robot.data.joint_pos_limits[:, arm_joint_ids]
    masses = robot.root_physx_view.get_masses()[0]
    print(
        "[MODEL] Arm body masses (kg): "
        + ", ".join(
            f"{name}={float(masses[i]):.4f}"
            for i, name in enumerate(robot.body_names)
            if name.startswith(("link", "piper", "gripper", "flange", "end_effector"))
        )
    )

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
    # The arm mount is rigidly attached to the fixed articulation root. Use its
    # measured transform, not a hard-coded base_link/piper_base_link offset.
    piper_pos_b, piper_quat_b = subtract_frame_transforms(
        root_pose_w[:, :3], root_pose_w[:, 3:7], piper_pose_w[:, :3], piper_pose_w[:, 3:7]
    )
    # A dedicated generator makes sampling independent of simulator RNG use.
    goal_generator = torch.Generator(device=sim.device).manual_seed(args_cli.seed)

    goal_steps = max(1, round(args_cli.goal_duration / sim_dt))
    count = 0
    sim_time = 0.0
    print_elapsed = 0.0
    joint_pos_des = reset_joint_pos[:, arm_joint_ids].clone()
    jacobian = None
    completed_rounds = 0
    failed_goals = 0

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
    print(
        f"[INFO] Random goals: frame=piper_base_link, seed={args_cli.seed}, "
        f"goals_per_cycle={args_cli.goals_per_cycle}, hold={args_cli.goal_duration}s, "
        f"ranges={GOAL_POSE_RANGES}"
    )

    while simulation_app.is_running():
        # Keep run_diff_ik.py's reset/compute split. Check the measured end
        # state before resetting so a failed target cannot be hidden by reset.
        if count % goal_steps == 0:
            if count:
                measured_pos, measured_quat = get_ee_state(robot, robot_entity_cfg.body_ids[0])
                pe, re = compute_pose_error(
                    measured_pos,
                    measured_quat,
                    diff_ik_controller.ee_pos_des,
                    diff_ik_controller.ee_quat_des,
                    rot_error_type="axis_angle",
                )
                pe = torch.linalg.vector_norm(pe, dim=-1)
                re = torch.linalg.vector_norm(re, dim=-1)
                passed = pe <= args_cli.position_tolerance
                if args_cli.command_type == "pose":
                    passed &= re <= args_cli.orientation_tolerance
                num_passed = int(passed.sum().item())
                failed_goals += scene.num_envs - num_passed
                completed_rounds += 1
                print(
                    f"[GOAL {completed_rounds}] {num_passed}/{scene.num_envs} environments reached tolerance: "
                    f"max_position_error={pe.max().item():.6f} m, "
                    f"max_orientation_error={re.max().item():.6f} rad"
                )
                if args_cli.max_cycles and completed_rounds >= args_cli.max_cycles * args_cli.goals_per_cycle:
                    break
            count = 0
            reset_joint_pos = reset_robot(robot, scene, args_cli.initial_arm_pose, arm_joint_ids)
            robot.set_joint_position_target(reset_joint_pos)
            joint_pos_des = reset_joint_pos[:, arm_joint_ids].clone()
            gravity_torque = get_gravity_torque(robot, arm_joint_ids)
            diff_ik_controller.reset()
            goal_pos_piper, goal_quat_piper, goal_rpy_piper = sample_goal_poses(
                scene.num_envs, sim.device, reset_ee_pos_b.dtype, goal_generator
            )
            goal_pos_b, goal_quat_b = combine_frame_transforms(
                piper_pos_b, piper_quat_b, goal_pos_piper, goal_quat_piper
            )
            print(
                f"[TARGET {completed_rounds + 1}] env=0, frame=piper_base_link, "
                f"xyz={goal_pos_piper[0].cpu().tolist()}, rpy={goal_rpy_piper[0].cpu().tolist()}"
            )
            set_goal(
                diff_ik_controller,
                goal_pos_b,
                goal_quat_b,
                reset_ee_pos_b,
                reset_ee_quat_b,
            )
        else:
            # PhysX expresses both linear and angular Jacobian rows in world
            # axes. Rotate them into the same root frame as the pose error.
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_joint_ids]
            ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
            root_pose_w = robot.data.root_pose_w
            joint_pos = robot.data.joint_pos[:, arm_joint_ids]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            world_to_root = matrix_from_quat(quat_inv(root_pose_w[:, 3:7]))
            jacobian = torch.cat(
                (
                    torch.bmm(world_to_root, jacobian[:, :3]),
                    torch.bmm(world_to_root, jacobian[:, 3:]),
                ),
                dim=1,
            )
            joint_pos_goal = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
            joint_pos_des = limit_joint_targets(joint_pos_goal, joint_pos_des, arm_limits, sim_dt)
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
            # Report the post-step state, not the stale pre-step measurement.
            ee_pos_b, ee_quat_b = get_ee_state(robot, robot_entity_cfg.body_ids[0])
            joint_pos = robot.data.joint_pos[:, arm_joint_ids]
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
                arm_joint_ids,
            )
    completed_goals = completed_rounds * scene.num_envs
    print(f"[SUMMARY] {completed_goals - failed_goals}/{completed_goals} random poses reached tolerance.")
    required_goals = args_cli.max_cycles * args_cli.goals_per_cycle * scene.num_envs
    if args_cli.check and (failed_goals or completed_goals != required_goals):
        raise RuntimeError(
            f"IK tracking check failed: {failed_goals} missed goals; "
            f"completed {completed_goals}/{required_goals} goals."
        )


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=args_cli.physics_dt, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    try:
        sim.set_camera_view(eye=(1.4, -1.4, 1.0), target=(0.2, 0.0, 0.45))
        scene_cfg = D1PiperSceneCfg(
            num_envs=args_cli.num_envs,
            env_spacing=2.0,
            replicate_physics=True,
            lazy_sensor_update=False,
        )
        scene_cfg.robot = make_fixed_base_robot_cfg()
        scene = InteractiveScene(scene_cfg)
        configure_helper_frame_masses(sim, scene)
        sim.reset()
        print("[INFO] Scene setup complete. Running standalone fixed-base IK test...")
        run_simulator(sim, scene)
    finally:
        # Unsubscribe Isaac Lab's stop/render callback before app shutdown.
        # Otherwise a finite test can keep rendering forever after STOP.
        sim.clear_all_callbacks()
        sim.clear_instance()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Kit's fast shutdown exits Python during close(). Preserve the error
        # and nonzero exit status before cleanup, including failed --check runs.
        traceback.print_exc()
        simulation_app.app.post_uncancellable_quit(1)
        raise
    finally:
        # This diagnostic does not record Replicator data.
        simulation_app.close(wait_for_replicator=False)
