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
# See the License for the specific language governing permissions and
# limitations under the License.

"""D1 + Piper-L task with a differential-IK arm action.

The policy controls only the 16 leg and wheel joints inherited from the WBC
task.  The arm action term consumes no policy dimensions: it reads a 7-D
end-effector pose command expressed in ``piper_base_link``, and uses
IsaacLab's ``DifferentialIKController`` directly in that same frame to compute
six arm joint targets.  PhysX gravity-compensation torque is added as actuator
feed-forward.  Joint state/action observations contain only locomotion
terms; the actor also receives the commanded end-effector pose.
"""

from copy import deepcopy

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from LeggedManip_Lab.assets.d1_piper_l.d1_piper_articulation_cfg import D1_PIPER_CFG
from LeggedManip_Lab.assets.d1_piper_l.ik_helpers import spawn_d1_piper_ik

from ... import mdp
from .d1_piper_common_env_cfg import (
    CommandsCfg,
    D1_LEG_JOINT_NAMES,
    D1_LEG_POS_JOINT_NAMES,
    D1_WHEEL_NAMES,
    D1PiperBaseEnvCfg,
)

@configclass
class D1PiperIKEnvCfg(D1PiperBaseEnvCfg):
    """16-D leg/wheel policy with the standalone script's arm IK from step one.

    The arm runs at 200 Hz; the leg/wheel policy runs at 50 Hz. Unlike the
    fixed-base diagnostic, training keeps the base free and tracks successive
    pose commands continuously, resetting the arm only at episode boundaries.
    """

    # Unlike WBC, this task uses the standard body-pose command.  Its complete
    # 7-D command (position + quaternion) is relative to piper_base_link.
    commands: CommandsCfg = CommandsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.robot = deepcopy(D1_PIPER_CFG).replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        # Correct helper-frame payloads before physics replication, without
        # modifying the original USD, shared D1_PIPER_CFG, or WBC task.
        self.scene.robot.spawn.func = spawn_d1_piper_ik
        self.scene.robot.spawn.articulation_props.fix_root_link = False
        self.sim.dt = 0.005
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.scene.contact_forces.update_period = self.sim.dt

        # Use modest randomization while learning balance with the arm load.
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (0.0, 2.0)
        self.events.add_base_com.params["com_range"] = {
            axis: (-0.01, 0.01) for axis in ("x", "y", "z")
        }
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.9, 1.1)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.9, 1.1)
        # Keep the arm's tested PD gains; randomize only policy-controlled joints.
        self.events.randomize_actuator_gains.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=D1_LEG_JOINT_NAMES
        )
        self.events.physics_material.params["static_friction_range"] = (0.6, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.6, 1.0)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.0)

        # Match the diagnostic's full uniform pose range in piper_base_link.
        # Independent XYZ/RPY limits do not guarantee every pose is reachable.
        self.commands.ee_pose.limit_ranges = mdp.command_cfg.UniformPoseCommandCfg.Ranges(
            pos_x=(0.15, 0.50),
            pos_y=(-0.30, 0.30),
            pos_z=(0.10, 0.55),
            roll=(-0.45, 0.45),
            pitch=(0.90, 2.00),
            yaw=(-0.60, 0.60),
        )
        self.commands.ee_pose.ranges = deepcopy(self.commands.ee_pose.limit_ranges)
        self.commands.ee_pose.resampling_time_range = (3.0, 3.0)
        self.commands.ee_pose.curriculum_enabled = False
        self.curriculum.pos_cmd_levels = None

        # The same bent reset pose as run_d1_piper_diff_ik.py avoids starting
        # pose IK at the neutral configuration's singularity and joint limits.
        self.scene.robot.init_state.joint_pos.pop("joint[1-6]", None)
        self.scene.robot.init_state.joint_pos.update(
            {
                "joint1": 0.0,
                "joint2": 1.0,
                "joint3": -1.5,
                "joint4": 0.0,
                "joint5": 0.5,
                "joint6": 0.0,
            }
        )

        # The command generator stores [p, q] in the arm-base frame.  Keep
        # this explicit here so future changes to the shared command defaults
        # cannot silently move the IK target to another frame.
        self.commands.ee_pose.root_name = "piper_base_link"

        # The arm is driven entirely by the command-driven IK action.  Keep
        # only the 16 leg/wheel joints in the actor's proprioceptive state.
        # The critic is narrowed below to the same leg/wheel state; arm
        # states are omitted because the arm is command-driven by IK; the
        # actor retains the desired end-effector pose.  Wheel positions
        # remain zero-valued placeholders, preserving the existing D1
        # observation convention.
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=D1_LEG_JOINT_NAMES, preserve_order=True
        )
        self.observations.policy.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=D1_WHEEL_NAMES, preserve_order=True
        )
        self.observations.policy.joint_vel.params["joint_names"] = D1_LEG_JOINT_NAMES
        # Do not expose the command-driven arm action in the actor history.
        # This returns the eight leg/wheel terms (16 values) explicitly.
        self.observations.policy.actions.func = mdp.last_leg_action

        # The IK arm is driven from the end-effector command and is not part
        # of the policy action.  Keep the privileged critic focused on the
        # quantities available to the locomotion policy: remove the arm
        # command and arm end-effector pose, and select only the 16 D1
        # leg/wheel joints from mixed joint observations.  This override is
        # local to the IK task; the shared CriticCfg remains unchanged for
        # the WBC and other D1-Piper tasks.
        self.observations.critic.pos_commands = None
        self.observations.critic.ee_link0_rel_pose = None
        self.observations.critic.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=D1_LEG_JOINT_NAMES, preserve_order=True
        )
        self.observations.critic.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=D1_WHEEL_NAMES, preserve_order=True
        )
        self.observations.critic.joint_vel.params["joint_names"] = D1_LEG_JOINT_NAMES
        self.observations.critic.joint_torques.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=D1_LEG_JOINT_NAMES, preserve_order=True
        )
        # The arm action term has zero policy dimensions.  Select the leg
        # action terms explicitly so this remains true if an arm term is
        # added or changed in a future configuration.
        self.observations.critic.actions.func = mdp.last_leg_action

        # JointAction clips *after* scaling and adding the default pose.
        # Bound absolute leg targets to +/-0.5 rad about the reset posture,
        # and wheel targets to +/-15 rad/s, including non-RSL callers.
        for leg in ("fl", "fr", "rl", "rr"):
            position_action = getattr(self.actions, f"joint_pos_{leg}")
            position_action.clip = {
                name: (
                    self.scene.robot.init_state.joint_pos[name] - 0.5,
                    self.scene.robot.init_state.joint_pos[name] + 0.5,
                )
                for name in position_action.joint_names
            }
            getattr(self.actions, f"joint_vel_{leg}").clip = {".*": (-15.0, 15.0)}

        # Keep the four leg position and four wheel velocity terms.  The arm
        # term below consumes zero policy dimensions and is driven directly by
        # commands.ee_pose.
        self.actions.joint_pos = mdp.CommandDifferentialIKGravityCompensationActionCfg(
            asset_name="robot",
            joint_names=["joint[1-6]"],
            body_name="end_effector",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
                ik_params={"lambda_val": 0.02},
            ),
            # Identity offset: the target frame is the actual end_effector
            # body, so no additional tool-frame transform is required.
            body_offset=mdp.CommandDifferentialIKGravityCompensationActionCfg.OffsetCfg(),
            gravity_compensation=True,
            command_name="ee_pose",
            reference_frame_name="piper_base_link",
            # Start tracking immediately, as in the standalone diagnostic.
            arm_control_start_iteration=None,
            steps_per_iteration=24,
            max_joint_target_velocity=1.0,
            debug_vis=False,
        )

        # The WBC task uses a mixed frame for historical reasons.  IK uses the
        # fully piper-base-relative command and matching reward functions.
        self.rewards.end_effector_position_tracking_exp.func = mdp.position_command_b_error_exp
        self.rewards.end_effector_position_tracking_exp.params["link0_name"] = "piper_base_link"
        self.rewards.end_effector_orientation_tracking.func = mdp.orientation_command_b_error
        self.rewards.end_effector_orientation_tracking.params["link0_name"] = "piper_base_link"
        self.rewards.end_effector_position_tracking_exp.weight = 0
        self.rewards.end_effector_orientation_tracking.weight = 0

        self.rewards.joint_power.weight = -2e-5  # powers
        self.rewards.track_lin_vel_xy_exp.weight = 3.5  # tracking_lin_vel
        self.rewards.track_ang_vel_z_exp.weight = 1.5  # tracking_ang_vel
        self.rewards.ang_vel_xy_l2.weight = -0.05  # ang_vel_xy
        self.rewards.dof_pos_limits.weight = -2.0 #-2.0  # dof_pos_limits
        self.rewards.joint_acc_l2.weight = -2.5e-7  # dof_acc
        #self.rewards.undesired_contacts.weight = -1.0  # collision
        self.rewards.action_rate_l2.weight = -0.01  # action_rate
        self.rewards.stand_still.weight = -1.0
        self.rewards.stand_nice.weight = -2.0
        # The command-driven arm is not a policy output.  Do not ask the
        # locomotion policy to undo its commanded motion or return it to zero.
        self.rewards.arm_deviation.weight = 0.0
        self.rewards.joint_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=D1_LEG_POS_JOINT_NAMES
        )
        self.rewards.dof_pos_limits.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=D1_LEG_POS_JOINT_NAMES
        )
        for reward in (self.rewards.dof_torques_l2, self.rewards.joint_power):
            reward.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=D1_LEG_JOINT_NAMES)
        #self.rewards.dont_wait.weight = -0.5  # dont_wait
        # self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -0.2, HIP_JOINT_NAMES)
        # self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_l1", -0.1, [".*thigh_joint", ".*calf_joint"])
        self.rewards.upward.weight = 0.25
        #self.rewards.lin_vel_z_l2.weight = 0
        #self.rewards.feet_gait.weight = 1.0  # 鼓励 步态
        self.rewards.wheel_body_speed_match.weight = -0.2  # 鼓励轮地速度匹配
        #self.rewards.joint_mirror.weight = -0.1  # 鼓励左右对称

        self.rewards.flat_orientation_l2.weight = -1.0 #-0.25  # 鼓励
        self.rewards.base_height.weight = -1.0  # 鼓励保持合适高度

        self.disable_zero_weight_rewards()
        


@configclass
class D1PiperIKEnvCfg_PLAY(D1PiperIKEnvCfg):
    """Play configuration for the D1-Piper differential-IK task."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Keep the same lightweight, deterministic playback settings as the
        # original WBC play task.
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.0
        self.observations.policy.enable_corruption = False
        # Match training: full pose workspace and IK active from the first step.
        self.actions.joint_pos.arm_control_start_iteration = None
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.commands.ee_pose.ranges = deepcopy(self.commands.ee_pose.limit_ranges)
