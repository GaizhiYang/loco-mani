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
feed-forward.  The actor policy observation likewise contains only the 16
leg/wheel joint states and 16 leg/wheel action history; the arm joint states
and command-driven arm action are reserved for the privileged critic/control
path.
"""

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ... import mdp
from .d1_piper_common_env_cfg import CommandsCfg, D1_LEG_JOINT_NAMES, D1_WHEEL_NAMES
from .wbc_env_cfg import D1PiperWBCEnvCfg
from .d1_piper_common_env_cfg import D1PiperBaseEnvCfg

from .d1_piper_common_env_cfg import D1PiperBaseEnvCfg
#from LeggedManip_Lab.assets.go2_piper.go2_piper_articulation_cfg import GO2_PIPER_CFG
from LeggedManip_Lab.assets.d1_piper_l.d1_piper_articulation_cfg import D1_PIPER_CFG
#from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.leggedmanip_lab_env_cfg import *
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.d1_piper_l.d1_piper_common_env_cfg import *

@configclass
class D1PiperIKEnvCfg(D1PiperBaseEnvCfg):
    """D1-Piper task with 16-D leg policy and command-driven arm IK."""

    # Unlike WBC, this task uses the standard body-pose command.  Its complete
    # 7-D command (position + quaternion) is relative to piper_base_link.
    commands: CommandsCfg = CommandsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

                # scene
        self.scene.robot: ArticulationCfg = D1_PIPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )

        # observations

        # curriculum

        # events
        #self.events.push_robot = None
        self.events.randomize_rigid_body_inertia = None
        self.events.reset_robot_joints = None

        # commands
        self.commands.ee_pose.curriculum_enabled = True

        # The all-zero Piper pose has a nearly singular 6-D pose Jacobian.
        # Start this IK task from the same interior, bent configuration used
        # by the standalone IK diagnostic.  This only changes the IK task;
        # the flat/WBC configurations retain their original reset pose.
        self.scene.robot.init_state.joint_pos.pop("joint[1-6]", None)
        self.scene.robot.init_state.joint_pos.update(
            {
                "joint1": 0.0,
                "joint2": 0.5,
                "joint3": -0.8,
                "joint4": 0.0,
                "joint5": 0.3,
                "joint6": 0.0,
            }
        )

        # The command generator stores [p, q] in the arm-base frame.  Keep
        # this explicit here so future changes to the shared command defaults
        # cannot silently move the IK target to another frame.
        self.commands.ee_pose.root_name = "piper_base_link"

        # The arm is driven entirely by the command-driven IK action.  Keep
        # only the 16 leg/wheel joints in the actor's proprioceptive state;
        # the privileged critic retains the full state from the common
        # configuration.  Wheel positions remain zero-valued placeholders,
        # preserving the existing D1 observation convention.
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
                ik_params={"lambda_val": 0.05},
            ),
            # Identity offset: the target frame is the actual end_effector
            # body, so no additional tool-frame transform is required.
            body_offset=mdp.CommandDifferentialIKGravityCompensationActionCfg.OffsetCfg(),
            gravity_compensation=True,
            command_name="ee_pose",
            reference_frame_name="piper_base_link",
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
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.commands.ee_pose.ranges = self.commands.ee_pose.limit_ranges
        self.commands.ee_pose.limit_ranges.pos_y = (0.0, 0.0)
