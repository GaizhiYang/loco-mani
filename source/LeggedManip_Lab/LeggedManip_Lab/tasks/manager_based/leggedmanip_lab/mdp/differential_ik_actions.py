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

"""Project-side Differential IK actions used by the D1-Piper tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_inv,
    skew_symmetric_matrix,
    subtract_frame_transforms,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DifferentialInverseKinematicsGravityCompensationAction(DifferentialInverseKinematicsAction):
    """Official Differential IK action plus joint gravity torque feed-forward.

    The inherited action term performs the full Isaac Lab Differential IK flow:
    action scaling/clipping, root-frame pose extraction, Jacobian selection,
    and ``DifferentialIKController.compute``.  This class only adds
    ``tau_gravity`` to the articulation's effort target after the position
    target has been computed.  The D1-Piper arm uses explicit PD actuators, so
    the actuator applies

    ``tau = Kp * (q_des - q) + Kd * (dq_des - dq) + tau_gravity``.

    PhysX exposes the gravity vector in generalized-DOF order.  For a floating
    articulation the first six entries are the base DOFs; for a fixed-base
    articulation there is no such offset.  The shape check below keeps this
    compatible with both layouts.
    """

    cfg: DifferentialInverseKinematicsGravityCompensationActionCfg

    def __init__(self, cfg: DifferentialInverseKinematicsGravityCompensationActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        if isinstance(self._joint_ids, slice):
            self._joint_ids_tensor = torch.arange(self._asset.num_joints, device=self.device, dtype=torch.long)
        else:
            self._joint_ids_tensor = torch.as_tensor(self._joint_ids, device=self.device, dtype=torch.long)
        self._gravity_torque = torch.zeros(
            self.num_envs, self._joint_ids_tensor.numel(), device=self.device, dtype=self._asset.data.joint_pos.dtype
        )

    def apply_actions(self) -> None:
        """Compute IK position targets and add gravity feed-forward effort."""

        # Keep the exact Isaac Lab Differential IK implementation as the
        # position-control path.
        super().apply_actions()

        self._apply_gravity_compensation()

    def _apply_gravity_compensation(self) -> None:
        """Add PhysX's gravity-compensation torques to the selected joints."""

        if not self.cfg.gravity_compensation:
            self._gravity_torque.zero_()
        else:
            rigid_props = getattr(self._asset.cfg.spawn, "rigid_props", None)
            gravity_disabled = rigid_props is not None and rigid_props.disable_gravity is True
            if gravity_disabled:
                self._gravity_torque.zero_()
            else:
                gravity = self._asset.root_physx_view.get_gravity_compensation_forces()

                # Depending on the PhysX articulation view, gravity may be
                # returned in joint-only order or generalized (base + joint)
                # order.  Detect the layout instead of hard-coding one API
                # variant.
                if gravity.shape[1] == self._asset.num_joints:
                    gravity_ids = self._joint_ids_tensor
                elif not self._asset.is_fixed_base and gravity.shape[1] >= self._asset.num_joints + 6:
                    gravity_ids = self._joint_ids_tensor + 6
                else:
                    raise RuntimeError(
                        "Unexpected gravity-compensation tensor shape: "
                        f"{tuple(gravity.shape)} for {self._asset.num_joints} articulation joints."
                    )

                self._gravity_torque[:] = gravity[:, gravity_ids]

        # Articulation actuators combine this effort with the inherited PD
        # position target.  This does not replace the IK position command.
        self._asset.set_joint_effort_target(self._gravity_torque, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._gravity_torque.zero_()
        else:
            self._gravity_torque[env_ids] = 0.0


@configclass
class DifferentialInverseKinematicsGravityCompensationActionCfg(DifferentialInverseKinematicsActionCfg):
    """Differential IK action configuration with optional joint gravity feed-forward."""

    class_type: type = DifferentialInverseKinematicsGravityCompensationAction
    gravity_compensation: bool = True
    """Whether to add PhysX's gravity-compensation effort to the PD target."""


class CommandDifferentialIKGravityCompensationAction(
    DifferentialInverseKinematicsGravityCompensationAction
):
    """Command-driven Differential IK with gravity feed-forward.

    This action term deliberately has zero policy action dimensions.  It reads
    the ``ee_pose`` command from the environment's command manager, where the
    target pose is expressed in ``reference_frame_name`` (the Piper arm base),
    transforms the measured pose and PhysX Jacobian to that same frame, and
    solves IK every simulation step.  The leg and wheel action terms therefore
    remain the only policy outputs.
    """

    cfg: "CommandDifferentialIKGravityCompensationActionCfg"

    @property
    def action_dim(self) -> int:
        """No policy output is consumed by this command-driven arm term."""

        return 0

    def __init__(self, cfg: "CommandDifferentialIKGravityCompensationActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)

        reference_body_ids, reference_body_names = self._asset.find_bodies(cfg.reference_frame_name)
        if len(reference_body_ids) != 1:
            raise ValueError(
                f"Expected one reference frame named '{cfg.reference_frame_name}'. "
                f"Found {len(reference_body_ids)}: {reference_body_names}."
            )
        self._reference_body_idx = reference_body_ids[0]
        self._env = env
        self._joint_pos_des = torch.zeros_like(self._asset.data.joint_pos[:, self._joint_ids])

    def process_actions(self, actions: torch.Tensor) -> None:
        """Consume the empty action slice; the command manager supplies the target."""

        if actions.shape[-1] != 0:
            raise ValueError(
                "CommandDifferentialIKGravityCompensationAction expects a zero-width policy action slice, "
                f"got shape {tuple(actions.shape)}."
            )

    def apply_actions(self) -> None:
        """Solve command-space IK and apply PD position plus gravity feed-forward."""

        # All three IK quantities below are expressed in piper_base_link:
        # current EE pose, desired command pose, and geometric Jacobian.
        ee_pos_piper, ee_quat_piper = self._compute_reference_frame_pose()
        command_pose = self._env.command_manager.get_command(self.cfg.command_name)
        if command_pose.shape[-1] != 7:
            raise ValueError(
                f"Command '{self.cfg.command_name}' must contain position + quaternion (7 values), "
                f"got shape {tuple(command_pose.shape)}."
            )

        # The command is already [p_piper_ee, q_piper_ee].  The controller is
        # frame-agnostic, so passing this command is valid because the current
        # pose and Jacobian below use exactly the same piper-base frame.
        self._ik_controller.set_command(command_pose)

        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        jacobian = self._compute_reference_frame_jacobian()
        self._joint_pos_des[:] = self._ik_controller.compute(
            ee_pos_piper, ee_quat_piper, jacobian, joint_pos
        )
        self._asset.set_joint_position_target(self._joint_pos_des, joint_ids=self._joint_ids)
        self._apply_gravity_compensation()

    def _compute_reference_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the end-effector pose relative to ``piper_base_link``."""

        ee_pos_w = self._asset.data.body_pos_w[:, self._body_idx]
        ee_quat_w = self._asset.data.body_quat_w[:, self._body_idx]
        reference_pos_w = self._asset.data.body_pos_w[:, self._reference_body_idx]
        reference_quat_w = self._asset.data.body_quat_w[:, self._reference_body_idx]
        ee_pos_piper, ee_quat_piper = subtract_frame_transforms(
            reference_pos_w,
            reference_quat_w,
            ee_pos_w,
            ee_quat_w,
        )

        # The D1 configuration uses the actual end_effector body as the target
        # frame. Keep the inherited offset support for future tool frames.
        if self.cfg.body_offset is not None:
            ee_pos_piper, ee_quat_piper = combine_frame_transforms(
                ee_pos_piper,
                ee_quat_piper,
                self._offset_pos,
                self._offset_rot,
            )
        return ee_pos_piper, ee_quat_piper

    def _compute_reference_frame_jacobian(self) -> torch.Tensor:
        """Return the arm Jacobian with linear/angular rows expressed in piper frame.

        PhysX returns the geometric Jacobian in world axes. Since the selected
        columns contain only arm joints, ``piper_base_link`` has zero velocity
        with respect to those columns. Therefore changing the expressed frame
        only requires rotating both linear and angular rows by
        ``R_world_piper.T``. No root-motion columns are included here.
        """

        jacobian_w = self.jacobian_w.clone()
        reference_quat_w = self._asset.data.body_quat_w[:, self._reference_body_idx]
        world_to_piper = matrix_from_quat(quat_inv(reference_quat_w))
        jacobian_piper = jacobian_w.clone()
        jacobian_piper[:, :3, :] = torch.bmm(world_to_piper, jacobian_w[:, :3, :])
        jacobian_piper[:, 3:, :] = torch.bmm(world_to_piper, jacobian_w[:, 3:, :])

        # If a future configuration controls an offset tool frame instead of
        # the actual end_effector body, account for that offset in the same
        # piper-frame Jacobian convention used by IsaacLab's action term.
        if self.cfg.body_offset is not None:
            jacobian_piper[:, :3, :] += torch.bmm(
                -skew_symmetric_matrix(self._offset_pos), jacobian_piper[:, 3:, :]
            )
            jacobian_piper[:, 3:, :] = torch.bmm(
                matrix_from_quat(self._offset_rot), jacobian_piper[:, 3:, :]
            )
        return jacobian_piper

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._joint_pos_des.zero_()
        else:
            self._joint_pos_des[env_ids] = 0.0


@configclass
class CommandDifferentialIKGravityCompensationActionCfg(
    DifferentialInverseKinematicsGravityCompensationActionCfg
):
    """Configuration for command-driven arm IK with no arm policy outputs."""

    class_type: type = CommandDifferentialIKGravityCompensationAction
    scale: tuple = ()
    clip: dict | None = None
    command_name: str = "ee_pose"
    reference_frame_name: str = "piper_base_link"
