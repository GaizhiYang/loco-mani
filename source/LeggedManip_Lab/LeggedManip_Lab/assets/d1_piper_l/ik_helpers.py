"""Scene-local D1-Piper model corrections shared by IK debugging and training."""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

from pxr import Gf, UsdPhysics

import isaaclab.sim as sim_utils


def fix_d1_piper_helper_frames(stage, robot_prim_paths, usd_path):
    """Remove fallback payloads from empty URDF frames before physics starts.

    Keep positive epsilon masses/inertias so PhysX does not infer fallback
    values, and keep both bodies addressable for end-effector tracking. Only
    author overrides in the current stage; never save the source USD layers.
    """

    urdf_path = Path(usd_path).parent.parent / "d1_piper_l.urdf"
    links = {link.get("name"): link for link in ET.parse(urdf_path).getroot().findall("link")}
    for name in ("end_effector", "gripper_link"):
        # Refuse to discard a real payload if the source description changes.
        if name not in links or len(links[name]) != 0:
            raise RuntimeError(f"Refusing to override {name}: the URDF link is not an empty helper frame.")
        for robot_path in robot_prim_paths:
            path = f"{robot_path}/{name}"
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError(f"Missing rigid helper frame: {path}")
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_api.CreateMassAttr(1.0e-4)
            mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(1.0e-8))
        logging.getLogger(__name__).info("Scene-only helper frame override: %s, mass=0.0001 kg", name)


@sim_utils.clone
def spawn_d1_piper_ik(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn and correct the source robot before cloning/physics replication.

    The outer clone decorator passes a concrete path to the standard USD
    spawner. All subsequent clones inherit the corrected mass/inertia, without
    disabling physics replication or changing WBC's shared asset defaults.
    """

    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    fix_d1_piper_helper_frames(prim.GetStage(), [str(prim.GetPath())], cfg.usd_path)
    return prim
