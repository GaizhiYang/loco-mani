# Copyright (c) 2025-2026, Junjie Zhu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Open the D1 + Piper-L URDF in Isaac Sim for visual inspection.

This script is intentionally independent of the training environments. It
imports assets/d1_piper_l/d1_piper_l.urdf directly, opens a GUI viewport by
default, and enables PhysX collider visualization so the arm mounting position
and collision geometry can be checked quickly.

Usage:

    ./isaaclab.sh -p scripts/view_d1_piper_l_urdf.py

Useful options:

    ./isaaclab.sh -p scripts/view_d1_piper_l_urdf.py --force-usd-conversion
    ./isaaclab.sh -p scripts/view_d1_piper_l_urdf.py --collision-from-visuals
    ./isaaclab.sh -p scripts/view_d1_piper_l_urdf.py --no-fix-base
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = REPO_ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/assets/d1_piper_l/d1_piper_l.urdf"
DEFAULT_USD_DIR = REPO_ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/assets/d1_piper_l/generated_usd"


def _path_arg(value: str) -> Path:
    """Resolve CLI paths relative to the repository root."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


parser = argparse.ArgumentParser(description="Visualize the D1 + Piper-L URDF in Isaac Sim.")
parser.add_argument("--urdf", type=_path_arg, default=DEFAULT_URDF, help="Path to the URDF to import.")
parser.add_argument(
    "--usd-dir",
    type=_path_arg,
    default=DEFAULT_USD_DIR,
    help="Directory where the imported USD asset will be generated.",
)
parser.add_argument(
    "--prim-path",
    type=str,
    default="/World/D1PiperL",
    help="Stage prim path where the imported robot will be spawned.",
)
parser.add_argument(
    "--spawn-z",
    type=float,
    default=0.45,
    help="Spawn height for the robot root. This does not change the URDF mounting transform.",
)
parser.add_argument(
    "--fix-base",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Fix the robot root to the world for easier static inspection.",
)
parser.add_argument(
    "--merge-fixed-joints",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Merge URDF fixed joints during import. Disabled by default to preserve mount links for inspection.",
)
parser.add_argument(
    "--make-instanceable",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Create an instanceable USD asset. Disabled by default to make collider debugging easier.",
)
parser.add_argument(
    "--self-collision",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Enable self-collision during URDF import.",
)
parser.add_argument(
    "--collision-from-visuals",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Generate collision geometry from visual meshes if authored collisions are insufficient.",
)
parser.add_argument(
    "--collision-type",
    type=str,
    default="Convex Hull",
    choices=("Convex Hull", "Convex Decomposition", "Bounding Sphere", "Bounding Cube"),
    help="Collision approximation used when the importer supports the modern collision_type field.",
)
parser.add_argument(
    "--collider-type",
    type=str,
    default="convex_hull",
    choices=("convex_hull", "convex_decomposition"),
    help="Collision approximation fallback for Isaac Lab versions using the older collider_type field.",
)
parser.add_argument(
    "--force-usd-conversion",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Regenerate the intermediate USD on every run so URDF edits are reflected immediately.",
)
parser.add_argument(
    "--disable-collider-display",
    action="store_true",
    help="Do not enable the PhysX collider viewport overlay automatically.",
)
parser.add_argument(
    "--show-collision-geometry",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Render imported collision prims directly as highlighted geometry.",
)
parser.add_argument(
    "--collision-only",
    action="store_true",
    help="Hide visual meshes and show collision geometry only.",
)
parser.add_argument(
    "--collision-opacity",
    type=float,
    default=0.45,
    help="Opacity of directly rendered collision geometry, from 0.0 to 1.0.",
)
parser.add_argument(
    "--dump-collision-paths",
    action="store_true",
    help="Print Piper prim paths and applied schemas for collision debugging.",
)
parser.add_argument(
    "--camera-eye",
    type=float,
    nargs=3,
    default=(1.25, -1.45, 0.95),
    metavar=("X", "Y", "Z"),
    help="Camera eye position.",
)
parser.add_argument(
    "--camera-target",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.18),
    metavar=("X", "Y", "Z"),
    help="Camera target position.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Prefer a visible Isaac Sim window unless the user explicitly passes --headless.
parser.set_defaults(headless=False)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import carb
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext


def _supported_cfg_fields(cfg_cls: type) -> set[str]:
    """Return dataclass/configclass fields when available."""
    return set(getattr(cfg_cls, "__dataclass_fields__", {}).keys())


def _make_joint_drive_cfg() -> object | None:
    """Create the joint-drive config required by recent Isaac Lab releases."""
    try:
        converter_cfg_cls = sim_utils.UrdfConverterCfg
        return converter_cfg_cls.JointDriveCfg(
            gains=converter_cfg_cls.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
        )
    except AttributeError:
        # Older releases did not require a nested gains configuration.
        return None


def _make_urdf_cfg() -> object:
    """Build a UrdfFileCfg while tolerating small Isaac Lab API differences."""
    try:
        from isaaclab.sim.spawners.from_files import UrdfFileCfg
    except ImportError:
        # Older Isaac Lab versions also expose spawner configs from isaaclab.sim.
        UrdfFileCfg = sim_utils.UrdfFileCfg

    fields = _supported_cfg_fields(UrdfFileCfg)
    kwargs = {
        "asset_path": str(args_cli.urdf),
        "usd_dir": str(args_cli.usd_dir),
        "usd_file_name": "d1_piper_l.usd",
        "force_usd_conversion": args_cli.force_usd_conversion,
        "make_instanceable": args_cli.make_instanceable,
        "fix_base": args_cli.fix_base,
        "merge_fixed_joints": args_cli.merge_fixed_joints,
        "self_collision": args_cli.self_collision,
        "collision_from_visuals": args_cli.collision_from_visuals,
        "joint_drive": _make_joint_drive_cfg(),
    }

    # Isaac Lab 3.x uses collision_type, while some 2.x releases use collider_type.
    if "collision_type" in fields:
        kwargs["collision_type"] = args_cli.collision_type
    elif "collider_type" in fields:
        kwargs["collider_type"] = args_cli.collider_type

    # Keep only fields known to this installed Isaac Lab when metadata is available.
    if fields:
        kwargs = {key: value for key, value in kwargs.items() if key in fields}

    try:
        return UrdfFileCfg(**kwargs)
    except TypeError:
        # Last-resort fallback for config classes that do not expose fields.
        minimal_kwargs = {
            "asset_path": str(args_cli.urdf),
            "usd_dir": str(args_cli.usd_dir),
            "force_usd_conversion": args_cli.force_usd_conversion,
            "fix_base": args_cli.fix_base,
        }
        joint_drive_cfg = _make_joint_drive_cfg()
        if joint_drive_cfg is not None:
            minimal_kwargs["joint_drive"] = joint_drive_cfg
        return UrdfFileCfg(**minimal_kwargs)


def _spawn_robot() -> None:
    """Import and spawn the D1 + Piper-L URDF."""
    if not args_cli.urdf.exists():
        raise FileNotFoundError(f"URDF not found: {args_cli.urdf}")

    args_cli.usd_dir.mkdir(parents=True, exist_ok=True)
    cfg = _make_urdf_cfg()

    try:
        from isaaclab.sim.spawners.from_files import spawn_from_urdf

        spawn_from_urdf(args_cli.prim_path, cfg, translation=(0.0, 0.0, args_cli.spawn_z))
    except ImportError:
        # Some Isaac Lab versions rely on the spawner function attached to the cfg.
        cfg.func(args_cli.prim_path, cfg, translation=(0.0, 0.0, args_cli.spawn_z))


def _enable_collider_display() -> None:
    """Enable collision body visualization in the active Isaac Sim viewport."""
    if args_cli.disable_collider_display:
        return

    settings = carb.settings.get_settings()
    # Equivalent to Viewport eye icon -> Show By Type -> Physics -> Colliders -> All.
    settings.set("/persistent/physics/visualizationDisplayColliders", 2)
    # Solid render meshes for collision mesh approximations.
    settings.set("/persistent/physics/visualizationCollisionMesh", True)
    settings.set("/persistent/physics/visualizationDisplayColliderNormals", False)
    settings.set("/physics/developmentMode", True)


def _show_collision_geometry() -> tuple[int, int]:
    """Make imported collision prims visible and color them red."""
    if not args_cli.show_collision_geometry:
        return 0, 0

    import omni.usd
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    robot_prim = stage.GetPrimAtPath(args_cli.prim_path)
    if not robot_prim.IsValid():
        raise RuntimeError(f"Spawned robot prim not found: {args_cli.prim_path}")

    collision_shape_count = 0
    piper_collision_shape_count = 0
    piper_link_names = {
        "piper_base_link",
        "link1",
        "link2",
        "link3",
        "link4",
        "link5",
        "link6",
        "flange_link",
        "gripper_base",
        "gripper_link",
        "gripper_link1",
        "gripper_link2",
    }

    for prim in Usd.PrimRange(robot_prim, Usd.TraverseInstanceProxies()):
        prim_path = str(prim.GetPath())
        path_parts = set(prim_path.split("/"))
        is_collision_branch = "/collisions" in prim_path
        has_collision_api = prim.HasAPI(UsdPhysics.CollisionAPI)

        if args_cli.dump_collision_paths and path_parts & piper_link_names:
            print(
                f"[COLLISION-DEBUG] {prim_path} type={prim.GetTypeName()} "
                f"instance_proxy={prim.IsInstanceProxy()} schemas={prim.GetAppliedSchemas()}"
            )

        if args_cli.collision_only and "/visuals" in prim_path:
            imageable = UsdGeom.Imageable(prim)
            if imageable and not prim.IsInstanceProxy():
                imageable.MakeInvisible()

        if not (is_collision_branch or has_collision_api):
            continue

        imageable = UsdGeom.Imageable(prim)
        if imageable and not prim.IsInstanceProxy():
            imageable.MakeVisible()
            imageable.CreatePurposeAttr().Set("default")

        gprim = UsdGeom.Gprim(prim)
        if gprim and not prim.IsInstanceProxy():
            gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.05, 0.05)])
            gprim.CreateDisplayOpacityAttr().Set([max(0.0, min(1.0, args_cli.collision_opacity))])

        if has_collision_api:
            collision_shape_count += 1
            if path_parts & piper_link_names:
                piper_collision_shape_count += 1

    return collision_shape_count, piper_collision_shape_count


def design_scene() -> None:
    """Create a minimal inspection scene."""
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    cfg_light = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg_light.func("/World/Light", cfg_light)

    _spawn_robot()


def main() -> None:
    """Run the visualization loop."""
    print(f"[INFO] Loading URDF: {args_cli.urdf}")
    print(f"[INFO] Generated USD directory: {args_cli.usd_dir}")
    print("[INFO] Collider overlay: " + ("disabled" if args_cli.disable_collider_display else "enabled"))

    sim_cfg = SimulationCfg(dt=1.0 / 60.0, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=list(args_cli.camera_eye), target=list(args_cli.camera_target))

    design_scene()
    sim.reset()
    # Apply the debug settings after PhysX and the viewport have initialized.
    _enable_collider_display()
    collision_count, piper_collision_count = _show_collision_geometry()
    print(
        f"[INFO] Imported collision shapes: total={collision_count}, "
        f"piper={piper_collision_count}"
    )

    print("[INFO] Scene ready. Close the Isaac Sim window or press Ctrl+C to exit.")
    if not args_cli.disable_collider_display:
        print("[INFO] If colliders are hidden by the UI, toggle: eye icon -> Show By Type -> Physics -> Colliders -> All.")

    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()
