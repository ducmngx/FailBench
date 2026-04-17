#!/usr/bin/env python
"""Precompute GraspGen 6-DoF grasps for every mesh-backed pick target.

Scans every ``scenes/<scene>/tasks.yaml`` for ``grasped_object`` references,
resolves each to its mesh file via the scene XML, dedupes by SHA256 of file
bytes, and invokes GraspGen once per unique mesh. Results land in
``cache/graspgen/<sha>.yml`` plus an ``index.json`` that the failbench_env
loader reads.

Runs from ``failbench_env`` — shells out to the GraspGen venv for inference
so no torch/CUDA is needed in the orchestrator process.

Usage:
    python scripts/precompute_grasps.py            # all scenes with tasks.yaml
    python scripts/precompute_grasps.py --scene scene_kitchen
    python scripts/precompute_grasps.py --mesh scenes/scene_kitchen/assets/cubesmall.stl
    python scripts/precompute_grasps.py --force    # regenerate even if cached
"""

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE_DIR = os.path.join(_REPO_ROOT, "cache", "graspgen")
_INDEX_PATH = os.path.join(_CACHE_DIR, "index.json")
_GRASPGEN_PY = os.path.join(_REPO_ROOT, "external/GraspGen/.venv/bin/python")
_GRASPGEN_SCRIPT = os.path.join(_REPO_ROOT, "external/GraspGen/scripts/demo_object_mesh.py")
_GRASPGEN_CFG = os.path.join(
    _REPO_ROOT, "external/GraspGen/checkpoints/checkpoints/graspgen_franka_panda.yml"
)
_CUDA_HOME = "/usr/local/cuda-12.2"


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_mesh_file(scene_dir: str, scene_xml: str, body_name: str) -> str | None:
    """Return absolute path to the visual mesh for ``body_name``, or None if primitive."""
    tree = ET.parse(scene_xml)
    root = tree.getroot()

    # meshdir lives in the root scene.xml OR in any <include>d file (MuJoCo
    # resolves compiler tags relative to the top-level scene). Conventionally
    # scenes put meshes in scenes/<scene>/assets/, so try that first.
    candidates = [os.path.join(scene_dir, "assets"), scene_dir]
    compiler = root.find("compiler")
    if compiler is not None and compiler.get("meshdir"):
        candidates.insert(0, os.path.normpath(os.path.join(scene_dir, compiler.get("meshdir"))))

    body = None
    for b in root.iter("body"):
        if b.get("name") == body_name:
            body = b
            break
    if body is None:
        return None

    # Find a geom with a mesh attribute (skip collision-only geoms when possible)
    mesh_name = None
    for geom in body.findall("geom"):
        if geom.get("mesh"):
            mesh_name = geom.get("mesh")
            # Prefer the visual geom (typically listed first, no _c suffix)
            if not mesh_name.endswith(("_c0", "_c1", "_c2", "_c3", "_c4")):
                break
    if mesh_name is None:
        return None  # primitive (box/cylinder/...)

    for m in root.iter("mesh"):
        if m.get("name") == mesh_name:
            f = m.get("file")
            for cand in candidates:
                p = os.path.normpath(os.path.join(cand, f))
                if os.path.exists(p):
                    return p
            return os.path.normpath(os.path.join(candidates[0], f))
    return None


def _inventory(scenes: list[str]) -> dict[str, str]:
    """Return {mesh_abs_path: rel_path_from_repo} for every mesh-backed pick target."""
    out: dict[str, str] = {}
    for scene in scenes:
        tasks_path = os.path.join(_REPO_ROOT, "scenes", scene, "tasks.yaml")
        scene_xml = os.path.join(_REPO_ROOT, "scenes", scene, "scene.xml")
        if not (os.path.exists(tasks_path) and os.path.exists(scene_xml)):
            continue
        with open(tasks_path) as f:
            tasks_yaml = yaml.safe_load(f)

        grasped_objects = {tasks_yaml.get("grasped_object", "object3")}
        for tdef in tasks_yaml.get("tasks", {}).values():
            if "grasped_object" in tdef:
                grasped_objects.add(tdef["grasped_object"])

        scene_dir = os.path.dirname(scene_xml)
        for body_name in grasped_objects:
            mesh_abs = _resolve_mesh_file(scene_dir, scene_xml, body_name)
            if mesh_abs is None:
                print(f"  {scene}/{body_name}: primitive (no mesh), skipping")
                continue
            if not os.path.exists(mesh_abs):
                print(f"  {scene}/{body_name}: mesh not found at {mesh_abs}")
                continue
            rel = os.path.relpath(mesh_abs, _REPO_ROOT)
            out[mesh_abs] = rel
    return out


def _run_graspgen(mesh_path: str, out_yaml: str, num_grasps: int) -> bool:
    env = os.environ.copy()
    env["CUDA_HOME"] = _CUDA_HOME
    env["PATH"] = f"{_CUDA_HOME}/bin:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = f"{_CUDA_HOME}/lib64:" + env.get("LD_LIBRARY_PATH", "")
    # Strip conda so the GraspGen venv runs cleanly.
    for k in ("CONDA_DEFAULT_ENV", "CONDA_PREFIX", "CONDA_PROMPT_MODIFIER"):
        env.pop(k, None)

    cmd = [
        _GRASPGEN_PY, _GRASPGEN_SCRIPT,
        "--mesh_file", mesh_path, "--mesh_scale", "1.0",
        "--gripper_config", _GRASPGEN_CFG,
        "--num_grasps", str(num_grasps),
        "--output_file", out_yaml,
        "--no-visualization",
    ]
    print(f"  $ {' '.join(cmd)}")
    r = subprocess.run(cmd, env=env, cwd=_REPO_ROOT)
    return r.returncode == 0 and os.path.exists(out_yaml)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", action="append",
                   help="Limit to one or more scenes (default: all scenes with tasks.yaml)")
    p.add_argument("--mesh", action="append",
                   help="Compute for specific mesh file(s), bypassing scene inventory")
    p.add_argument("--num_grasps", type=int, default=200)
    p.add_argument("--force", action="store_true", help="Regenerate even if cached")
    args = p.parse_args()

    for path in (_GRASPGEN_PY, _GRASPGEN_SCRIPT, _GRASPGEN_CFG):
        if not os.path.exists(path):
            sys.exit(f"Missing GraspGen prerequisite: {path}\n"
                     f"Run scripts/install_graspgen.sh and scripts/download_graspgen_models.sh first.")

    os.makedirs(_CACHE_DIR, exist_ok=True)

    # Build (mesh_abs, rel_path) inventory
    if args.mesh:
        inventory = {}
        for m in args.mesh:
            m_abs = os.path.abspath(m)
            inventory[m_abs] = os.path.relpath(m_abs, _REPO_ROOT)
    else:
        scenes = args.scene or [
            os.path.basename(os.path.dirname(p))
            for p in sorted(glob.glob(os.path.join(_REPO_ROOT, "scenes/*/tasks.yaml")))
        ]
        print(f"Scanning scenes: {scenes}")
        inventory = _inventory(scenes)

    if not inventory:
        print("No mesh-backed pick targets found.")
        return

    # Load or init the index
    if os.path.exists(_INDEX_PATH):
        with open(_INDEX_PATH) as f:
            index = json.load(f)
    else:
        index = {}

    # Dedup by SHA
    sha_to_mesh: dict[str, str] = {}
    for mesh_abs, rel in inventory.items():
        sha = _sha256_file(mesh_abs)
        index[rel] = sha
        sha_to_mesh.setdefault(sha, mesh_abs)

    print(f"\n{len(inventory)} pick target(s) → {len(sha_to_mesh)} unique mesh(es) to process")
    ok, skipped, failed = 0, 0, 0
    for sha, mesh_abs in sha_to_mesh.items():
        out_yaml = os.path.join(_CACHE_DIR, f"{sha}.yml")
        name = os.path.basename(mesh_abs)
        if os.path.exists(out_yaml) and not args.force:
            print(f"\n[{sha[:10]}] {name}: cached, skipping")
            skipped += 1
            continue
        print(f"\n[{sha[:10]}] {name}: running GraspGen ...")
        if _run_graspgen(mesh_abs, out_yaml, args.num_grasps):
            ok += 1
        else:
            failed += 1
            print(f"    FAILED for {name}")

    with open(_INDEX_PATH, "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    print(f"\nDone. {ok} generated, {skipped} cached, {failed} failed.")
    print(f"Index: {_INDEX_PATH}")


if __name__ == "__main__":
    main()
