#!/bin/bash
# Download/clone external MuJoCo assets for FailBench scenes.
# Run from the external_assets/ directory.
#
# Usage:
#   cd external_assets
#   bash setup.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== FailBench External Assets Setup ==="

# --- vikashplus/object_sim ---
if [ ! -d "vikashplus_object_sim" ]; then
    echo "Cloning vikashplus/object_sim..."
    git clone --depth 1 https://github.com/vikashplus/object_sim.git vikashplus_object_sim
else
    echo "vikashplus_object_sim already exists, skipping."
fi

# --- vikashplus/furniture_sim ---
if [ ! -d "vikashplus_furniture_sim" ]; then
    echo "Cloning vikashplus/furniture_sim..."
    git clone --depth 1 https://github.com/vikashplus/furniture_sim.git vikashplus_furniture_sim
else
    echo "vikashplus_furniture_sim already exists, skipping."
fi

# --- vikashplus/YCB_sim ---
if [ ! -d "vikashplus_YCB_sim" ]; then
    echo "Cloning vikashplus/YCB_sim..."
    git clone --depth 1 https://github.com/vikashplus/YCB_sim.git vikashplus_YCB_sim
else
    echo "vikashplus_YCB_sim already exists, skipping."
fi

# --- kevinzakka/mujoco_scanned_objects ---
# Large repo (1030 objects). Shallow clone; specific objects selected at scene-build time.
if [ ! -d "kevinzakka_mujoco_scanned_objects" ]; then
    echo "Cloning kevinzakka/mujoco_scanned_objects (shallow)..."
    git clone --depth 1 https://github.com/kevinzakka/mujoco_scanned_objects.git kevinzakka_mujoco_scanned_objects
else
    echo "kevinzakka_mujoco_scanned_objects already exists, skipping."
fi

echo ""
echo "=== Done. Assets ready in: $SCRIPT_DIR ==="
echo "Repos downloaded:"
ls -d */ 2>/dev/null || echo "(none)"
