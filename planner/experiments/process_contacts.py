"""Process captured contact data: project to image, annotate, and export summaries."""

import csv
import json
import os
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import mujoco

from planner.experiments.data_capture import ContactProjector
from planner.experiments.manager import load_sample_npz


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------


def process_npz(npz_path: str, scene_xml_path: str,
                camera_name: str = None,
                camera_lookat=None, camera_distance=None,
                camera_azimuth=None, camera_elevation=None,
                width: int = 640, height: int = 480) -> dict:
    """Load an .npz experiment file and process all contact data.

    Returns a dict with:
        "contacts"   – list of per-contact dicts
        "summary"    – per-object aggregated summary dicts
        "pre_rgb"    – (H, W, 3) uint8 image
        "pixels"     – (N, 2) pixel coords
        "depths"     – (N,) depths
        "in_frame"   – (N,) bool mask
    """
    data = load_sample_npz(npz_path)
    model = mujoco.MjModel.from_xml_path(scene_xml_path)
    projector = ContactProjector(model, camera_name=camera_name,
                                 camera_lookat=camera_lookat,
                                 camera_distance=camera_distance,
                                 camera_azimuth=camera_azimuth,
                                 camera_elevation=camera_elevation,
                                 width=width, height=height)

    positions = data["contact_positions"]       # (N, 3)
    forces = data["contact_forces"]             # (N, 6)
    geom_pairs = data["contact_geom_pairs"]     # (N, 2)
    failure_ids = data["contact_failure_id"]     # (N,)
    failure_modes = data["failure_modes"]        # (M,) string array
    pre_rgb = data["pre_rgb"]                   # (H, W, 3)

    n_contacts = len(positions)
    if n_contacts == 0:
        return {
            "contacts": [],
            "summary": {},
            "pre_rgb": pre_rgb,
            "pixels": np.zeros((0, 2)),
            "depths": np.zeros(0),
            "in_frame": np.zeros(0, dtype=bool),
        }

    # Project all contact positions to pixel coordinates
    pixels, depths = projector.project(positions)
    mask = projector.in_frame(pixels, depths)

    # Force magnitudes (L2 norm of linear force components)
    force_magnitudes = np.linalg.norm(forces[:, :3], axis=1)

    # Build per-contact records
    contacts: List[dict] = []
    # Accumulator for per-object summary
    obj_accum: Dict[str, dict] = {}

    for i in range(n_contacts):
        g1, g2 = int(geom_pairs[i, 0]), int(geom_pairs[i, 1])
        name_a = projector.geom_name(g1)
        name_b = projector.geom_name(g2)
        fmode = str(failure_modes[int(failure_ids[i])])
        mag = float(force_magnitudes[i])

        record = {
            "object_a": name_a,
            "object_b": name_b,
            "x": float(positions[i, 0]),
            "y": float(positions[i, 1]),
            "z": float(positions[i, 2]),
            "force_N": round(mag, 4),
            "force_full": forces[i].tolist(),
            "pixel_u": float(pixels[i, 0]),
            "pixel_v": float(pixels[i, 1]),
            "in_frame": bool(mask[i]),
            "failure_mode": fmode,
        }
        contacts.append(record)

        # Aggregate by contact pair (both objects involved)
        pair_key = tuple(sorted([name_a, name_b]))
        pair_label = f"{pair_key[0]} vs {pair_key[1]}"
        if pair_label not in obj_accum:
            obj_accum[pair_label] = {
                "object_a": pair_key[0],
                "object_b": pair_key[1],
                "contact_count": 0,
                "peak_force_N": 0.0,
                "total_force_N": 0.0,
                "positions": [],
            }
        acc = obj_accum[pair_label]
        acc["contact_count"] += 1
        acc["peak_force_N"] = max(acc["peak_force_N"], mag)
        acc["total_force_N"] += mag
        acc["positions"].append(positions[i])

    # Finalize per-object summaries
    summary: Dict[str, dict] = {}
    for key, acc in obj_accum.items():
        pos_array = np.array(acc["positions"])
        mean_pos = pos_array.mean(axis=0)
        summary[key] = {
            "object_a": acc["object_a"],
            "object_b": acc["object_b"],
            "contact_count": acc["contact_count"],
            "peak_force_N": round(acc["peak_force_N"], 4),
            "mean_force_N": round(acc["total_force_N"] / acc["contact_count"], 4),
            "mean_position": {"x": round(float(mean_pos[0]), 4),
                              "y": round(float(mean_pos[1]), 4),
                              "z": round(float(mean_pos[2]), 4)},
        }

    return {
        "contacts": contacts,
        "summary": summary,
        "pre_rgb": pre_rgb,
        "pixels": pixels,
        "depths": depths,
        "in_frame": mask,
        "force_magnitudes": force_magnitudes,
    }


# ---------------------------------------------------------------------------
# Annotated image
# ---------------------------------------------------------------------------


def render_annotated_image(result: dict, output_path: str,
                           max_points: int = 5000,
                           figsize=(10, 7.5)) -> str:
    """Overlay projected contact points on the pre-failure RGB image.

    Points are colored by force magnitude. Object labels are placed at
    the mean contact position for each impacted object.

    Returns the output file path.
    """
    pre_rgb = result["pre_rgb"]
    pixels = result["pixels"]
    mask = result["in_frame"]
    forces = result["force_magnitudes"]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(pre_rgb)

    if mask.any():
        vis_px = pixels[mask]
        vis_f = forces[mask]

        # Subsample if too many points
        if len(vis_px) > max_points:
            idx = np.random.default_rng(0).choice(len(vis_px), max_points, replace=False)
            vis_px = vis_px[idx]
            vis_f = vis_f[idx]

        # Normalize forces for colormap
        f_min, f_max = vis_f.min(), vis_f.max()
        if f_max > f_min:
            f_norm = (vis_f - f_min) / (f_max - f_min)
        else:
            f_norm = np.zeros_like(vis_f)

        sc = ax.scatter(vis_px[:, 0], vis_px[:, 1], c=f_norm,
                        cmap="hot", s=4, alpha=0.6, edgecolors="none")
        plt.colorbar(sc, ax=ax, label="Force magnitude (normalized)", shrink=0.7)

    # Label impacted contact pairs at their mean pixel position
    summary = result["summary"]
    for pair_name, info in summary.items():
        # Average pixel positions for contacts in this pair
        obj_pixels = []
        for i, c in enumerate(result["contacts"]):
            pair = tuple(sorted([c["object_a"], c["object_b"]]))
            pair_label = f"{pair[0]} vs {pair[1]}"
            if pair_label == pair_name and result["in_frame"][i]:
                obj_pixels.append(result["pixels"][i])
        if obj_pixels:
            mean_px = np.mean(obj_pixels, axis=0)
            label = f"{pair_name}\n{info['peak_force_N']:.1f} N"
            ax.annotate(label, xy=mean_px, fontsize=5, color="cyan",
                        ha="center", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6))

    ax.set_title("Contact Points Projected onto Pre-Failure Image")
    ax.set_xlim(0, pre_rgb.shape[1])
    ax.set_ylim(pre_rgb.shape[0], 0)
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Export: CSV and JSON
# ---------------------------------------------------------------------------


def export_csv(contacts: List[dict], output_path: str) -> str:
    """Write per-contact records to CSV."""
    if not contacts:
        return output_path
    columns = ["object_a", "object_b", "x", "y", "z", "force_N",
               "pixel_u", "pixel_v", "in_frame", "failure_mode"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(contacts)
    return output_path


def export_json(result: dict, output_path: str) -> str:
    """Write contacts + per-object summary to JSON."""
    out = {
        "per_object_summary": result["summary"],
        "num_contacts": len(result["contacts"]),
        "num_in_frame": int(result["in_frame"].sum()) if len(result["in_frame"]) else 0,
    }
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    return output_path


def print_summary(result: dict) -> None:
    """Print human-readable contact summary to stdout."""
    summary = result["summary"]
    if not summary:
        print("No contacts detected.")
        return
    print(f"\n{'='*60}")
    print(f"Contact Summary — {len(result['contacts']):,} total contacts")
    print(f"{'='*60}")
    for pair_name, info in sorted(summary.items(), key=lambda x: -x[1]["peak_force_N"]):
        mp = info["mean_position"]
        print(f"  {info['object_a']} vs {info['object_b']}")
        print(f"    at ({mp['x']:.3f}, {mp['y']:.3f}, {mp['z']:.3f})")
        print(f"    peak force: {info['peak_force_N']:.2f} N, "
              f"mean: {info['mean_force_N']:.2f} N, "
              f"contacts: {info['contact_count']}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Process FailBench contact data")
    parser.add_argument("npz_file", help="Path to .npz experiment file")
    parser.add_argument("--scene-xml", required=True, help="Path to scene XML")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    parser.add_argument("--camera", default="front_cam")
    parser.add_argument("--no-image", action="store_true", help="Skip annotated image")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.npz_file))[0]

    result = process_npz(args.npz_file, args.scene_xml, camera_name=args.camera)
    print_summary(result)

    csv_path = os.path.join(args.output_dir, f"{base}_contacts.csv")
    export_csv(result["contacts"], csv_path)
    print(f"CSV written to {csv_path}")

    json_path = os.path.join(args.output_dir, f"{base}_summary.json")
    export_json(result, json_path)
    print(f"JSON written to {json_path}")

    if not args.no_image:
        img_path = os.path.join(args.output_dir, f"{base}_annotated.png")
        render_annotated_image(result, img_path)
        print(f"Annotated image written to {img_path}")


if __name__ == "__main__":
    main()
