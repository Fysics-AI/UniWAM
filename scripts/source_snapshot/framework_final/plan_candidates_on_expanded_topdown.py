#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import habitat_sim


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan observer candidates on expanded topdown maps.")
    parser.add_argument("--expanded-sample-dir", type=Path, required=True)
    parser.add_argument("--annotation-json", type=Path, required=True)
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--sector-radius-m", type=float, default=3.0)
    parser.add_argument("--sector-half-angle-deg", type=float, default=40.0)
    parser.add_argument("--observer-min-distance-m", type=float, default=0.5)
    parser.add_argument("--observer-max-distance-m", type=float, default=4.0)
    parser.add_argument("--preferred-distance-m", type=float, default=2.0)
    parser.add_argument("--preferred-angles-deg", default="0,20,40")
    parser.add_argument("--candidate-min-separation-m", type=float, default=0.5)
    parser.add_argument("--candidate-min-angle-diff-deg", type=float, default=15.0)
    parser.add_argument("--candidate-min-distance-diff-m", type=float, default=0.5)
    parser.add_argument("--seed-angle-tolerance-deg", type=float, default=12.0)
    parser.add_argument("--candidate-count", type=int, default=3)
    parser.add_argument("--selection-mode", choices=["random", "farthest", "diverse", "preferred"], default="preferred")
    parser.add_argument("--nav-height-slack-m", type=float, default=0.2)
    parser.add_argument("--nav-sample-band-m", type=float, default=0.3)
    parser.add_argument("--sample-pixel-step", type=int, default=2)
    parser.add_argument("--operating-search-max-distance-m", type=float, default=2.0)
    parser.add_argument("--operating-search-step-m", type=float, default=0.01)
    parser.add_argument("--operating-min-distance-m", type=float, default=0.2)
    parser.add_argument("--operating-max-distance-m", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def denorm_999_to_pixel(coord: list[int] | tuple[int, int], width: int, height: int) -> tuple[int, int]:
    x = int(round(float(coord[0]) * float(width) / 999.0))
    y = int(round(float(coord[1]) * float(height) / 999.0))
    return int(np.clip(x, 0, max(0, width - 1))), int(np.clip(y, 0, max(0, height - 1)))


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32)


def parse_float_list(spec: str) -> list[float]:
    return [float(v.strip()) for v in str(spec).split(',') if str(v).strip()]


def footprint_extent_stats(bounds_xz: list[float] | tuple[float, float, float, float] | None) -> dict:
    if bounds_xz is None or len(bounds_xz) != 4:
        return {
            "width_m": 0.0,
            "depth_m": 0.0,
            "diag_m": 0.0,
            "max_dim_m": 0.0,
        }
    x0, z0, x1, z1 = [float(v) for v in bounds_xz]
    width_m = abs(float(x1) - float(x0))
    depth_m = abs(float(z1) - float(z0))
    diag_m = float(math.hypot(width_m, depth_m))
    return {
        "width_m": float(width_m),
        "depth_m": float(depth_m),
        "diag_m": float(diag_m),
        "max_dim_m": float(max(width_m, depth_m)),
    }


def image_xy_to_world_xz(pixel_x: float, pixel_y: float, p2w: dict) -> tuple[float, float]:
    u = float(p2w["u_min"]) + (float(pixel_x) + 0.5) * (float(p2w["u_max"]) - float(p2w["u_min"])) / float(p2w["out_width"])
    f = float(p2w["f_front"]) - (float(pixel_y) + 0.5) * (float(p2w["f_front"]) - float(p2w["f_back"])) / float(p2w["out_height"])
    wx = float(p2w["stand_x"]) + float(p2w["right_x"]) * u + float(p2w["forward_x"]) * f
    wz = float(p2w["stand_z"]) + float(p2w["right_z"]) * u + float(p2w["forward_z"]) * f
    return float(wx), float(wz)


def world_xz_to_image_xy(world_x: float, world_z: float, p2w: dict) -> tuple[int, int]:
    dx = float(world_x) - float(p2w["stand_x"])
    dz = float(world_z) - float(p2w["stand_z"])
    u = dx * float(p2w["right_x"]) + dz * float(p2w["right_z"])
    f = dx * float(p2w["forward_x"]) + dz * float(p2w["forward_z"])
    px = (u - float(p2w["u_min"])) * float(p2w["out_width"]) / max(1e-6, float(p2w["u_max"]) - float(p2w["u_min"])) - 0.5
    py = (float(p2w["f_front"]) - f) * float(p2w["out_height"]) / max(1e-6, float(p2w["f_front"]) - float(p2w["f_back"])) - 0.5
    return int(round(px)), int(round(py))


def build_nav_cache_union(pathfinder: habitat_sim.PathFinder, nav_y: float, sample_band_m: float, height_slack_m: float) -> dict:
    lower_b, upper_b = pathfinder.get_bounds()
    lower = np.asarray(lower_b, dtype=np.float32)
    upper = np.asarray(upper_b, dtype=np.float32)
    sample_heights = np.arange(float(nav_y) - float(sample_band_m), float(nav_y) + float(sample_band_m) + 0.05, 0.1)
    nav_bool = None
    for sample_y in sample_heights:
        nav_grid = np.asarray(pathfinder.get_topdown_view(0.02, float(sample_y), float(height_slack_m)), dtype=np.uint8)
        layer = nav_grid > 0
        nav_bool = layer if nav_bool is None else (nav_bool | layer)
    return {
        "nav_y": float(nav_y),
        "meters_per_pixel": 0.02,
        "lower": lower,
        "upper": upper,
        "nav_bool": np.asarray(nav_bool, dtype=bool),
    }


def build_local_navmask(nav_cache: dict, p2w: dict) -> np.ndarray:
    nav_bool = np.asarray(nav_cache["nav_bool"], dtype=bool)
    lower = np.asarray(nav_cache["lower"], dtype=np.float32)
    mpp = float(nav_cache["meters_per_pixel"])
    h, w = nav_bool.shape[:2]
    xs = np.arange(int(p2w["out_width"]), dtype=np.float32)
    ys = np.arange(int(p2w["out_height"]), dtype=np.float32)
    u_vals = float(p2w["u_min"]) + (xs + 0.5) * (float(p2w["u_max"]) - float(p2w["u_min"])) / float(p2w["out_width"])
    f_vals = float(p2w["f_front"]) - (ys + 0.5) * (float(p2w["f_front"]) - float(p2w["f_back"])) / float(p2w["out_height"])
    world_x = float(p2w["stand_x"]) + float(p2w["right_x"]) * u_vals.reshape(1, -1) + float(p2w["forward_x"]) * f_vals.reshape(-1, 1)
    world_z = float(p2w["stand_z"]) + float(p2w["right_z"]) * u_vals.reshape(1, -1) + float(p2w["forward_z"]) * f_vals.reshape(-1, 1)
    cols = np.clip(np.round((world_x - float(lower[0])) / max(mpp, 1e-6) - 0.5).astype(np.int32), 0, max(0, w - 1))
    rows = np.clip(np.round((world_z - float(lower[2])) / max(mpp, 1e-6) - 0.5).astype(np.int32), 0, max(0, h - 1))
    return np.asarray(nav_bool[rows, cols], dtype=bool)


def line_ok_with_endpoint_tail(observer_world_xz: np.ndarray, endpoint_world_xz: np.ndarray, p2w: dict, navmask: np.ndarray, step_m: float = 0.02) -> bool:
    p0 = np.asarray(endpoint_world_xz, dtype=np.float32)
    p1 = np.asarray(observer_world_xz, dtype=np.float32)
    delta = p1 - p0
    dist = float(np.linalg.norm(delta))
    if dist < 1e-6:
        return False
    direction = delta / dist
    seen_nav = False
    traveled = 0.0
    while traveled <= dist + 1e-6:
        p = p0 + direction * float(traveled)
        px, py = world_xz_to_image_xy(float(p[0]), float(p[1]), p2w)
        in_nav = 0 <= px < navmask.shape[1] and 0 <= py < navmask.shape[0] and bool(navmask[py, px])
        if in_nav:
            seen_nav = True
        elif seen_nav:
            return False
        traveled += float(step_m)
    return seen_nav


def find_operating_point_from_front_line(
    *,
    edge_center_world_xz: np.ndarray,
    outward_normal_world_xz: np.ndarray,
    p2w: dict,
    navmask: np.ndarray,
    max_distance_m: float,
    step_m: float,
    min_distance_m: float,
    target_max_distance_m: float,
) -> dict | None:
    center = np.asarray(edge_center_world_xz, dtype=np.float32)
    normal = normalize(np.asarray(outward_normal_world_xz, dtype=np.float32))
    if float(np.linalg.norm(normal)) < 1e-6:
        return None
    dist = 0.0
    while dist <= float(max_distance_m) + 1e-6:
        p = center + normal * float(dist)
        px, py = world_xz_to_image_xy(float(p[0]), float(p[1]), p2w)
        if 0 <= px < navmask.shape[1] and 0 <= py < navmask.shape[0] and bool(navmask[py, px]):
            if float(dist) < float(min_distance_m):
                dist += float(step_m)
                continue
            if float(dist) > float(target_max_distance_m):
                return None
            return {
                "operating_world_xz": [float(p[0]), float(p[1])],
                "operating_topdown_xy": [int(px), int(py)],
                "distance_from_edge_m": float(dist),
            }
        dist += float(step_m)
    return None


def main() -> None:
    args = build_parser().parse_args()
    expanded_dir = args.expanded_sample_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    expanded_meta = json.loads((expanded_dir / "meta.json").read_text(encoding="utf-8"))
    source_meta = json.loads(Path(expanded_meta["source_meta_json"]).read_text(encoding="utf-8"))
    ann = json.loads(args.annotation_json.expanduser().resolve().read_text(encoding="utf-8"))
    target = ann["targets"][int(args.target_index)]

    img_bgr = cv2.imread(str(expanded_dir / "local_topdown_rgb.png"))
    if img_bgr is None:
        raise FileNotFoundError(expanded_dir / "local_topdown_rgb.png")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    p2w_exp = expanded_meta["topdown"]["pixel_to_world_xz"]
    p2w_src = source_meta["topdown"]["pixel_to_world_xz"]
    target_center_xy = expanded_meta["targets"][int(args.target_index)]["center_xy"]
    target_center_world = np.asarray(image_xy_to_world_xz(target_center_xy[0], target_center_xy[1], p2w_exp), dtype=np.float32)
    target_source_row = source_meta["targets"][int(args.target_index)]
    footprint_stats = footprint_extent_stats(
        target_source_row.get("target_world_bounds_xz")
        or target_source_row.get("target_object_footprint_xz")
    )

    edge_line = target["operating_edge_line"]
    src_p1 = denorm_999_to_pixel(edge_line[0], int(p2w_src["out_width"]), int(p2w_src["out_height"]))
    src_p2 = denorm_999_to_pixel(edge_line[1], int(p2w_src["out_width"]), int(p2w_src["out_height"]))
    edge_p1_world = np.asarray(image_xy_to_world_xz(src_p1[0], src_p1[1], p2w_src), dtype=np.float32)
    edge_p2_world = np.asarray(image_xy_to_world_xz(src_p2[0], src_p2[1], p2w_src), dtype=np.float32)
    edge_center_world = 0.5 * (edge_p1_world + edge_p2_world)
    tangent = normalize(edge_p2_world - edge_p1_world)
    normal = normalize(np.asarray([-tangent[1], tangent[0]], dtype=np.float32))
    if float(np.dot(normal, target_center_world - edge_center_world)) > 0:
        normal = -normal

    traj = json.loads(Path(source_meta["trajectory_json"]).read_text(encoding="utf-8"))
    scene_source = str(traj.get("scene_path", "")).strip() or str(traj["scene_id"])
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_dataset_config_file = str(traj["dataset_config"])
    backend.scene_id = str(scene_source)
    backend.create_renderer = False
    backend.enable_physics = False
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [habitat_sim.agent.AgentConfiguration()]))
    try:
        nav_cache = build_nav_cache_union(
            pathfinder=sim.pathfinder,
            nav_y=float(source_meta["nav_y"]),
            sample_band_m=float(args.nav_sample_band_m),
            height_slack_m=float(args.nav_height_slack_m),
        )
    finally:
        sim.close()

    navmask = build_local_navmask(nav_cache, p2w_exp)
    overlay = img.copy()
    mask = navmask
    overlay[mask] = overlay[mask] * 0.78 + np.array([255, 200, 90], dtype=np.float32) * 0.22
    overlay = np.clip(np.round(overlay), 0, 255).astype(np.uint8)

    edge_p1_xy = world_xz_to_image_xy(float(edge_p1_world[0]), float(edge_p1_world[1]), p2w_exp)
    edge_p2_xy = world_xz_to_image_xy(float(edge_p2_world[0]), float(edge_p2_world[1]), p2w_exp)
    center_xy = world_xz_to_image_xy(float(target_center_world[0]), float(target_center_world[1]), p2w_exp)
    cv2.line(overlay, edge_p1_xy, edge_p2_xy, (0, 255, 255), 3, cv2.LINE_AA)

    operating = find_operating_point_from_front_line(
        edge_center_world_xz=edge_center_world,
        outward_normal_world_xz=normal,
        p2w=p2w_exp,
        navmask=navmask,
        max_distance_m=float(args.operating_search_max_distance_m),
        step_m=float(args.operating_search_step_m),
        min_distance_m=float(args.operating_min_distance_m),
        target_max_distance_m=float(args.operating_max_distance_m),
    )
    if operating is None:
        raise RuntimeError("failed to find operating point from front line toward navigable boundary")
    operating_world = np.asarray(operating["operating_world_xz"], dtype=np.float32)
    operating_xy = tuple(int(v) for v in operating["operating_topdown_xy"])
    cv2.drawMarker(overlay, operating_xy, (0, 255, 0), cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)

    preferred_distance_effective_m = float(args.preferred_distance_m)

    samples = []
    candidates_ok = []
    rng = random.Random(int(args.seed))
    for py in range(0, img.shape[0], int(args.sample_pixel_step)):
        for px in range(0, img.shape[1], int(args.sample_pixel_step)):
            if not bool(navmask[py, px]):
                continue
            wx, wz = image_xy_to_world_xz(px, py, p2w_exp)
            world = np.asarray([float(wx), float(wz)], dtype=np.float32)
            vec_operating = world - operating_world
            dist_operating = float(np.linalg.norm(vec_operating))
            if dist_operating < float(args.observer_min_distance_m) or dist_operating > float(args.observer_max_distance_m):
                continue
            direction = normalize(vec_operating)
            cosang = float(np.clip(np.dot(direction, normal), -1.0, 1.0))
            ang = float(math.degrees(math.acos(cosang)))
            if ang > float(args.sector_half_angle_deg):
                continue
            vec_target = world - target_center_world
            dist_target = float(np.linalg.norm(vec_target))
            samples.append((px, py))
            if line_ok_with_endpoint_tail(world, edge_p1_world, p2w_exp, navmask) and line_ok_with_endpoint_tail(world, edge_p2_world, p2w_exp, navmask):
                candidates_ok.append(
                    {
                        "observer_topdown_xy": [int(px), int(py)],
                        "observer_world_xz": [float(world[0]), float(world[1])],
                        "distance_m": float(dist_target),
                        "operating_distance_m": float(dist_operating),
                        "target_center_distance_m": float(dist_target),
                        "angle_deg": float(ang),
                    }
                )

    preferred_angles = [abs(float(v)) for v in parse_float_list(args.preferred_angles_deg)]

    def seed_rank(row: dict) -> tuple[float, float, float, float]:
        op_dist = float(row.get("operating_distance_m", row["distance_m"]))
        ang = abs(float(row["angle_deg"]))
        angle_cost = min(abs(ang - pref) for pref in preferred_angles) if preferred_angles else ang
        shortfall = max(0.0, float(preferred_distance_effective_m) - op_dist)
        overflow = max(0.0, op_dist - float(preferred_distance_effective_m))
        return (shortfall * 4.0 + overflow, float(angle_cost), shortfall, -float(op_dist))

    candidates_ok = sorted(candidates_ok, key=seed_rank)
    all_points_json = out_dir / 'observer_points_all.json'
    all_points_json.write_text(json.dumps(candidates_ok, ensure_ascii=False, indent=2), encoding='utf-8')

    selected = []
    seed = candidates_ok[0] if candidates_ok else None
    if seed is not None:
        selected.append(seed)
        seed_angle = float(seed['angle_deg'])
        seed_dist = float(seed.get('operating_distance_m', seed['distance_m']))
        base_dist = max(float(preferred_distance_effective_m), float(seed_dist))
        desired_dists = [base_dist, min(float(args.observer_max_distance_m), base_dist + 0.75), min(float(args.observer_max_distance_m), base_dist + 1.5)]
        remaining = [r for r in candidates_ok[1:] if abs(float(r['angle_deg']) - seed_angle) <= float(args.seed_angle_tolerance_deg)]
        for desired in desired_dists[1:]:
            if len(selected) >= int(args.candidate_count):
                break
            ranked = sorted(
                remaining,
                key=lambda r: (
                    abs(float(r.get('operating_distance_m', r['distance_m'])) - float(desired)),
                    abs(float(r['angle_deg']) - seed_angle),
                    -float(r.get('operating_distance_m', r['distance_m'])),
                ),
            )
            pick = None
            for row in ranked:
                w = np.asarray(row['observer_world_xz'], dtype=np.float32)
                if any(float(np.linalg.norm(w - np.asarray(kept['observer_world_xz'], dtype=np.float32))) < float(args.candidate_min_separation_m) for kept in selected):
                    continue
                pick = row
                break
            if pick is not None:
                selected.append(pick)
                remaining = [r for r in remaining if r is not pick]
        if len(selected) < int(args.candidate_count):
            for row in candidates_ok[1:]:
                if row in selected:
                    continue
                w = np.asarray(row['observer_world_xz'], dtype=np.float32)
                if any(float(np.linalg.norm(w - np.asarray(kept['observer_world_xz'], dtype=np.float32))) < float(args.candidate_min_separation_m) for kept in selected):
                    continue
                selected.append(row)
                if len(selected) >= int(args.candidate_count):
                    break

    for px, py in samples:
        cv2.circle(overlay, (int(px), int(py)), 1, (120, 120, 255), -1, cv2.LINE_AA)
    for row in selected:
        px, py = [int(v) for v in row["observer_topdown_xy"]]
        cv2.circle(overlay, (px, py), 8, (255, 80, 80), 2, cv2.LINE_AA)
        cv2.arrowedLine(overlay, (px, py), operating_xy, (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.05)

    out_img = out_dir / "topdown_observer_candidates.png"
    cv2.imwrite(str(out_img), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    result = {
        "expanded_sample_dir": str(expanded_dir),
        "annotation_json": str(args.annotation_json.expanduser().resolve()),
        "target_index": int(args.target_index),
        "annotation_target": target,
        "observer_candidate_count": len(candidates_ok),
        "observer_selected_count": len(selected),
        "distance_anchor": "operating_point",
        "preferred_distance_effective_m": float(preferred_distance_effective_m),
        "target_footprint": footprint_stats,
        "operating_point_topdown_xy": [int(operating_xy[0]), int(operating_xy[1])],
        "operating_point_world_xz": [float(operating_world[0]), float(operating_world[1])],
        "operating_distance_from_edge_m": float(operating["distance_from_edge_m"]),
        "observer_seed_point": seed,
        "observer_points": selected,
        "artifacts": {
            "topdown_observer_candidates": str(out_img),
            "observer_points_all_json": str(all_points_json),
        },
    }
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
