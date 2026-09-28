"""Public experimental workflow: inspect, train, and review one configured scene."""
import argparse
import json
from datetime import datetime
from pathlib import Path
import numpy as np
from scene_config import ROOT, load_config
from sfm_data import read_scene, scene_bounds, image_geometry, scaled_intrinsics


def inspect(cfg):
    paths, poses, K, indices, points = read_scene(cfg["data"])
    lo, hi = scene_bounds(cfg["scene"], points)
    first, wh = image_geometry(paths[0], cfg["data"])
    h, w = first.shape[:2]
    for path in paths[1:]:
        image, _ = image_geometry(path, cfg["data"])
        if image.shape[:2] != (h, w):
            raise ValueError(f"Inconsistent image dimensions: {path}")
    print(f"Registered cameras: {len(paths)}; finite sparse points: {len(points)}")
    print(f"Original images: {w}x{h}; training images: {wh[0]}x{wh[1]}")
    print(f"Training RGB memory: {len(paths) * wh[0] * wh[1] * 12 / 1024**3:.2f} GiB (plus sampling maps)")
    print("Camera IDs:", indices.tolist())
    print("Training K:\n", scaled_intrinsics(K, (w, h), wh, cfg["data"]["sfm_max_side"]))
    print("Scene bounds:", json.dumps({"box_min": lo.tolist(), "box_max": hi.tolist()}))
    if len(points):
        inside = np.all((points >= lo) & (points <= hi), axis=1).mean()
        print(f"Sparse points inside box: {inside:.1%}")
    print("Camera position range:", poses[:, :3, 3].min(axis=0), poses[:, :3, 3].max(axis=0))
    print("Images and camera arrays are readable. Inspect geometry visually in SfM before training.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inspect", "train", "review", "snapshot"])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--steps", type=int, help="Override the number of additional training steps")
    parser.add_argument("--model-id", help="Override output/review model ID")
    parser.add_argument("--resume", metavar="MODEL_ID", help="Continue weights and optimizer state from this ID")
    parser.add_argument("--no-web", action="store_true", help="Disable the browser preview during training")
    parser.add_argument("--output", type=Path, help="Snapshot PNG path (must not already exist)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.steps is not None:
        if args.steps <= 0:
            parser.error("--steps must be positive")
        cfg["training"]["steps"] = args.steps
    if args.model_id:
        if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in args.model_id):
            parser.error("Invalid --model-id")
        cfg["training"]["model_id"] = args.model_id
    if args.no_web:
        cfg["preview"]["enabled"] = False
    if args.command == "inspect":
        return inspect(cfg)
    if args.command in ("review", "snapshot"):
        output = None
        if args.command == "snapshot":
            output = args.output or Path(cfg["training"]["directory"]) / (
                cfg["training"]["model_id"] + "_snapshot_" + datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".png")
            if output.exists():
                parser.error(f"Snapshot already exists: {output}")
        import render_nerf
        return render_nerf.main(config=cfg, output=output)
    from run_ffnerf_OG import train
    t = cfg["training"]
    directory = Path(t["directory"])
    if not args.resume and list(directory.glob(f"{t['model_id']}_weights_*.npz")):
        parser.error("Checkpoints already exist for this model ID. Use --resume or choose a new --model-id.")
    paths, poses, K, indices, points = read_scene(cfg["data"])
    lo, hi = scene_bounds(cfg["scene"], points)
    cfg["scene"].update(box_min=lo.tolist(), box_max=hi.tolist())
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    manifest = directory / f"{t['model_id']}_config_{stamp}.json"
    manifest.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"Resolved run config: {manifest}")
    train(kernel_file=ROOT / "ffnerf_OG.cl", npz_path=Path(cfg["data"]["reconstruction"]),
          demo_path=None, rt_file=None, training_dir=directory, model_id=t["model_id"],
          load_id=args.resume, steps=t["steps"], period=t["period"],
          samples_per_ray=t["samples_per_ray"], batch_rays=t["batch_rays"],
          lr=t["lr"], dt=0.2, rng_seed=t["seed"], lmbda_base=t["lmd"], config=cfg)


if __name__ == "__main__":
    main()
