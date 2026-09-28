"""Read the custom SfM Stage 2 and BA archives without an OpenCL dependency."""
from pathlib import Path, PureWindowsPath
import cv2
import numpy as np


def read_scene(data):
    path = Path(data["reconstruction"])
    with np.load(path, allow_pickle=True) as archive:
        d = {k: archive[k] for k in archive.files}
    paths = d["img_paths"]
    n = len(paths)
    keep = np.ones(n, dtype=bool)
    for key in ("posed", "keep_mask"):
        if key in d:
            mask = np.asarray(d[key], dtype=bool)
            if mask.shape != (n,):
                raise ValueError(f"{key} must align with img_paths")
            keep &= mask
    if not keep.any():
        raise ValueError("No registered cameras in reconstruction")
    if "c2w" in d:
        poses = d["c2w"]
    else:
        poses = d["RT"]
        direction = str(d["rt_dir"].reshape(-1)[0]).lower()
        if direction == "w2c":
            poses = np.linalg.inv(poses)
        elif direction != "c2w":
            raise ValueError(f"Unknown rt_dir: {direction}")
    if poses.shape != (n, 4, 4):
        raise ValueError("Camera transforms must align with img_paths and have shape (N,4,4)")
    poses = np.asarray(poses[keep], dtype=np.float32)
    if not np.isfinite(poses).all():
        raise ValueError("Non-finite camera transforms")
    K = np.asarray(d["K"], dtype=np.float32)
    if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Expected finite shared 3x3 intrinsics with positive focal lengths")
    resolved = []
    for stored in paths[keep]:
        if data.get("images_dir"):
            p = Path(data["images_dir"]) / PureWindowsPath(str(stored)).name
        else:
            p = Path(str(stored))
            if not p.is_absolute():
                p = path.parent / p
        if not p.is_file():
            raise FileNotFoundError(f"Image missing: {p}. Set data.images_dir to relocate images.")
        resolved.append(p)
    points = np.asarray(d.get("X", d.get("X_track", np.empty((0, 3)))), dtype=np.float32)
    if "X" not in d and "has_X" in d:
        points = points[np.asarray(d["has_X"], dtype=bool)]
    points = points[np.isfinite(points).all(axis=1)]
    indices = np.asarray(d.get("active_cam_global", np.arange(n)), dtype=np.int32)[keep]
    return resolved, poses, K, indices, points


def scene_bounds(scene, points):
    lo, hi = scene["box_min"], scene["box_max"]
    if lo is None and hi is None:
        if not len(points):
            raise ValueError("No finite sparse points: set scene.box_min and scene.box_max")
        q_lo, q_hi = np.percentile(points, [1, 99], axis=0)
        center = 0.5 * (q_lo + q_hi)
        # Use the sphere circumscribing the percentile AABB, then enlarge its
        # radius by 3/2. The resulting scene AABB is a cube, which leaves
        # useful space around the sparse reconstruction without following
        # individual outliers.
        radius = max(0.5 * float(np.linalg.norm(q_hi - q_lo)), 0.01) * 1.5
        lo, hi = center - radius, center + radius
    elif lo is None or hi is None:
        raise ValueError("Set both scene.box_min and scene.box_max, or leave both null")
    lo, hi = np.asarray(lo, dtype=np.float32), np.asarray(hi, dtype=np.float32)
    if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all() or not (hi > lo).all():
        raise ValueError("Scene bounds must be finite xyz vectors with box_max > box_min")
    return lo, hi


def image_geometry(path, data):
    im = cv2.imread(str(path))
    if im is None:
        raise ValueError(f"Cannot decode image: {path}")
    h, w = im.shape[:2]
    max_side = data["max_side"]
    scale = min(1.0, max_side / max(h, w)) if max_side > 0 else 1.0
    return im, (max(1, int(w * scale)), max(1, int(h * scale)))


def scaled_intrinsics(K, original_wh, target_wh, sfm_max_side):
    w, h = original_wh
    s = min(1.0, sfm_max_side / max(w, h)) if sfm_max_side > 0 else 1.0
    # Match Stage 1's integer resize dimensions.
    sw, sh = max(1, int(w * s)), max(1, int(h * s))
    out = K.copy()
    out[0] *= target_wh[0] / sw
    out[1] *= target_wh[1] / sh
    return out


def load_images(data, paths, K):
    first, (w, h) = image_geometry(paths[0], data)
    h0, w0 = first.shape[:2]
    images = np.empty((len(paths), h, w, 3), dtype=np.float32)
    for i, path in enumerate(paths):
        im = first if i == 0 else cv2.imread(str(path))
        if im is None or im.shape[:2] != (h0, w0):
            raise ValueError(f"Unreadable image or inconsistent image dimensions: {path}")
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
        images[i] = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    return images, scaled_intrinsics(K, (w0, h0), (w, h), data["sfm_max_side"]), (w, h)
