
import ast
import os
import re
import time
import queue
import threading
import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import math
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import numpy as np
import cv2
from tqdm import tqdm
import signal

import nerf_web_remote as nwr
from render_nerf import build_c2w_rt_yaw_pitch

import socket
import struct

import ffnerf_tools as ft
import pyopencl as cl


_last_sigint_time = time.time() - 2.0
STOP_REQUESTED = False
def _handle_sigint(sig, frame):
    global STOP_REQUESTED, _last_sigint_time
    now = time.time()
    if STOP_REQUESTED and (now - _last_sigint_time) < 2.0:
        print("\n[CTRL+C] Forced exit.")
        os._exit(130)
    _last_sigint_time = now
    STOP_REQUESTED = True
    print("\n[CTRL+C] Graceful shutdown requested…")




###############################################################
#   Training parameters
###############################################################
LMBDA_BASE= 1e-4    # 1e-4
LR_FLOOR  = 5e-6

T_MID     = 1500 #1000 #2730     # maybe (1024 * 64) / (batch_rays * samples_per_ray)
#    or rather to 256^3 / (1024*log2(64)) * (batch_rays * log2(samples_per_ray)) -> 2 730 * (batch_rays * log2(samples_per_ray))
#   1024, 64 -> 2730  |  1024,128 -> 2340  |  2048, 32 -> 1638..  |  2048, 64 -> 1365..
sh_multiplier_function = lambda step: np.clip((step-(2*T_MID))/(T_MID),0.0, 1.0)
occ_u_multiplier_function = lambda step: (step>T_MID) #0 # min(1.0,max(0.0,(step-0.5*T_MID)/(T_MID))) * OCC_U_STR #
# (
#     (-10.0 < (- (8.0 / (T_MID / 2)) * (step - T_MID)) and (- (8.0 / (T_MID / 2)) * (step - T_MID)) < 10.0) *
#         1.0 / (1.0 + np.exp((- (8.0 / (T_MID / 2)) * (step - T_MID)))) + \
#     float(step >= T_MID)
# )
color_MLP_rel_lr_function = lambda step: 0.15+0.85*min(1.0,max(0.0,(step-T_MID)/(T_MID)))

BP_ema=0.05
BP_fg_frac=0.5
BP_bg_cap=16
BP_fg_mask_fn=None
BP_fg_tau=8
BP_fg_dilate=0

W_prev, H_prev = 360, 640
down = 5 # for early on down=8 is good, later down=6 gives better detail in ok time
samples_per_ray_prev = 64# 32




# ----------------------------
# Dataset loading
# ----------------------------
def invert_w2c_to_c2w(T):
    """T: (4,4) world-to-camera -> camera-to-world"""
    R = T[:3,:3]
    t = T[:3,3]
    Rcw = R.T
    tcw = -R.T @ t
    out = np.eye(4, dtype=T.dtype)
    out[:3,:3] = Rcw
    out[:3,3]  = tcw
    return out

def apply_world_transform(c2w_all, T_world, X=None):
    """
    c2w_all: (N,4,4)
    T_world: (4,4) mapping old-world -> new-world
    X: (M,3) optional points
    """
    c2w_new = (T_world[None, :, :] @ c2w_all).astype(c2w_all.dtype)

    X_new = None
    if X is not None and X.size > 0:
        Xh = np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=1)
        Xn = (T_world @ Xh.T).T
        X_new = Xn[:, :3]
    return c2w_new, X_new

def rotation_from_to(a, b, eps=1e-8):
    a = a / (np.linalg.norm(a) + eps)
    b = b / (np.linalg.norm(b) + eps)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))

    if s < eps:
        # a parallel to b
        if c > 0:
            return np.eye(3, dtype=np.float32)
        # 180-degree: choose any orthogonal axis
        axis = np.array([1,0,0], dtype=np.float32)
        if abs(a[0]) > 0.9:
            axis = np.array([0,1,0], dtype=np.float32)
        v = np.cross(a, axis)
        v = v / (np.linalg.norm(v) + eps)
        K = np.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]], dtype=np.float32)
        return (np.eye(3, dtype=np.float32) + 2 * (K @ K))

    v = v / (s + eps)
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]], dtype=np.float32)
    R = np.eye(3, dtype=np.float32) + s * K + (1 - c) * (K @ K)
    return R


def parse_rt_file(rt_path: Path) -> dict[int, np.ndarray]:
    rt_map = {}
    with rt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            idx = int(parts[0])
            vals = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            if vals.size != 16:
                raise ValueError(f"RT line for idx={idx} has {vals.size} floats (expected 16).")
            rt_map[idx] = vals.reshape(4, 4)
    return rt_map

def parse_image_index_from_name(p: Path) -> int | None:
    s = p.stem
    m = re.search(r"pyramid_no\.(\d+)", s)
    if m:
        return int(m.group(1))
    ints = re.findall(r"(\d+)", s)
    if ints:
        return int(ints[-1])
    return None

import numpy as np

def _resize_scale_max_side(h: int, w: int, max_side: int) -> float:
    """Scale s used by resize_max_side() given an original (h,w)."""
    if max_side <= 0:
        return 1.0
    m = max(h, w)
    if m <= max_side:
        return 1.0
    return float(max_side) / float(m)

def K_for_original_from_resized(K_resized: np.ndarray, h_orig: int, w_orig: int, max_side: int = 1600) -> np.ndarray:
    """
    If K_resized corresponds to an image that was created from original by resize_max_side(max_side),
    return K_orig that matches the original pixel coordinates.
    """
    s = _resize_scale_max_side(h_orig, w_orig, max_side)
    if s == 1.0:
        return K_resized.astype(np.float32).copy()
    S_inv = np.array([[1.0/s, 0.0,   0.0],
                      [0.0,   1.0/s, 0.0],
                      [0.0,   0.0,   1.0]], dtype=np.float32)
    return (S_inv @ K_resized.astype(np.float32))


def _load_images_sorted(img_paths):
    img_paths = list(img_paths)
    if not img_paths:
        raise FileNotFoundError("No images found.")
    first = cv2.imread(str(img_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Failed to read image: {img_paths[0]}")
    first_rgb = cv2.cvtColor(first, cv2.COLOR_BGR2RGB)
    H, W, C = first_rgb.shape
    if C != 3:
        raise RuntimeError("Expected 3-channel images.")

    Ims = np.empty((len(img_paths), H, W, 3), dtype=np.float32)
    for i, p in enumerate(img_paths):
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read image: {p}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (H, W):
            raise ValueError(f"Image size mismatch: {p} has {rgb.shape[:2]}, expected {(H, W)}")
        Ims[i] = rgb.astype(np.float32) / 255.0
    return Ims, W, H

def load_dataset_fused(
    data_dir: Path | None = None,
    rt_file: Path | None = None,
    npz_path: Path | None = None,
):
    """
    Either:
      - npz_path provided (new format or old 'c2w' format)
    or:
      - data_dir (+ optional rt_file) provided for txt+images demo format

    Returns: Ims, c2w_all, K, img_indices, (W,H)
    """
    if npz_path is not None:
        npz_path = Path(npz_path)
        data = np.load(npz_path, allow_pickle=True)
        keys = set(data.keys())

        img_paths = [Path(p) for p in data["img_paths"]]
        Ims, W, H = _load_images_sorted(img_paths)  # assumes img_paths are aligned as stored

        K = data["K"].astype(np.float32) if "K" in keys else None
        MAX_SIDE = 1600  # match your reconstruction preprocessing

        if K is not None:
            # K in NPZ is for resized images (max_side=1600). Convert to original-image K.
            H0, W0 = Ims.shape[1], Ims.shape[2] # Ims is (N,H,W,3) -> sizes are the same by construction
            K = K_for_original_from_resized(K, h_orig=H0, w_orig=W0, max_side=MAX_SIDE)

        if "c2w" in keys:
            c2w_all = data["c2w"].astype(np.float32)
            # If file contains posed mask and still stores all, you may want to filter.
            posed = data.get("posed", np.ones((c2w_all.shape[0],), dtype=bool)).astype(bool)
            # Align with img_paths: your NPZ loader earlier returned posed-only tuples.
            # Here we keep it simple: if posed exists and lengths match, filter both.
            if posed.shape[0] == c2w_all.shape[0] and len(img_paths) == c2w_all.shape[0]:
                keep = np.where(posed)[0]
                Ims = Ims[keep]
                c2w_all = c2w_all[keep]
                img_paths = [img_paths[i] for i in keep]
        elif "RT" in keys and "rt_dir" in keys:
            RT = data["RT"].astype(np.float32)

            rt_dir = data["rt_dir"]
            if isinstance(rt_dir, np.ndarray):
                rt_dir = str(rt_dir.reshape(-1)[0])
            else:
                rt_dir = str(rt_dir)

            if rt_dir.lower() == "c2w":
                c2w_all = RT
            elif rt_dir.lower() == "w2c":
                c2w_all = np.stack([invert_w2c_to_c2w(RT[i]) for i in range(RT.shape[0])], axis=0).astype(np.float32)
            else:
                raise ValueError(f"Unknown rt_dir={rt_dir!r} (expected 'c2w' or 'w2c')")
        else:
            raise KeyError(f"Unrecognized NPZ format. Keys={sorted(keys)}")

        # indices: keep whatever is available; else 0..N-1
        if "active_cam_global" in keys:
            img_indices = data["active_cam_global"].astype(np.int32)
            # if we filtered posed above, also filter indices:
            if img_indices.shape[0] != Ims.shape[0] and "posed" in keys:
                posed = data["posed"].astype(bool)
                keep = np.where(posed)[0]
                img_indices = img_indices[keep] if img_indices.shape[0] == posed.shape[0] else np.arange(Ims.shape[0], dtype=np.int32)
        else:
            img_indices = np.arange(Ims.shape[0], dtype=np.int32)

        return Ims, c2w_all, K, img_indices, (W, H)

    # --- demo txt + folder images
    if data_dir is None:
        raise ValueError("Provide either npz_path or data_dir.")
    data_dir = Path(data_dir)
    if rt_file is None:
        rt_file = data_dir / "pyramids_RT.txt"
    rt_file = Path(rt_file)

    rt_map = parse_rt_file(rt_file)

    img_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        img_paths.extend(sorted(data_dir.glob(ext)))
    if not img_paths:
        raise FileNotFoundError(f"No images found in {data_dir.resolve()} (png/jpg/jpeg).")

    pairs = []
    for p in img_paths:
        idx = parse_image_index_from_name(p)
        if idx is None:
            continue
        if idx not in rt_map:
            continue
        pairs.append((idx, p))
    if not pairs:
        raise RuntimeError("Found images but couldn't match any to RT indices.")

    pairs.sort(key=lambda x: x[0])
    ordered_paths = [p for (_, p) in pairs]
    Ims, W, H = _load_images_sorted(ordered_paths)

    c2w_all = np.stack([rt_map[idx] for (idx, _) in pairs], axis=0).astype(np.float32)
    img_indices = np.array([idx for (idx, _) in pairs], dtype=np.int32)

    K = None  # demo format has no intrinsics unless you add it
    return Ims, c2w_all, K, img_indices, (W, H)


# ----------------------------
# Save measurement helpers
# ----------------------------
def ensure_training_dir(training_dir: Path) -> Path:
    training_dir = Path(training_dir)
    training_dir.mkdir(parents=True, exist_ok=True)
    return training_dir


def open_loss_log(training_dir: Path, model_id: str):
    training_dir = ensure_training_dir(training_dir)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    ms = int((time.time() * 1000) % 1000)
    path = training_dir / f"{model_id}_loss_{ts}-{ms:03d}.csv"
    f = open(path, "a", buffering=1, encoding="utf-8")
    f.write("wall_s,step,lr_f,mse,rmse_rgb,l2mean,inflight,batch_q\n")  # <-- lr_f added
    return path, f


def write_loss_row(f, wall_s: float, step: int, lr_f: int, mse: float, rmse: float, l2mean: float, inflight, batch_q: int):
    infl_s = "|".join(str(x) for x in inflight)
    f.write(f"{wall_s:.6f},{step},{lr_f:.3e},{mse:.9e},{rmse:.9e},{l2mean:.9e},{infl_s},{batch_q}\n")

# ----------------------------
# Graph representation helpers
@dataclass(frozen=True)
class TensorSpec:
    name: str
    dim: int
    role: str  # "act" or "delta"

@dataclass(frozen=True)
class ParamSpec:
    name: str
    out: str       # delta tensor name (next)
    inp: str       # activation tensor name (prev)
    out_dim: int
    in_dim: int
    bias: Optional[str] = None

def _prod(shape: Tuple[int, ...]) -> int:
    p = 1
    for s in shape: p *= int(s)
    return p





def derive_nerf_weights(layers: dict, edges: dict):
    """
    Returns: weights: OrderedDict {W: (out_dim, in_dim)}
    """

    # --- derive weights from edges ---
    weights = OrderedDict()
    for node_out, (W, node_in) in edges.items():
        if node_out not in layers:
            raise KeyError(f"edge out node '{node_out}' missing from layers")
        if node_in not in layers:
            raise KeyError(f"edge in node '{node_in}' missing from layers")
        out_dim = int(layers[node_out])
        in_dim  = int(layers[node_in])
        weights[W] = (out_dim, in_dim)
        
    needed_acts_nodes = []
    for _, (_, node_in) in edges.items():
        # allow nodes like dir0 that you don't want stored:
        # you can exclude by not using them as weight-edge inputs.
        needed_acts_nodes.append(node_in)

    # preserve insertion order but unique
    seen = set()
    needed_acts_nodes = [n for n in needed_acts_nodes if not (n in seen or seen.add(n))]
    activations = OrderedDict((f"a:{n}", int(layers[n])) for n in needed_acts_nodes)
    needed_delta_nodes = list(edges.keys())  # insertion order of edges dict matters (py3.7+)
    deltas = OrderedDict((f"d:{n}", int(layers[n])) for n in needed_delta_nodes)

    return weights, activations, deltas

# ----------------------------
# Producer / batch queue
# ----------------------------
def make_fg_mask_from_rgb(Ims, tau=8, dilate=0):
    """
    Ims: (N,H,W,3) uint8 in [0..255] OR float in [0..1]
    tau: threshold in uint8 space if uint8; in [0..255] also for float (converted)
    dilate: optional radius (0..2 typical), cheap square dilation
    returns: list of (H,W) bool masks
    """
    assert Ims.ndim == 4 and Ims.shape[-1] >= 3
    N, H, W, _ = Ims.shape

    # Normalize threshold depending on dtype/range
    if np.issubdtype(Ims.dtype, np.floating):
        thr = float(tau) / 255.0
        mx = Ims[..., :3].max(axis=-1)
        fg0 = (mx > thr)
    else:
        mx = Ims[..., :3].max(axis=-1)  # uint8
        fg0 = (mx > np.uint8(tau))

    masks = [fg0[i].astype(bool, copy=False) for i in range(N)]

    if dilate > 0:
        # simple binary dilation without scipy: max over shifted windows
        r = int(dilate)
        out = []
        for m in masks:
            acc = m.copy()
            for dy in range(-r, r + 1):
                ys0 = max(0, -dy); ys1 = min(H, H - dy)
                yd0 = max(0,  dy); yd1 = min(H, H + dy)
                for dx in range(-r, r + 1):
                    xs0 = max(0, -dx); xs1 = min(W, W - dx)
                    xd0 = max(0,  dx); xd1 = min(W, W + dx)
                    acc[yd0:yd1, xd0:xd1] |= m[ys0:ys1, xs0:xs1]
            out.append(acc)
        masks = out

    return masks


def precompute_fg_indices(fg_masks):
    """returns list of 1D int32 arrays of flat indices into H*W"""
    return [np.flatnonzero(m.ravel()).astype(np.int32, copy=False) for m in fg_masks]

def __make_fg_mask_from_alpha(Ims_rgba, alpha_tau=1e-3): # current data does not use this
    a = Ims_rgba[..., 3]
    return [ (a[i] > alpha_tau) for i in range(Ims_rgba.shape[0]) ]


@dataclass
class Batch:
    im_idx: np.ndarray   # (B,) int32
    idx: np.ndarray      # (B,) int32
    idy: np.ndarray      # (B,) int32
    rays: np.ndarray
    targets: np.ndarray

def _update_weights_u8(weights_u8, fg_masks, im_idx, idx, idy, err, ema=0.05, bg_cap=16):
    """
    weights_u8: list[np.ndarray] each (H, W) uint8
    fg_masks: list[ndarray(H,W) bool] OR None
    im_idx/idx/idy: (B,) int arrays, pixel coords (x=idx, y=idy)
    err: (B,) float, bigger => sample more
    ema: EMA rate (0..1)
    bg_cap: max target weight for background pixels (1..255)
    """
    # Robust normalization: map err -> [1..255] with a soft knee.
    # Use median for stability.
    med = float(np.median(err)) if err.size else 0.0
    scale = (med + 1e-6) * 2.0  # knee
    score = err / scale
    score = np.clip(score, 0.0, 8.0)
    target_w = (1.0 + 254.0 * (score / 8.0)).astype(np.float32)  # 1..255

    use_mask = fg_masks is not None

    for k in range(im_idx.size):
        im = int(im_idx[k])
        x = int(idx[k])
        y = int(idy[k])

        w_t = float(target_w[k])
        if use_mask and (not bool(fg_masks[im][y, x])):
            if bg_cap < 255:
                w_t = min(w_t, float(bg_cap))

        w_old = float(weights_u8[im][y, x])
        w_new = (1.0 - ema) * w_old + ema * w_t
        weights_u8[im][y, x] = np.uint8(np.clip(int(w_new + 0.5), 1, 255))


def _sample_pixels_weighted_rejection(rng, wmap_u8, B, uniform_frac=0.5):
    """
    wmap_u8: (H, W) uint8 weights in [1..255]
    returns idx, idy arrays of length B (x, y)
    Weighted part uses rejection: propose uniform pixel, accept with prob w/255.
    """
    H, W = wmap_u8.shape
    idx = np.empty((B,), dtype=np.int32)
    idy = np.empty((B,), dtype=np.int32)

    n_uniform = int(B * uniform_frac)
    # Uniform samples
    idx[:n_uniform] = rng.integers(0, W, size=(n_uniform,), dtype=np.int32)
    idy[:n_uniform] = rng.integers(0, H, size=(n_uniform,), dtype=np.int32)

    # Weighted rejection samples
    for i in range(n_uniform, B):
        while True:
            x = int(rng.integers(0, W, dtype=np.int32))
            y = int(rng.integers(0, H, dtype=np.int32))
            # accept with prob w/255
            if rng.integers(0, 256, dtype=np.int32) <= int(wmap_u8[y, x]):
                idx[i] = x
                idy[i] = y
                break

    return idx, idy

def _sample_pixels_fg_weighted_rejection(rng, wmap_u8, fg_indices, H, W, n, max_tries=10_000):
    """
    Weighted rejection, but proposals are restricted to FG pixels via fg_indices.
    """
    x = np.empty((n,), dtype=np.int32)
    y = np.empty((n,), dtype=np.int32)
    if fg_indices.size == 0:
        # fallback: uniform if no FG available
        x[:] = rng.integers(0, W, size=n, dtype=np.int32)
        y[:] = rng.integers(0, H, size=n, dtype=np.int32)
        return x, y

    # Note: if fg_indices is huge, this is still fine; each trial is O(1).
    for i in range(n):
        tries = 0
        while True:
            pick = int(fg_indices[rng.integers(0, fg_indices.size)])
            yy = pick // W
            xx = pick - yy * W
            if rng.integers(0, 256, dtype=np.int32) <= int(wmap_u8[yy, xx]):
                x[i] = xx
                y[i] = yy
                break
            tries += 1
            if tries >= max_tries:
                # safety fallback: just take the proposed FG pixel
                x[i] = xx
                y[i] = yy
                break
    return x, y

def _sample_pixels_mixed_fg_bg(rng, wmap_u8, fg_indices, H, W, B, fg_frac=0.8, bg_uniform=True):
    """
    Returns idx, idy (x,y).
    - FG part: weighted rejection, restricted to fg_indices
    - BG part: uniform (recommended) OR weighted (if you want)
    """
    B_fg = int(B * fg_frac)
    B_bg = B - B_fg

    # Background samples
    if B_bg > 0:
        if bg_uniform:
            x_bg = rng.integers(0, W, size=B_bg, dtype=np.int32)
            y_bg = rng.integers(0, H, size=B_bg, dtype=np.int32)
        else:
            # if you really want BG to be weighted too, call your existing sampler
            x_bg, y_bg = _sample_pixels_weighted_rejection(rng, wmap_u8, B_bg, uniform_frac=0.0)
    else:
        x_bg = np.empty((0,), np.int32)
        y_bg = np.empty((0,), np.int32)

    # Foreground samples
    if B_fg > 0:
        x_fg, y_fg = _sample_pixels_fg_weighted_rejection(rng, wmap_u8, fg_indices, H, W, B_fg)
    else:
        x_fg = np.empty((0,), np.int32)
        y_fg = np.empty((0,), np.int32)

    # Concatenate
    idx = np.concatenate([x_fg, x_bg], axis=0)
    idy = np.concatenate([y_fg, y_bg], axis=0)
    return idx, idy

def batch_producer(
    stop_evt, out_q, feedback_q, rng_seed,
    Ims, RTs,K, W, H,
    batch_rays,
    ema=0.05,
    fg_frac=0.8, bg_cap=16,
    fg_mask_fn=None,    # callable(Ims)->list[H,W]bool ; if None, uses RGB threshold
    fg_tau=8, fg_dilate=0,
):
    rng = np.random.default_rng(rng_seed)
    num_images = Ims.shape[0]

    # per-image weight maps (H, W), uint8 in [1..255]
    weights_u8 = [np.full((H, W), 16, dtype=np.uint8) for _ in range(num_images)]

    # --- Foreground definition (pluggable) ---
    if fg_mask_fn is None:
        fg_masks = make_fg_mask_from_rgb(Ims, tau=fg_tau, dilate=fg_dilate)
    else:
        fg_masks = fg_mask_fn(Ims)

    fg_indices = precompute_fg_indices(fg_masks)

    fb_im = []
    fb_x = []
    fb_y = []
    fb_e = []

    def drain_feedback(max_items=2048):
        nonlocal fb_im, fb_x, fb_y, fb_e
        n = 0
        while n < max_items:
            try:
                im_idx_f, idx_f, idy_f, err_f = feedback_q.get_nowait()
            except queue.Empty:
                break
            fb_im.append(im_idx_f)
            fb_x.append(idx_f)
            fb_y.append(idy_f)
            fb_e.append(err_f)
            n += 1

        if fb_im:
            im_all = np.concatenate(fb_im).astype(np.int32, copy=False)
            x_all  = np.concatenate(fb_x).astype(np.int32, copy=False)
            y_all  = np.concatenate(fb_y).astype(np.int32, copy=False)
            e_all  = np.concatenate(fb_e).astype(np.float32, copy=False)

            _update_weights_u8(
                weights_u8,
                fg_masks=fg_masks,
                im_idx=im_all, idx=x_all, idy=y_all, err=e_all,
                ema=ema,
                bg_cap=bg_cap,
            )

            fb_im.clear(); fb_x.clear(); fb_y.clear(); fb_e.clear()

    while not stop_evt.is_set():
        drain_feedback()

        if out_q.full():
            time.sleep(0.001)
            continue

        # Choose images uniformly for now
        im_idx = rng.integers(0, num_images, size=(batch_rays,), dtype=np.int32)

        idx = np.empty((batch_rays,), dtype=np.int32)
        idy = np.empty((batch_rays,), dtype=np.int32)

        # group by image, sample with FG/BG quota per image group
        for im in range(num_images):
            sel = np.where(im_idx == im)[0]
            if sel.size == 0:
                continue

            # If you want to keep some uniform mixing inside FG as well,
            # tuneable fg_frac; BG is uniform by default in sampler.
            # x_s, y_s = _sample_pixels_mixed_fg_bg(
            #     rng,
            #     wmap_u8=weights_u8[im],
            #     fg_indices=fg_indices[im],
            #     H=H, W=W,
            #     B=sel.size,
            #     fg_frac=fg_frac,
            #     bg_uniform=False,
            # )
            x_s, y_s = _sample_pixels_weighted_rejection(rng,weights_u8[im],sel.size,uniform_frac=0.2)

            idx[sel] = x_s
            idy[sel] = y_s

        # rays, targets = ft.rays_targets_from_indices(im_idx, idx, idy, Ims, RTs, W, H)
        rays, targets = ft.rays_targets_from_indices(im_idx, idx, idy, Ims, RTs, W, H, K=K)

        try:
            out_q.put(Batch(im_idx=im_idx, idx=idx, idy=idy, rays=rays, targets=targets), timeout=0.1)
        except queue.Full:
            time.sleep(0.001)
            continue



# ----------------------------
# Prepared IO triple-buffering
# ----------------------------
@dataclass
class PreparedBatch:
    io: ft.RayBatch
    batch: Batch
    ev_upload: cl.Event            # upload completion (targets upload depends on rays upload)
    ev_done: cl.Event | None = None  # filled by trainer: last event of this training step
    step_id: int = 0

def _ev_complete(ev: cl.Event) -> bool:
    # pyopencl exposes command_execution_status on Event in most builds
    return ev.command_execution_status == cl.command_execution_status.COMPLETE

def _prune_completed_inflight(inflight: list[PreparedBatch], free_io_q: "queue.Queue[ft.RayBatch]") -> list[PreparedBatch]:
    """
    Returns remaining inflight list (not complete yet), and returns completed IOs back to free_io_q.
    NOTE: Do NOT put back IOs that you still want to keep for periodic loss/feedback; handle that outside.
    """
    keep: list[PreparedBatch] = []
    for pb in inflight:
        if pb.ev_done is not None and _ev_complete(pb.ev_done):
            try:
                free_io_q.put_nowait(pb.io)
            except queue.Full:
                # shouldn't happen if sizes match, but don't crash
                pass
        else:
            keep.append(pb)
    return keep


def io_preparer_thread(
    stop_evt: threading.Event,
    batch_q: "queue.Queue[Batch]",
    free_io_q: "queue.Queue[ft.RayBatch]",
    ready_q: "queue.Queue[PreparedBatch]",
):
    """
    Pulls host Batches from batch_q, grabs a free RayBatch, uploads rays+targets,
    and pushes PreparedBatch into ready_q.

    Uses io.queue for copies; safe for in-order and out-of-order queues.
    Trainer must add wait_for=[pb.ev_upload] on OOO queues.
    """
    while not stop_evt.is_set():
        try:
            batch = batch_q.get(timeout=0.1)
        except queue.Empty:
            continue

        # Get a free IO buffer
        try:
            io = free_io_q.get(timeout=0.1)
        except queue.Empty:
            # trainer behind: retry soon
            time.sleep(0.0005)
            continue

        try:
            # Enqueue copies
            wait = [io.ev_last] if getattr(io, "ev_last", None) is not None else None
            ev_r = cl.enqueue_copy(io.queue, io.rays_buf, batch.rays, wait_for=wait)
            ev_t = cl.enqueue_copy(io.queue, io.targets_buf, batch.targets, wait_for=[ev_r])
            io.ev_last = ev_t

            pb = PreparedBatch(io=io, batch=batch, ev_upload=ev_t)

            # Push to ready queue
            try:
                ready_q.put(pb)
            except queue.Full:
                ev_t.wait()
                # return IO and drop batch (or block longer; I drop to avoid deadlock)
                try:
                    free_io_q.put_nowait(io)
                except queue.Full:
                    pass
                time.sleep(0.0005)

        except Exception:
            # If anything fails, return IO so we don't leak capacity
            try:
                free_io_q.put_nowait(io)
            except queue.Full:
                pass
            raise


# ============================================================
# Live preview IPC (training = server, viewer = client)
# ============================================================

def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        chunk = sock.recv(n - got)
        if not chunk:
            raise ConnectionError("socket closed")
        view[got:got + len(chunk)] = chunk
        got += len(chunk)
    return bytes(buf)

class PreviewIPC:
    """
    Training-side preview server:
    - accepts at most 1 client
    - receives RT updates in a background thread
    - training thread enqueues + downloads preview
    - sends frames to client
    """
    # Init pose: 'I' + 16f  (send once when a client connects)
    _I_FMT = "<c16f"
    _I_SIZE = struct.calcsize(_I_FMT)

    # RT msg: 'R' + 16f + moving(u8) + save(u8)
    _RT_FMT = "<c16fBB"
    _RT_SIZE = struct.calcsize(_RT_FMT)

    # Frame header: 'F' + W,H,step,nbytes (u32)
    _F_HDR_FMT = "<cIIII"
    _F_HDR_SIZE = struct.calcsize(_F_HDR_FMT)

    def __init__(self, host="127.0.0.1", port=5005):
        self.host = host
        self.port = int(port)

        self._srv = None
        self._client = None
        self._rx_thr = None
        self._stop = threading.Event()

        self._lock = threading.Lock()
        self._init_RT = None  # np.ndarray(4,4) float32 (c2w) to send on connect
        self.last_rt_time = 0.0
        self.last_moving = 0
        self.pending_RT = None          # np.ndarray(4,4) float32
        self.pending_request = False
        self.save_request = False

    def set_init_rt(self, RT: np.ndarray):
        """
        RT: (4,4) float32, camera-to-world. This will be sent once to the viewer on connect.
        """
        RT = np.asarray(RT, dtype=np.float32).reshape(4, 4)
        with self._lock:
            self._init_RT = RT.copy()

    def start(self):
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((self.host, self.port))
        self._srv.listen(1)
        self._srv.settimeout(0.2)

        self._rx_thr = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx_thr.start()
        print(f"[preview ipc] listening on {self.host}:{self.port}")

    def stop(self):
        self._stop.set()
        try:
            if self._client is not None:
                self._client.close()
        except Exception:
            pass
        try:
            if self._srv is not None:
                self._srv.close()
        except Exception:
            pass
        if self._rx_thr is not None:
            self._rx_thr.join(timeout=0.5)

    def _accept_once(self):
        if self._srv is None or self._client is not None:
            return
        try:
            c, addr = self._srv.accept()
        except socket.timeout:
            return
        c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._client = c
        print(f"[preview ipc] client connected from {addr}")
        
        # --- NEW: send init pose (if available) ---
        init_RT = None
        with self._lock:
            if self._init_RT is not None:
                init_RT = self._init_RT.copy()

        if init_RT is not None:
            payload = struct.pack(self._I_FMT, b"I", *init_RT.reshape(-1).tolist())
            try:
                self._client.sendall(payload)
                print("[preview ipc] sent init RT")
            except Exception as e:
                print(f"[preview ipc] init send failed: {repr(e)}")
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None

    def _rx_loop(self):
        while not self._stop.is_set():
            # accept client if needed
            self._accept_once()
            if self._client is None:
                continue
            try:
                msg = _recv_exact(self._client, self._RT_SIZE)
                typ, *rest = struct.unpack(self._RT_FMT, msg)
                if typ != b"R":
                    continue
                floats = rest[:16]
                moving = int(rest[16])
                save = int(rest[17])

                RT = np.array(floats, dtype=np.float32).reshape(4, 4)

                with self._lock:
                    self.pending_RT = RT
                    self.pending_request = True
                    self.last_rt_time = time.time()
                    self.last_moving = moving
                    if save:
                        self.save_request = True

            except Exception as e:
                print(f"[preview ipc] client disconnected: {repr(e)}")
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None

    def pop_pending_rt(self):
        with self._lock:
            if not self.pending_request or self.pending_RT is None:
                return None, 0, 0.0
            RT = self.pending_RT.copy()
            self.pending_request = False
            moving = int(self.last_moving)
            t = float(self.last_rt_time)
        return RT, moving, t

    def pop_save_request(self) -> bool:
        with self._lock:
            if not self.save_request:
                return False
            self.save_request = False
            return True

    def has_client(self) -> bool:
        return self._client is not None

    def send_frame(self, bgr_u8: np.ndarray, step: int):
        """
        bgr_u8: (H,W,3) uint8
        """
        if self._client is None:
            return
        H, W, C = bgr_u8.shape
        assert C == 3
        payload = bgr_u8.tobytes(order="C")
        hdr = struct.pack(self._F_HDR_FMT, b"F", W, H, int(step), len(payload))
        try:
            self._client.sendall(hdr + payload)
        except Exception as e:
            print(f"[preview ipc] send failed: {repr(e)}")
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None


# ----------------------------
# Training loop
# ----------------------------

def train(
    kernel_file: Path,
    npz_path: Path,
    demo_path: Path,
    rt_file: Path,
    training_dir: Path,
    model_id: str,
    load_id: str | None,
    steps: int,
    period: int,
    samples_per_ray: int,
    batch_rays: int,
    lr: float,
    dt: float,
    rng_seed: int,
    lmbda_base = LMBDA_BASE,
    config=None,
):
    global STOP_REQUESTED
    STOP_REQUESTED = False

    if samples_per_ray > ft.S_MAX:
        raise ValueError(f"samples_per_ray={samples_per_ray} must be <= S_MAX={ft.S_MAX}")
    if batch_rays <= 0:
        raise ValueError("batch_rays must be > 0")

    from scene_config import DEFAULTS
    from sfm_data import read_scene, load_images, scene_bounds
    preview = config["preview"] if config else DEFAULTS["preview"]
    W_prev, H_prev = preview["width"], preview["height"]
    down = preview["downscale"]
    samples_per_ray_prev = preview["samples_per_ray"]
    if config:
        paths, RTs, K_sfm, img_indices, points = read_scene(config["data"])
        Ims, K, (W, H) = load_images(config["data"], paths, K_sfm)
        box_min, box_max = scene_bounds(config["scene"], points)
    else:
        if npz_path is not None:
            raise ValueError("Use ffnerf_cli.py train --config for SfM scenes")
        Ims, RTs, K, img_indices, (W, H) = load_dataset_fused(
            npz_path=npz_path, data_dir=demo_path, rt_file=rt_file)
        box_min = np.array([-3, -3, -3], np.float32)
        box_max = np.array([3, 3, 3], np.float32)
    K_preview = None if K is None else K.copy()
    if K_preview is not None:
        K_preview[0] *= W_prev / W
        K_preview[1] *= H_prev / H
    print("Scene bounds:", box_min, box_max)
    view_ids = preview["views"]
    if any(type(i) is not int or i < 0 or i >= len(RTs) for i in view_ids):
        raise ValueError("preview.views must contain valid registered-camera indices")

    print(f"Loaded dataset: num_images={Ims.shape[0]}, W={W}, H={H}")
    print(f"RT file: {rt_file}")
    print(f"Image indices (first 10): {img_indices[:10].tolist()}")

    # Device / queue
    plat, dev = ft.pick_first_device()
    print("Platform:", plat.name, "|", plat.vendor, "|", plat.version)
    print("Device:  ", dev.name, "|", dev.vendor, "|", dev.version)
    print("OpenCL C:", getattr(dev, "opencl_c_version", "N/A"))

    ctx = cl.Context([dev])
    queue_cl, is_ooo = ft.make_queue(ctx, dev)
    print("Queue device:", queue_cl.device.name)
    print("Queue context devices:", [d.name for d in queue_cl.context.devices])
    print("Queue:   ", "OUT-OF-ORDER" if is_ooo else "IN-ORDER")

    # Build Nerf (or load)
    # IMPORTANT: layers/weights order MUST match kernel arg order expectations.
    occ = ft.OccParams(
        G_occ=32,
        box_min=box_min,
        box_max=box_max,
        rho_min=np.float32(1.0),
        rho_max=np.float32(16.0),
        dt_min=np.float32(np.max(box_max-box_min) / 8192),
        dt_max=np.float32(12.8/samples_per_ray / 6 * np.max(box_max-box_min)),
        t_near_clip=np.float32(config["scene"]["near"] if config else 0.3),
        # AABB intersection supplies the real far bound; keep this guard large so
        # cameras outside the scene box are not clipped before reaching it.
        t_far_clip=np.float32(1.0e9),
        occ_update_strength0=np.float32(ft.OCC_U_STR),
    ).finalize()
    layers = {
        "d0":  ft.L_POS,         # apos
        "d1":  ft.M,             # ad1
        "d2":  ft.D_OUT,         # (not stored separately; lives inside c0)
        "c0":  ft.D_OUT + ft.L_DIR, # ac_in
        "c1":  ft.M,             # ac1
        "c2":  ft.M,             # ac2
        "rgb": ft.C_OUT,         # output preact (delta stored)
        "dir0": ft.L_DIR,        # not-stored, inside c0
    }
    edges = {
        "d1":  ("Wd0", "d0"),
        "d2":  ("Wd1", "d1"),
        "c1":  ("Wc0", "c0"),
        "c2":  ("Wc1", "c1"),
        "rgb": ("Wc2", "c2"),
    }
    print(derive_nerf_weights(layers, edges))
    nerf = ft.Nerf.from_kernels(
        ctx = ctx, 
        queue = queue_cl, 
        kernel_file=kernel_file or ft.KERNEL_FILE_DEFAULT,
        decay_file=Path(ft.KERNEL_DECAY),
        weight_update_file=Path(ft.KERNEL_W_B_UPDATES),
        background="random" if (npz_path is not None) else "black",
        layers = layers,
        edges = edges, # format: node: (W_name, input_node)
        training_dir=training_dir,
        model_id=model_id,
        load_id=load_id,
        transfer_fmt=ft.TRANSFER_FMT,
        ffnerf_kernel_name=ft.KERNEL_NAME,
        decay_kernel_name="occ_nibble_decay_write_both_q24",
        wu_kernel_name="update_ffnerf_oneWG",
        build_options_extra=None,
        rng_seed=rng_seed,
        occ=occ,
        table_size_entries=ft.TABLE_SIZE_ENTRIES,
        f_per_level=ft.F_PER_LEVEL,
    )
    # make sure Nerf uses the same queue object as your pipeline
    nerf.ctx = ctx
    nerf.queue = queue_cl

    # -----------------------------
    # Producer thread + queues
    # -----------------------------
    IO_DEPTH = 3
    batch_q: queue.Queue[Batch] = queue.Queue(maxsize=IO_DEPTH*2)
    feedback_q: queue.Queue = queue.Queue(maxsize=IO_DEPTH*8)

    stop_evt = threading.Event()
    prod = threading.Thread(
        target=batch_producer,
        args=(stop_evt, batch_q, feedback_q, rng_seed, Ims, RTs, K, W, H, batch_rays),
        kwargs=dict(ema=BP_ema,fg_frac=BP_fg_frac,bg_cap=BP_bg_cap,
                    fg_mask_fn=BP_fg_mask_fn,fg_tau=BP_fg_tau, fg_dilate=BP_fg_dilate,),
        daemon=True,
    )
    prod.start()

    # -----------------------------
    # Triple-buffered RayBatch pool + IO-preparer
    # -----------------------------
    free_io_q: queue.Queue[ft.RayBatch] = queue.Queue(maxsize=IO_DEPTH)
    ready_io_q: queue.Queue[PreparedBatch] = queue.Queue(maxsize=IO_DEPTH)

    for _ in range(IO_DEPTH):
        io = ft.RayBatch.init(ctx, queue_cl, num_rays=batch_rays, samples_per_ray=samples_per_ray, with_host_arrays=False)
        free_io_q.put_nowait(io)

    io_stop_evt = threading.Event()
    io_thr = threading.Thread(
        target=io_preparer_thread,
        args=(io_stop_evt, batch_q, free_io_q, ready_io_q),
        daemon=True,
    )
    io_thr.start()

    inflight: list[PreparedBatch] = []
    out_host_period = np.empty((batch_rays, 4), dtype=np.float32)

    def _wait_until_slot_available():
        nonlocal inflight
        while len(inflight) >= IO_DEPTH:
            pb0 = inflight[0]
            if pb0.ev_done is not None:
                cl.wait_for_events([pb0.ev_done])
            inflight = _prune_completed_inflight(inflight, free_io_q)

    def _take_one_completed_for_metrics() -> PreparedBatch | None:
        nonlocal inflight
        if inflight:
            # Reserve the batch before its IO is recycled by the producer.
            pb = inflight.pop(-1)
            cl.wait_for_events([pb.ev_done])
            return pb
        return None

    # LR schedule bits from your original
    lmbda = lmbda_base * np.sqrt((batch_rays * samples_per_ray) / (1024 * 64))

    # -----------------------------
    # Preview jobs (using Nerf forward-only)
    # -----------------------------
    export_training_previews = True
    preview_jobs = []
    if export_training_previews: # TODO: modify so all of the rays and memory are already in the output shape, and have only one IO!!!
        W_ds, H_ds = W_prev // down, H_prev // down
        x_ds = np.tile(np.arange(W_ds, dtype=np.int32), H_ds)
        y_ds = np.repeat(np.arange(H_ds, dtype=np.int32), W_ds)
        idx_flat = np.clip(x_ds * down + (down // 2), 0, W_prev - 1)
        idy_flat = np.clip(y_ds * down + (down // 2), 0, H_prev - 1)

        RT_LIST = RTs[view_ids]

        for view_id, RT in enumerate(RT_LIST):
            rays_preview = ft.rays_from_rt_indices(RT, idx_flat, idy_flat, W_prev, H_prev, K=K_preview)
            n = rays_preview.shape[0]

            io_p = ft.RayBatch.init(ctx, queue_cl, num_rays=n, samples_per_ray=samples_per_ray_prev, with_host_arrays=True)
            io_p.upload_rays(rays_preview, wait=True)
            io_p.upload_targets(np.zeros((n, 4), np.float32), wait=True)

            preview_jobs.append(dict(
                view_id=view_id,
                io=io_p,
                W_ds=W_ds, H_ds=H_ds,
                samples_per_ray=samples_per_ray_prev,
            ))

    # -----------------------------
    # Web remote setup (optional)
    # -----------------------------
    enable_web_remote = preview["enabled"]
    web_remote = None
    io_web = None
    ev_web = None
    last_web_enqueue_t = 0.0

    # policy
    min_interval_moving = 0.2  # >=5 fps while moving
    interval_still = 0.5       # 2 fps while still
    still_grace = 0.15

    if enable_web_remote:
        # bind to 0.0.0.0 so phone can connect over Wi-Fi
        web_remote = nwr.WebRemoteServer(host=preview["host"], port=preview["port"])

        # --- INIT FROM DATASET (first c2w / RT) ---
        if RTs is not None and getattr(RTs, "shape", None) is not None and RTs.shape[0] > 0:
            web_remote.set_initial_from_c2w(RTs[0])
        else:
            print("[web remote] warning: RTs empty; using default init cam")

        web_remote.start()

        W_cam, H_cam = W_prev, H_prev
        W_ds, H_ds, idx_flat, idy_flat = nwr.make_preview_indices(W_cam, H_cam, down)

        n_preview = idx_flat.size
        io_web = ft.RayBatch.init(ctx, queue_cl, num_rays=n_preview, samples_per_ray=samples_per_ray_prev, with_host_arrays=True)
        zeros_t = np.zeros((n_preview, 4), np.float32)
        cl.enqueue_copy(queue_cl, io_web.targets_buf, zeros_t).wait()

        print(f"[web remote] preview {W_ds}x{H_ds} rays={n_preview} spr={samples_per_ray_prev}")

    # -----------------------------
    # IPC live preview (viewer connects, training renders)
    # -----------------------------
    enable_ipc_preview = False
    ipc_host = "127.0.0.1"
    ipc_port = 5005

    preview_ipc = None
    io_ipc = None
    ev_ipc = None
    last_ipc_enqueue_t = 0.0

    # policy
    min_interval_moving = 0.2   # >= 5 fps while moving
    interval_still = 0.5        # 2 fps while still
    still_grace = 0.15          # if RT arrived recently, treat as moving

    if enable_ipc_preview:
        preview_ipc = PreviewIPC(ipc_host, ipc_port)
        preview_ipc.set_init_rt(RTs[0])
        preview_ipc.start()

        # Ensure idx_flat/idy_flat exist; if you disabled export_training_previews,
        # create them here similarly.
        try:
            _ = idx_flat.shape[0]
            _ = idy_flat.shape[0]
        except NameError:
            W_ds, H_ds = W_prev // down, H_prev // down
            x_ds = np.tile(np.arange(W_ds, dtype=np.int32), H_ds)
            y_ds = np.repeat(np.arange(H_ds, dtype=np.int32), W_ds)
            idx_flat = np.clip(x_ds * down + (down // 2), 0, W_prev - 1).astype(np.int32)
            idy_flat = np.clip(y_ds * down + (down // 2), 0, H_prev - 1).astype(np.int32)

        n_preview = idx_flat.size  # 9216 for 72x128
        io_ipc = ft.RayBatch.init(ctx, queue_cl, num_rays=n_preview, samples_per_ray=samples_per_ray_prev, with_host_arrays=True)
        zeros_t = np.zeros((n_preview, 4), np.float32)
        cl.enqueue_copy(queue_cl, io_ipc.targets_buf, zeros_t).wait()

        print(f"[preview ipc] ready: {W_ds}x{H_ds}, rays={n_preview}, spr={samples_per_ray_prev}")


    # -----------------------------
    # Loss logging
    # -----------------------------
    loss_log_path, loss_log_f = open_loss_log(training_dir, model_id)
    print(f"Loss log -> {loss_log_path.resolve()}")

    # SIGINT handler
    old_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _handle_sigint)

    t_start = time.time()
    last_print = t_start
    last_save = t_start
    last_preview = t_start

    step0 = int(getattr(nerf, "steps_trained", 0))
    step = step0
    last_train_event = None
    pending_web_save = False

    try:
        do_report_state = True
        do_first_preview = True
        while step < step0 + steps:
            if STOP_REQUESTED:
                print("[I] Stopping training loop cleanly.")
                break
            # ---- web remote: read controls & handle stop ----
            if web_remote is not None:
                ctrl = web_remote.pop_controls()
                pending_web_save |= ctrl.save_request
                if ctrl.stop_request:
                    STOP_REQUESTED = True

            # ---- web remote: if preview finished, download + stream ----
            if web_remote is not None and io_web is not None and ev_web is not None:
                try:
                    done = (ev_web.command_execution_status == cl.command_execution_status.COMPLETE)
                except Exception:
                    done = False

                if done:
                    out, ev_read = io_web.download_out(wait=False, wait_for=[ev_web])
                    ev_read.wait()

                    img = out[:, :3].reshape((H_ds, W_ds, 3))
                    img = np.clip(img, 0.0, 1.0)
                    bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

                    ok, jpg = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    if ok:
                        web_remote.broadcast_jpeg(jpg.tobytes(), W_ds, H_ds, step)

                    if pending_web_save:
                        fname = Path(training_dir) / f"{model_id}_preview_{W_ds}p_step{step}_live.png"
                        cv2.imwrite(str(fname), bgr)
                        print(f"[web remote] saved {fname}")
                        pending_web_save = False

                    ev_web = None


            # ---- IPC preview: if finished, download + send; handle save request ----
            if preview_ipc is not None and io_ipc is not None:
                if ev_ipc is not None:
                    try:
                        done = (ev_ipc.command_execution_status == cl.command_execution_status.COMPLETE)
                    except Exception:
                        done = False
                    if done:
                        out, ev_read = io_ipc.download_out(wait=False, wait_for=[ev_ipc])
                        ev_read.wait()
                        img = out[:, :3].reshape((H_ds, W_ds, 3))
                        img = np.clip(img, 0.0, 1.0)
                        bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

                        # send to viewer
                        preview_ipc.send_frame(bgr, step=step)

                        # handle 'P' save (save current on-screen frame)
                        if preview_ipc.pop_save_request():
                            fname = Path(training_dir) / f"{model_id}_preview_{W_ds}p_step{step}_live.png"
                            cv2.imwrite(str(fname), bgr)
                            print(f"[preview ipc] saved {fname}")

                        ev_ipc = None


            # LR schedule
            lr_f = np.float32(np.maximum(
                lr * np.sqrt((1024 * 64) / (batch_rays * samples_per_ray)) * np.exp(-lmbda * step),
                LR_FLOOR
            ))
            color_MLP_rel_lr = color_MLP_rel_lr_function(step)

            # spherical coordinates multiplier &
            # occ update ramp
            sh_multiplier = sh_multiplier_function(step)
            occ_update_strength = np.float32(occ.occ_update_strength0 * occ_u_multiplier_function(step))   

            # -----------------------------
            # Periodic metrics + feedback (does not enqueue training, same as your current behavior)
            # -----------------------------
            if period > 0 and (step % period == 0) and do_report_state:
                pbm = _take_one_completed_for_metrics()
                if pbm is None:
                    print(f"[step {step}] loss check skipped (no completed batch yet).")
                    do_report_state = False
                    continue

                cl.wait_for_events([pbm.ev_done])

                cl.enqueue_copy(queue_cl, out_host_period, pbm.io.out_buf).wait()
                pred = out_host_period[:, :3]
                targ = pbm.batch.targets[:, :3]

                err = np.clip(np.sqrt(np.mean((pred - targ) ** 2, axis=1)).astype(np.float32), 0.0, 1.0)
                try:
                    feedback_q.put_nowait((pbm.batch.im_idx.copy(), pbm.batch.idx.copy(), pbm.batch.idy.copy(), err))
                except queue.Full:
                    pass

                mse = float(np.mean((pred - targ) ** 2))
                rmse = float(np.sqrt(mse))
                l2 = float(np.sqrt(np.mean(np.sum((pred - targ) ** 2, axis=1))))
                dt_now = time.time() - t_start
                write_loss_row(loss_log_f, dt_now, step, lr_f, mse, rmse, l2, [len(inflight)], batch_q.qsize())

                print(f"[step {step}] dt={dt_now:.2f}s  lr_f={lr_f:.3e}  mse={mse:.6e}  rmse_rgb={rmse:.6e}  l2mean={l2:.6e}")

                # recycle IO
                try:
                    free_io_q.put_nowait(pbm.io)
                except queue.Full:
                    pass
                do_report_state=False
                continue

            # -----------------------------
            # Get prepared IO + launch training via Nerf
            # -----------------------------
            _wait_until_slot_available()

            try:
                pb = ready_io_q.get(timeout=0.2)
                step += 1
                do_report_state = True

            except queue.Empty:
                time.sleep(0.0005)
                continue

            # Parameter updates and scratch buffers are shared across batches.
            # Uploads may overlap; compute must wait for the previous readers/writer.
            wait_for_upload = [e for e in (pb.ev_upload, last_train_event, ev_web, ev_ipc) if e is not None]

            # enqueue training step (ffnerf + weight_update inside)
            _, ev_step = nerf.step(
                io=pb.io,
                lr_f=float(lr_f),
                do_train=True,
                return_output=False,
                sh_multiplier=float(sh_multiplier),
                color_MLP_rel_lr=color_MLP_rel_lr,
                occ_update_strength=float(occ_update_strength),
                rng_u32=None,
                wait=False,
                global_size=(pb.io.B_total,),
                local_size=(pb.io.samples_per_ray,),
                wait_for=wait_for_upload,
            )

            pb.ev_done = ev_step 
            last_train_event = ev_step
            pb.step_id = step
            inflight.append(pb)

            # ---- web remote: schedule preview render (serialized after training step) ----
            if web_remote is not None and io_web is not None and ev_web is None:
                nowp = time.time()
                moving = (ctrl.moving_flag != 0) or ((nowp - ctrl.last_input_time) < still_grace)
                interval = min_interval_moving if moving else interval_still

                if (nowp - last_web_enqueue_t) >= interval:
                    RT_live = build_c2w_rt_yaw_pitch(ctrl.cam_pos, ctrl.yaw, ctrl.pitch)
                    rays = ft.rays_from_rt_indices(RT_live, idx_flat, idy_flat, W_prev, H_prev, K=K_preview)

                    ev_r = cl.enqueue_copy(queue_cl, io_web.rays_buf, rays, wait_for=[ev_step])
                    _, ev_web = nerf.step(
                        io=io_web,
                        lr_f=0.0,
                        do_train=False,
                        return_output=False,
                        occ_update_strength=0.0,
                        wait=False,
                        global_size=(io_web.B_total,),
                        local_size=(io_web.samples_per_ray,),
                        wait_for=[ev_r],
                    )
                    last_web_enqueue_t = nowp

            # ---- IPC preview: enqueue according to moving/still policy ----
            if preview_ipc is not None and io_ipc is not None and ev_ipc is None and preview_ipc.has_client():
                nowp = time.time()
                RT_new, moving_flag, t_last = preview_ipc.pop_pending_rt()

                # determine moving/still
                moving = (moving_flag != 0) or ((nowp - t_last) < still_grace)
                interval = min_interval_moving if moving else interval_still

                if (nowp - last_ipc_enqueue_t) >= interval:
                    # If no fresh RT, still mode should re-render occasionally so training updates show up
                    if RT_new is None:
                        # no request -> render the first RT in RT_LIST[0] or a reasonable default
                        # (viewer will start sending RTs immediately once connected)
                        RT_new = RT_LIST[0].copy() if "RT_LIST" in locals() else np.eye(4, dtype=np.float32)

                    rays_ipc = ft.rays_from_rt_indices(RT_new, idx_flat, idy_flat, W_prev, H_prev, K=K_preview)

                    ev_r = cl.enqueue_copy(queue_cl, io_ipc.rays_buf, rays_ipc, wait_for=[ev_step])
                    _, ev_ipc = nerf.step(
                        io=io_ipc,
                        lr_f=0.0,
                        do_train=False,
                        return_output=False,
                        occ_update_strength=0.0,
                        wait=False,
                        global_size=(io_ipc.B_total,),
                        local_size=(io_ipc.samples_per_ray,),
                        wait_for=[ev_r],
                    )
                    last_ipc_enqueue_t = nowp


            now = time.time()
            if now - last_print > 15.0:
                print(f"[step {step}] dt={now-t_start:.2f}s inflight={len(inflight)}  batch_q={batch_q.qsize()}  ready_q={ready_io_q.qsize()}  free_io={free_io_q.qsize()}  lr_f={lr_f:.3e}")
                last_print = now

            # -----------------------------
            # Periodic save + preview (uses Nerf tools)
            # -----------------------------
            dT_report_save = preview["save_seconds"]
            if now - last_save > dT_report_save:
                # ensure inflight done (simple policy: finish everything)
                queue_cl.finish()

                saved = nerf.save(min_new_file_seconds=300)
                if saved is not None:
                    print(f"Saved model -> {saved.resolve()}")

                last_save = now

            dT_report_preview = preview["image_seconds"]
            if now - last_preview > dT_report_preview or do_first_preview == True:
                if export_training_previews and preview_jobs:
                    imgs = []
                    for job in preview_jobs:
                        out, ev_preview = nerf.step(
                            io=job["io"],
                            lr_f=0.0,
                            do_train=False,
                            return_output=True,
                            occ_update_strength=0.0,
                            wait=True,
                            global_size=(job["io"].B_total,),
                            local_size=(job["io"].samples_per_ray,),
                            wait_for=[ev_step],
                        )
                        img = out[:, :3].reshape((job["H_ds"], job["W_ds"], 3))
                        img = np.clip(img, 0.0, 1.0)
                        img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                        imgs.append(img)

                    fname = Path(training_dir) / f"{model_id}_preview_{preview_jobs[0]['W_ds']}p_step{step}_multiview.png"
                    concatenated = np.concatenate(imgs, axis=1)
                    cv2.imwrite(str(fname), concatenated)
                    print(f"[preview] saved {fname}")
                    do_first_preview = False

                last_preview = now

    finally:
        # stop threads
        if preview_ipc is not None:
            preview_ipc.stop()
        if web_remote is not None:
            web_remote.stop()

        stop_evt.set()
        io_stop_evt.set()
        prod.join(timeout=1.0)
        io_thr.join(timeout=1.0)

        # wait outstanding work
        try:
            queue_cl.finish()
        except Exception:
            pass

        # final save
        try:
            saved = nerf.save(min_new_file_seconds=0)
            if saved is not None:
                print(f"Saved model -> {saved.resolve()}")
        except Exception as e:
            print("[W] Final save failed:", repr(e))
            raise

        loss_log_f.close()
        signal.signal(signal.SIGINT, old_handler)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "render"], default="train")
    ap.add_argument("--kernel", type=str, default=ft.KERNEL_FILE_DEFAULT)
    ap.add_argument("--rng_seed", type=int, default=777)

    ap.add_argument("--npz_path", type=str, default="")
    ap.add_argument("--demo_path", type=str, default="./test_data_gen")
    ap.add_argument("--rt_file", type=str, default="")

    ap.add_argument("--training_dir", type=str, default="./training")
    ap.add_argument("--model_id", type=str, default="ffnerf")
    ap.add_argument("--load_id", type=str, default="")

    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--period", type=int, default=100)

    ap.add_argument("--samples_per_ray", type=int, default=32)
    ap.add_argument("--batch_rays", type=int, default=64)

    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lmd", type=float, default=LMBDA_BASE)
    ap.add_argument("--dt", type=float, default=0.2)  # used only to derive dt_max default

    ap.add_argument("--snapshot_im", type=int, default=0)
    ap.add_argument("--save_name", type=str, default="./training/snapshot")
    ap.add_argument("--chunk_rays", type=int, default=4096)
    ap.add_argument("--downscale", type=int, default=1)
    ap.add_argument("--RT_input", type=str, default="None")

    args = ap.parse_args()

    kernel_file = Path(args.kernel)
    npz_path = Path(args.npz_path) if args.npz_path else None
    demo_path = Path(args.demo_path)
    # data_dir = Path(args.data_dir)
    training_dir = Path(args.training_dir)
    rt_file = Path(args.rt_file) if args.rt_file else (demo_path / "pyramids_RT.txt")
    load_id = args.load_id.strip() if args.load_id.strip() else None

    if args.mode == "train":
        train(
            kernel_file=kernel_file,
            npz_path=npz_path,
            demo_path=demo_path,
            rt_file=rt_file,
            training_dir=training_dir,
            model_id=args.model_id,
            load_id=load_id,
            steps=int(args.steps),
            period=int(args.period),
            samples_per_ray=int(args.samples_per_ray),
            batch_rays=int(args.batch_rays),
            lr=float(args.lr),
            dt=float(args.dt),
            rng_seed=int(args.rng_seed),
            lmbda_base=(args.lmd),
        )
    else:
        if not load_id:
            raise ValueError("--mode render requires --load_id <id> to load a saved model.")

        RT_input = None
        if args.RT_input != "None":
            RT_input = np.asarray(ast.literal_eval(args.RT_input), dtype=np.float32)
            if RT_input.size == 16:
                RT_input = RT_input.reshape(4, 4)
            if RT_input.shape != (4, 4):
                raise ValueError(f"RT_input must be shape (4,4), got {RT_input.shape}")

        ap.error("Use ffnerf_cli.py snapshot --config <scene.json> to render a saved model")


if __name__ == "__main__":
    main()
