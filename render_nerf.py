#!/usr/bin/env python3
# render_nerf.py
"""
live_nerf_viewer_5x5_25pass.py

Live viewer for your PyOpenCL NeRF with:
- Fixed W,H = 640x360
- 25-step progressive rendering using 5x5 offsets
  (each step renders 72*128 = 9216 rays; one pixel per 5x5 block)
- While moving: keep showing the last completed image; schedule exactly ONE job at a time,
  and keep a ONE-SLOT pending RT request that gets overwritten by the newest camera pose.
  => you get updates during continuous motion as soon as each pass0 job completes.
- While still: keep refining passes 1..24 for the current RT.
- Display fill: only within each 5x5 block (no cross-block blur).
  Missing pixels in a block are filled with the mean of known pixels in that block.
  After pass0, each block is uniformly colored by that single pixel.

REQUIREMENTS / ASSUMPTIONS:
1) In Nerf.step(), remove any unconditional `ev_ff.wait()` so wait=False is actually async.
2) You added: nerf.setup_buffers(num_rays, samples_per_ray, do_train=False) and it works.

Place this script next to run_ffnerf_OG.py (same folder) and run it.

Controls:
- WASD: move in horizontal plane (relative to camera yaw/pitch)
- Q/E: down/up (world Z)
- Arrow keys: yaw/pitch
- ESC or window close: exit
"""
import socket
import struct

import time
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pygame
import pyopencl as cl

import ffnerf_tools as ft

import threading
import queue
import argparse

# ----------------------------
# Config
# ----------------------------

W, H = 96*12, 54*12  #128*5,72*5 #160,90,#960,544 # 1920, 1080 #1280, 720 #640, 360
BLOCK_SIZE = 12   # B: block is BxB pixels; total passes = B*B
WINDOW_SCALE = (720)//H
assert WINDOW_SCALE>0
FPS = 30

SAMPLES_PER_RAY = 64
PASS_RAYS = (W // BLOCK_SIZE) * (H // BLOCK_SIZE)  # one ray per block per pass  # eg 128 * 72 = 9216
TOTAL_PASSES = BLOCK_SIZE * BLOCK_SIZE
IO_DEPTH = 2  # single running job; one IO is enough. set 2 if you want slack.

# Populated by the shared scene config before opening the viewer.
TRAINING_DIR = Path("./training")
MODEL_ID = "ffnerf"
LOAD_ID = MODEL_ID
KERNEL_FILE = Path(ft.KERNEL_FILE_DEFAULT)
VIEW_K = None
INITIAL_RT = None

# If you have constants in run_ffnerf_OG, prefer them; else fall back to defaults.
DECAY_FILE = Path(getattr(ft, "KERNEL_DECAY", "./decay_occ_grid_in-st.cl"))
WU_FILE = Path(getattr(ft, "KERNEL_W_B_UPDATES", "./weight_update.cl"))
TRANSFER_FMT = getattr(ft, "TRANSFER_FMT", "fp16")
KERNEL_NAME = getattr(ft, "KERNEL_NAME", "ffnerf_train_step")

MOVE_SPEED = 1.0   # world units / sec
TURN_SPEED = 1.0   # rad / sec

# When stationary for this long, start refinement passes beyond pass0
REFINE_AFTER_SECONDS = 0.07

# ---------------------------
# Socket helper
# ---------------------------
def recv_exact(sock: socket.socket, n: int) -> bytes:
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

_RT_FMT = "<c16fBB"
_F_HDR_FMT = "<cIIII"
_F_HDR_SIZE = struct.calcsize(_F_HDR_FMT)


# ----------------------------
# Camera RT
# ----------------------------

def build_c2w_rt_yaw_pitch(pos_xyz: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    """
    Camera-to-world 4x4 RT.     -> R=[(eu),(ev),(er)].T
    World up: +Z.
    yaw: rotate around +Z       -> right arrow is negative
    pitch: up/down              -> around right (u)  'up arrow' is negative
    Camera forward axis is + w_cam   -> u x v = w
    """
    pos_xyz = np.asarray(pos_xyz, dtype=np.float32).reshape(3,)

    # Use a real world-up (most common); only change if your dataset truly uses -Z as up.
    up_world = np.array([0.0, 0.0, +1.0], dtype=np.float32)

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)

    # forward in world coords (your current convention)
    forward = np.array([cy * cp, sy * cp, sp], dtype=np.float32)
    forward /= (np.linalg.norm(forward) + 1e-12)

    right = np.cross(forward, up_world).astype(np.float32)
    rn = np.linalg.norm(right)
    if rn < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        right /= (rn + 1e-12)

    up = np.cross(right, forward).astype(np.float32)
    up /= (np.linalg.norm(up) + 1e-12)    # ok so far

    R = np.stack([right, -up, forward], axis=1).astype(np.float32) # Important: in camera view, e_v is pointing down

    # ---- FIX: expected 180° rotation around forward axis (roll by pi)
    # This flips both camera x and y, leaves z unchanged.
    # Q_roll_pi = np.diag(np.array([-1.0, -1.0, 1.0], dtype=np.float32))
    # R = R @ Q_roll_pi

    RT = np.eye(4, dtype=np.float32)
    RT[:3, :3] = R
    RT[:3, 3] = pos_xyz
    return RT

# ----------------------------
# 5x5 Voronoi fill maps (mosaic refinement)
# ----------------------------

def precompute_voronoi_maps_block(block_size: int, offsets: List[Tuple[int, int]]):
    """
    offsets: list of (di,dj) in the order you render passes; expected length B*B.
    Returns:
      maps[k] for k=1..B*B, each is (B,B,2) int:
        maps[k][dj,di] = (src_di, src_dj) of nearest chosen sample among first k offsets.
    """
    B = int(block_size)
    if B <= 0:
        raise ValueError("block_size must be positive")
    total = B * B
    if len(offsets) != total:
        raise ValueError(f"offsets length must be block_size^2={total}, got {len(offsets)}")

    maps = [None] * (total + 1)  # ignore index 0

    # grid points inside a block in (di,dj)
    grid = np.array([(i, j) for j in range(B) for i in range(B)], dtype=np.int32)  # (B^2,2)

    for k in range(1, total + 1):
        chosen = np.array(offsets[:k], dtype=np.int32)  # (k,2)
        d2 = ((grid[:, None, :] - chosen[None, :, :]) ** 2).sum(axis=2)  # (B^2,k)
        arg = d2.argmin(axis=1)     # (B^2,)
        nearest = chosen[arg]       # (B^2,2)
        maps[k] = nearest.reshape(B, B, 2)  # indexed by [dj,di]
    return maps


def compose_display_voronoi_block(
    fb: np.ndarray,
    known: np.ndarray,
    fallback_img: np.ndarray,
    block_size: int,
    vor_map_k: np.ndarray
) -> np.ndarray:
    """
    Voronoi (nearest-sample) fill inside each block_size x block_size block.
    If a whole block has no known samples yet, fallback to fallback_img in that block.

    fb: (H,W,3) float32
    known: (H,W) bool
    fallback_img: (H,W,3) float32
    block_size: int (B)
    vor_map_k: (B,B,2) int, where vor_map_k[dj,di] = (src_di, src_dj)
    """
    B = int(block_size)
    Hb, Wb = H // B, W // B

    fb_blk = fb.reshape(Hb, B, Wb, B, 3).transpose(0, 2, 1, 3, 4)   # (Hb,Wb,B,B,3) [dj,di]
    kn_blk = known.reshape(Hb, B, Wb, B).transpose(0, 2, 1, 3)      # (Hb,Wb,B,B)

    any_known = kn_blk.any(axis=(2, 3))  # (Hb,Wb)

    src_di = vor_map_k[..., 0]  # (B,B)
    src_dj = vor_map_k[..., 1]  # (B,B)

    # gather: for each block, for each (dj,di), pick fb_blk at nearest sampled location
    src_colors = fb_blk[:, :, src_dj, src_di, :]  # (Hb,Wb,B,B,3)

    fallback_blk = fallback_img.reshape(Hb, B, Wb, B, 3).transpose(0, 2, 1, 3, 4)
    out_blk = np.where(any_known[:, :, None, None, None], src_colors, fallback_blk)

    out = out_blk.transpose(0, 2, 1, 3, 4).reshape(H, W, 3)
    return out


# ----------------------------
# Progressive sampling: 25 offsets in a 5x5 block
# ----------------------------

def farthest_point_order_block(block_size: int) -> List[Tuple[int, int]]:
    """
    Order offsets (di,dj) in {0..block_size-1}^2 by farthest-point sampling.
    Starts at center, then repeatedly picks the point farthest from the chosen set.
    """
    B = int(block_size)
    if B <= 0:
        raise ValueError("block_size must be positive")

    pts = [(i, j) for j in range(B) for i in range(B)]

    # start at the "center" (for even B, choose the lower center)
    c = (B // 2, B // 2)
    chosen = [c]
    remaining = [p for p in pts if p != c]

    def d2(a, b):
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

    while remaining:
        best = None
        best_score = -1.0
        for p in remaining:
            score = min(d2(p, c) for c in chosen)
            if score > best_score:
                best_score = score
                best = p
        chosen.append(best)
        remaining.remove(best)
    return chosen


def make_pass_indices_block(W: int, H: int, block_size: int, di: int, dj: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    One full-frame pass:
    sample one pixel in each block_size x block_size block at offset (di,dj).
    Requires W,H divisible by block_size.
    """
    B = int(block_size)
    if (W % B) != 0 or (H % B) != 0:
        raise ValueError(f"This scheme requires W and H divisible by block_size={B}.")

    bx = np.arange(W // B, dtype=np.int32)
    by = np.arange(H // B, dtype=np.int32)
    xx, yy = np.meshgrid(bx, by)

    idx = (xx * B + di).reshape(-1).astype(np.int32)
    idy = (yy * B + dj).reshape(-1).astype(np.int32)
    return idx, idy


def draw_loading_screen(screen, W, H, WINDOW_SCALE, t, status="Building OpenCL program..."):
    """
    Simple placeholder so the OS doesn't mark the app unresponsive.
    """
    # Background
    screen.fill((15, 15, 18))

    # A small animated bar
    bar_w = int(0.7 * W * WINDOW_SCALE)
    bar_h = max(6, int(0.02 * H * WINDOW_SCALE))
    x0 = (W * WINDOW_SCALE - bar_w) // 2
    y0 = (H * WINDOW_SCALE) // 2

    # Outline
    pygame.draw.rect(screen, (60, 60, 70), pygame.Rect(x0, y0, bar_w, bar_h), width=1)

    # Fill (oscillating)
    phase = 0.5 * (1.0 + math.sin(2.0 * math.pi * 0.6 * t))
    fill_w = max(1, int(phase * (bar_w - 2)))
    pygame.draw.rect(screen, (140, 140, 170), pygame.Rect(x0 + 1, y0 + 1, fill_w, bar_h - 2))

    # Text
    font = pygame.font.Font(None, 28)
    txt1 = font.render(status, True, (220, 220, 230))
    txt2 = pygame.font.Font(None, 22).render("Close window / press ESC to cancel", True, (170, 170, 190))

    screen.blit(txt1, (x0, y0 - 40))
    screen.blit(txt2, (x0, y0 + 20))


# ----------------------------
# Block-local fill (no cross-block blur)
# ----------------------------

def compose_display_block_local(fb: np.ndarray, known: np.ndarray, fallback_img: np.ndarray) -> np.ndarray:
    """
    Fill missing pixels using per-5x5-block mean of known pixels, only within each block.
    If a pixel is still unknown (possible before the first pass0), use fallback_img there.
    """
    Hb, Wb = H // 5, W // 5

    fb_blk = fb.reshape(Hb, 5, Wb, 5, 3).transpose(0, 2, 1, 3, 4)    # (Hb,Wb,5,5,3)
    kn_blk = known.reshape(Hb, 5, Wb, 5).transpose(0, 2, 1, 3)       # (Hb,Wb,5,5)

    kn_f = kn_blk[..., None].astype(np.float32)
    s = (fb_blk * kn_f).sum(axis=(2, 3), keepdims=True)  # (Hb,Wb,1,1,3)
    c = kn_f.sum(axis=(2, 3), keepdims=True)            # (Hb,Wb,1,1,1)

    eps = 1e-8
    mean = s / (c + eps)

    filled_blk = np.where(kn_blk[..., None], fb_blk, mean)
    filled = filled_blk.transpose(0, 2, 1, 3, 4).reshape(H, W, 3)

    # fallback where still unknown
    filled = np.where(known[..., None], filled, fallback_img)
    return filled


# ----------------------------
# OpenCL event helpers
# ----------------------------

def ev_complete(ev: cl.Event) -> bool:
    return ev.get_info(cl.event_info.COMMAND_EXECUTION_STATUS) == cl.command_execution_status.COMPLETE


# ----------------------------
# Render job container
# ----------------------------

@dataclass
class RenderJob:
    rt_id: int
    pass_id: int
    RT: np.ndarray
    idx: np.ndarray
    idy: np.ndarray
    io: ft.RayBatch
    ev_done: cl.Event


# ----------------------------
# build program
# ----------------------------

def build_nerf_and_io_worker(result_q, error_q, stop_evt):
    """
    Runs the heavy OpenCL build + Nerf construction in a background thread.
    Sends (ctx, queue_cl, is_ooo, nerf, free_ios, pass_idx, pass_idy) on success.
    """
    try:
        # If user closed the window early, abort fast
        if stop_evt.is_set():
            return

        plat, dev = ft.pick_first_device()

        ctx = cl.Context([dev])
        queue_cl, is_ooo = ft.make_queue(ctx, dev)

        # Build/load nerf (same as your current main() init)
        box_min = np.array([-3.0, -3.0, -3.0], np.float32)
        box_max = np.array([+3.0, +3.0, +3.0], np.float32)

        occ = ft.OccParams(
            G_occ=32,
            box_min=box_min,
            box_max=box_max,
            rho_min=np.float32(1.0),
            rho_max=np.float32(16.0),
            dt_min=np.float32(np.max(box_max - box_min) / 8192),
            dt_max=np.float32(12.8 / SAMPLES_PER_RAY / 6 * np.max(box_max - box_min)),
            t_near_clip=np.float32(0.3),
            t_far_clip=np.float32(np.linalg.norm(box_max - box_min)),
            occ_update_strength0=np.float32(0.0),
        ).finalize()

        L_POS = ft.L_POS
        M = ft.M
        D_OUT = ft.D_OUT
        L_DIR = ft.L_DIR
        C_OUT = ft.C_OUT

        layers = {
            "d0":  L_POS,
            "d1":  M,
            "d2":  D_OUT,
            "c0":  D_OUT + L_DIR,
            "c1":  M,
            "c2":  M,
            "rgb": C_OUT,
            "dir0": L_DIR,
        }
        edges = {
            "d1":  ("Wd0", "d0"),
            "d2":  ("Wd1", "d1"),
            "c1":  ("Wc0", "c0"),
            "c2":  ("Wc1", "c1"),
            "rgb": ("Wc2", "c2"),
        }

        nerf = ft.Nerf.from_kernels(
            ctx=ctx,
            queue=queue_cl,
            kernel_file=KERNEL_FILE,
            decay_file=DECAY_FILE,
            weight_update_file=WU_FILE,
            background="black",
            layers=layers,
            edges=edges,
            training_dir=TRAINING_DIR,
            model_id=MODEL_ID,
            load_id=LOAD_ID,
            transfer_fmt=TRANSFER_FMT,
            ffnerf_kernel_name=KERNEL_NAME,
            decay_kernel_name="occ_nibble_decay_write_both_q24",
            wu_kernel_name="update_ffnerf_oneWG",
            build_options_extra=None,
            rng_seed=777,
            occ=occ,
            table_size_entries=getattr(ft, "TABLE_SIZE_ENTRIES", 2**19),
            f_per_level=getattr(ft, "F_PER_LEVEL", 2),
        )
        nerf.queue = queue_cl

        # Ensure scratch matches viewer workload
        nerf.setup_buffers(PASS_RAYS, SAMPLES_PER_RAY, do_train=False)

        # IO pool
        free_ios = []
        for _ in range(IO_DEPTH):
            io = ft.RayBatch.init(ctx, queue_cl, num_rays=PASS_RAYS, samples_per_ray=SAMPLES_PER_RAY, with_host_arrays=True)
            io.targets_host = np.zeros((PASS_RAYS, 4), np.float32)
            cl.enqueue_copy(queue_cl, io.targets_buf, io.targets_host).wait()
            free_ios.append(io)

        # Precompute 25 passes (offset order) + indices + Voronoi maps
        offsets = farthest_point_order_block(BLOCK_SIZE)

        # NEW: Voronoi fill maps for k=1..25 (mosaic refinement)
        vor_maps = precompute_voronoi_maps_block(BLOCK_SIZE, offsets)

        pass_idx = []
        pass_idy = []
        for di, dj in offsets:
            idx, idy = make_pass_indices_block(W, H, BLOCK_SIZE, di, dj)
            pass_idx.append(idx)
            pass_idy.append(idy)

        if stop_evt.is_set():
            return

        result_q.put((ctx, queue_cl, is_ooo, nerf, free_ios, pass_idx, pass_idy, vor_maps))

    except Exception as e:
        error_q.put(e)


# ----------------------------
# Main
# ----------------------------
def main_client(connect: str):
    host, port_s = connect.split(":")
    port = int(port_s)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setblocking(False)

    pygame.init()
    screen = pygame.display.set_mode((W * WINDOW_SCALE, H * WINDOW_SCALE))
    clock = pygame.time.Clock()

    cam_pos = np.array([-2.0, 0.0, 0.5], dtype=np.float32)
    yaw = 0.0
    pitch = 0.0

    frozen_bgr = np.zeros((H, W, 3), np.uint8)
    latest_step = 0

    last_time = time.time()
    last_send_t = 0.0
    SEND_FPS = 30.0
    send_interval = 1.0 / SEND_FPS

    running = True
    while running:
        now = time.time()
        dt = now - last_time
        last_time = now

        moved = False
        save = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_p:
                    save = 1

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            running = False

        # rotation
        if keys[pygame.K_LEFT]:
            yaw += TURN_SPEED * dt; moved = True
        if keys[pygame.K_RIGHT]:
            yaw -= TURN_SPEED * dt; moved = True
        if keys[pygame.K_UP]:
            pitch -= TURN_SPEED * dt; moved = True
        if keys[pygame.K_DOWN]:
            pitch += TURN_SPEED * dt; moved = True
        pitch = max(-1.45, min(1.45, pitch))

        # movement (same as your existing viewer)
        up_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        forward = np.array([cy * cp, sy * cp, sp], dtype=np.float32)
        forward /= (np.linalg.norm(forward) + 1e-12)

        right = np.cross(forward, up_world).astype(np.float32)
        rn = np.linalg.norm(right)
        if rn < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            right /= (rn + 1e-12)
        up = np.cross(right, forward).astype(np.float32)
        up /= (np.linalg.norm(up) + 1e-12)

        v = np.zeros(3, dtype=np.float32)
        if keys[pygame.K_w]: v += forward; moved = True
        if keys[pygame.K_s]: v -= forward; moved = True
        if keys[pygame.K_d]: v += right;   moved = True
        if keys[pygame.K_a]: v -= right;   moved = True
        if keys[pygame.K_e]: v += up;      moved = True
        if keys[pygame.K_q]: v -= up;      moved = True

        if np.linalg.norm(v) > 1e-8:
            v /= (np.linalg.norm(v) + 1e-12)
            cam_pos += (MOVE_SPEED * dt) * v

        RT_new = build_c2w_rt_yaw_pitch(cam_pos, yaw, pitch)

        # send RT at fixed rate (and immediately on save press)
        if (now - last_send_t) >= send_interval or save:
            msg = struct.pack(_RT_FMT, b"R", *RT_new.reshape(-1).tolist(), int(moved), int(save))
            try:
                sock.sendall(msg)
            except Exception:
                running = False
            last_send_t = now

        # receive frames (nonblocking)
        try:
            # read header if available
            sock.setblocking(False)
            hdr = sock.recv(_F_HDR_SIZE, socket.MSG_PEEK)
            if len(hdr) == _F_HDR_SIZE:
                hdr = recv_exact(sock, _F_HDR_SIZE)
                typ, w, h, step, nbytes = struct.unpack(_F_HDR_FMT, hdr)
                if typ == b"F":
                    payload = recv_exact(sock, nbytes)
                    img = np.frombuffer(payload, dtype=np.uint8).reshape((h, w, 3))
                    frozen_bgr = img.copy()
                    latest_step = int(step)
        except BlockingIOError:
            pass
        except Exception:
            # connection died
            running = False

        # display
        rgb = frozen_bgr[..., ::-1]
        surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        if WINDOW_SCALE != 1:
            surf = pygame.transform.scale(surf, (W * WINDOW_SCALE, H * WINDOW_SCALE))
        screen.blit(surf, (0, 0))
        pygame.display.set_caption(f"NeRF Viewer (IPC) | step={latest_step} | P=save | ESC=quit")
        pygame.display.flip()
        clock.tick(FPS)

    try:
        sock.close()
    except Exception:
        pass
    pygame.quit()


def main(config=None, output=None):
    global W, H, BLOCK_SIZE, WINDOW_SCALE, PASS_RAYS, TOTAL_PASSES
    global SAMPLES_PER_RAY, TRAINING_DIR, MODEL_ID, LOAD_ID, VIEW_K, INITIAL_RT, MOVE_SPEED
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, help="Shared scene JSON configuration")
    ap.add_argument("--connect", type=str, default="", help="host:port to connect to training preview server")
    args = ap.parse_args([] if config is not None else None)

    if args.connect:
        return main_client(args.connect)

    if config is None:
        if args.config is None:
            ap.error("standalone review requires --config")
        from scene_config import load_config
        config = load_config(args.config)
    from sfm_data import read_scene, image_geometry, scaled_intrinsics
    paths, poses, K, _, _ = read_scene(config["data"])
    v, t = config["viewer"], config["training"]
    if not 0 <= v["camera"] < len(poses):
        raise ValueError("viewer.camera is outside the registered-camera list")
    INITIAL_RT = poses[v["camera"]]
    first, _ = image_geometry(paths[v["camera"]], config["data"])
    W, H = v["width"], v["height"]
    VIEW_K = scaled_intrinsics(K, (first.shape[1], first.shape[0]), (W, H), config["data"]["sfm_max_side"])
    BLOCK_SIZE = v["block_size"]
    WINDOW_SCALE = 1
    PASS_RAYS = (W // BLOCK_SIZE) * (H // BLOCK_SIZE)
    TOTAL_PASSES = BLOCK_SIZE * BLOCK_SIZE
    SAMPLES_PER_RAY = v["samples_per_ray"]
    MOVE_SPEED = v["move_speed"]
    TRAINING_DIR, MODEL_ID = Path(t["directory"]), t["model_id"]
    LOAD_ID = MODEL_ID
    if not list(TRAINING_DIR.glob(f"{LOAD_ID}_weights_*.npz")):
        raise FileNotFoundError(f"No checkpoints for {LOAD_ID} in {TRAINING_DIR}")

    if output is not None:
        # Render the exact dataset pose, including roll, without opening a window.
        result_q, error_q, stop_evt = queue.Queue(), queue.Queue(), threading.Event()
        build_nerf_and_io_worker(result_q, error_q, stop_evt)
        if not error_q.empty():
            raise error_q.get()
        _, queue_cl, _, nerf, ios, pass_idx, pass_idy, _ = result_q.get()
        frame = np.zeros((H, W, 3), np.float32)
        io = ios[0]
        for idx, idy in zip(pass_idx, pass_idy):
            rays = ft.rays_from_rt_indices(INITIAL_RT, idx, idy, W, H, K=VIEW_K)
            io.upload_rays(rays, wait=True)
            out, _ = nerf.step(io=io, lr_f=0.0, do_train=False, return_output=True,
                               occ_update_strength=0.0, wait=True,
                               global_size=(io.B_total,), local_size=(io.samples_per_ray,))
            frame[idy, idx] = out[:, :3]
        queue_cl.finish()
        if not np.isfinite(frame).all():
            raise ValueError("Snapshot contains non-finite pixels")
        import cv2
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        bgr = cv2.cvtColor((np.clip(frame, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(output), bgr):
            raise OSError(f"Could not save snapshot: {output}")
        print(f"Saved snapshot: {output.resolve()}")
        return


    # pygame init FIRST (so we can show loading UI)
    pygame.init()
    screen = pygame.display.set_mode((W * WINDOW_SCALE, H * WINDOW_SCALE))
    clock = pygame.time.Clock()

    # Start worker thread for OpenCL build + model load
    result_q = queue.Queue(maxsize=1)
    error_q = queue.Queue(maxsize=1)
    stop_evt = threading.Event()

    worker = threading.Thread(
        target=build_nerf_and_io_worker,
        args=(result_q, error_q, stop_evt),
        daemon=True,
    )
    worker.start()

    # Loading loop (keeps window responsive)
    t0 = time.time()
    while True:
        # allow quit during build
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stop_evt.set()
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                stop_evt.set()
                pygame.quit()
                return

        if not error_q.empty():
            e = error_q.get()
            stop_evt.set()
            pygame.quit()
            raise e

        if not result_q.empty():
            (ctx, queue_cl, is_ooo, nerf, free_ios, pass_idx, pass_idy, vor_maps) = result_q.get()
            break

        draw_loading_screen(
            screen, W, H, WINDOW_SCALE,
            t=time.time() - t0,
            status="Building OpenCL program / loading model..."
        )
        pygame.display.flip()
        clock.tick(30)

    # ---- from here on, continue with your existing viewer loop ----
    # (keep using `screen`, `clock`, and the received ctx/queue_cl/nerf/free_ios/pass_idx/pass_idy)


    def launch_pass(RT: np.ndarray, pass_id: int, rt_id: int) -> Optional[RenderJob]:
        if not free_ios:
            return None
        io = free_ios.pop()

        idx = pass_idx[pass_id]
        idy = pass_idy[pass_id]
        rays = ft.rays_from_rt_indices(RT, idx, idy, W, H, K=VIEW_K)

        ev_r = cl.enqueue_copy(queue_cl, io.rays_buf, rays)
        # forward-only, async
        _, ev_done = nerf.step(
            io=io,
            lr_f=0.0,
            do_train=False,
            return_output=False,
            occ_update_strength=0.0,
            wait=False,
            global_size=(io.B_total,),
            local_size=(io.samples_per_ray,),
            wait_for=[ev_r],
        )
        return RenderJob(rt_id=rt_id, pass_id=pass_id, RT=RT, idx=idx, idy=idy, io=io, ev_done=ev_done)

    # Persistent buffers for current refinement image (current_RT only)
    fb = np.zeros((H, W, 3), np.float32)
    known = np.zeros((H, W), np.bool_)

    # No ghosting: keep a "frozen" last completed display frame to show while moving
    frozen_img = np.zeros((H, W, 3), np.float32)

    # Track how many passes have completed for the current_rt_id (1..25)
    completed_passes = 0

    # current refinement target
    current_rt_id: Optional[int] = None
    current_RT: Optional[np.ndarray] = None
    pass_cursor = 0  # next pass to schedule for current_RT

    running_job: Optional[RenderJob] = None
    pending_RT: Optional[np.ndarray] = None  # overwritten while moving
    pending_request = False

    # Start at the selected dataset camera (free navigation uses world +Z up).
    from nerf_web_remote import cam_state_from_c2w
    cam_pos, yaw, pitch = cam_state_from_c2w(INITIAL_RT)

    rt_id_counter = 0
    still_timer = 0.0

    last_time = time.time()
    running = True

    while running:
        now = time.time()
        dt = now - last_time
        last_time = now

        moved = False

        # events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            running = False

        # rotation
        if keys[pygame.K_LEFT]:
            yaw += TURN_SPEED * dt # rotation around world_up, right handed convention
            moved = True
        if keys[pygame.K_RIGHT]:
            yaw -= TURN_SPEED * dt
            moved = True
        if keys[pygame.K_UP]:
            pitch -= TURN_SPEED * dt #rotation around right, but pitch should tilt fwd when K_UP
            moved = True
        if keys[pygame.K_DOWN]:
            pitch += TURN_SPEED * dt
            moved = True

        pitch = max(-1.45, min(1.45, pitch))

        # Must match build_c2w_rt_yaw_pitch convention
        _="""
        Camera-to-world 4x4 RT.     -> R=[(eu),(ev),(er)].T
        World up: +Z.
        yaw: rotate around +Z       -> right arrow is negative
        pitch: up/down              -> around right (u)  'up arrow' is negative
        Camera forward axis is + w_cam   -> u x v = w
        """
        up_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        yaw_eff = yaw
        cy, sy = math.cos(yaw_eff), math.sin(yaw_eff)
        cp, sp = math.cos(pitch), math.sin(pitch)
        forward = np.array([cy * cp, sy * cp, sp], dtype=np.float32)
        forward /= (np.linalg.norm(forward) + 1e-12)

        right = np.cross(forward, up_world).astype(np.float32)
        rn = np.linalg.norm(right)
        if rn < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            right /= (rn + 1e-12)

        up = np.cross(right, forward).astype(np.float32)

        v = np.zeros(3, dtype=np.float32)
        if keys[pygame.K_w]:
            v += forward; moved = True
        if keys[pygame.K_s]:
            v -= forward; moved = True
        if keys[pygame.K_d]:
            v += right; moved = True
        if keys[pygame.K_a]:
            v -= right; moved = True
        if keys[pygame.K_e]:
            v += up; moved = True
        if keys[pygame.K_q]:
            v -= up; moved = True

        if np.linalg.norm(v) > 1e-8:
            v /= (np.linalg.norm(v) + 1e-12)
            cam_pos += (MOVE_SPEED * dt) * v
        #ok
        RT_new = build_c2w_rt_yaw_pitch(cam_pos, yaw, pitch)

        # motion -> update pending RT (overwrite)
        if moved:
            still_timer = 0.0
            pending_RT = RT_new
            pending_request = True
        else:
            still_timer += dt

        # ---- retire completed job (if any) ----
        if running_job is not None and ev_complete(running_job.ev_done):
            out, ev_read = running_job.io.download_out(wait=False, wait_for=[running_job.ev_done])
            ev_read.wait()
            rgb = np.clip(out[:, :3], 0.0, 1.0)

                        # If pass0 finished, it defines the new current refinement target.
            if running_job.pass_id == 0:
                current_rt_id = running_job.rt_id
                current_RT = running_job.RT
                pass_cursor = 1
                fb.fill(0.0)
                known.fill(False)
                completed_passes = 0  # will become 1 once we apply pass0 pixels below

            # Apply results only if they belong to current_rt_id
            # (prevents refinements from older RTs smearing into new view)
            if current_rt_id is not None and running_job.rt_id == current_rt_id:
                fb[running_job.idy, running_job.idx, :] = rgb
                known[running_job.idy, running_job.idx] = True

                # update completed pass count (pass_id is 0-based)
                completed_passes = max(completed_passes, running_job.pass_id + 1)

                # Build full-bright mosaic estimate for the current partial data
                k = max(1, min(TOTAL_PASSES, completed_passes))
                display_img = compose_display_voronoi_block(
                    fb, known,
                    fallback_img=frozen_img,   # blocks with no samples yet fall back to last frozen frame
                    block_size=BLOCK_SIZE,
                    vor_map_k=vor_maps[k]
                )
                frozen_img = np.clip(display_img, 0.0, 1.0)  # update what we show while moving

            # recycle IO
            free_ios.append(running_job.io)
            running_job = None

        # ---- scheduling: exactly one job at a time + one-slot overwrite-able pending ----
        if running_job is None:
            if pending_request and pending_RT is not None:
                # schedule newest view pass0 (movement streaming)
                rt_id_counter += 1
                running_job = launch_pass(pending_RT, pass_id=0, rt_id=rt_id_counter)
                pending_request = False
                pending_RT = None

            else:
                # no pending request
                if current_RT is None:
                    # initial: schedule pass0 for current RT
                    rt_id_counter += 1
                    running_job = launch_pass(INITIAL_RT, pass_id=0, rt_id=rt_id_counter)

                else:
                    # refine only if still and not yet done
                    if still_timer >= REFINE_AFTER_SECONDS and pass_cursor < TOTAL_PASSES:
                        running_job = launch_pass(current_RT, pass_id=pass_cursor, rt_id=current_rt_id)
                        pass_cursor += 1

        # ---- display (no ghosting): show the frozen mosaic image ----
        # pygame expects (W,H,3). We have (H,W,3).
        # img_disp = np.flipud(img_u8) / np.fliplr(img_u8) : # vertical flip / # horizontal flip
        img_u8 = (np.clip(frozen_img, 0.0, 1.0) * 255.0).astype(np.uint8)
        img_disp = img_u8
        surf = pygame.surfarray.make_surface(img_disp.swapaxes(0, 1))

        if WINDOW_SCALE != 1:
            surf = pygame.transform.scale(surf, (W * WINDOW_SCALE, H * WINDOW_SCALE))
        screen.blit(surf, (0, 0))


        filled = float(known.mean()) * 100.0 if current_RT is not None else 0.0
        inflight = 1 if running_job is not None else 0
        pygame.display.set_caption(
            f"NeRF Live {BLOCK_SIZE}x{BLOCK_SIZE}x{TOTAL_PASSES} | filled={filled:5.1f}% | done={completed_passes:02d}/{TOTAL_PASSES} | next={pass_cursor:02d}/{TOTAL_PASSES} | inflight={inflight} | pending={'Y' if pending_request else 'N'}"
        )

        pygame.display.flip()
        clock.tick(FPS)

    try:
        queue_cl.finish()
    except Exception:
        pass
    pygame.quit()


if __name__ == "__main__":
    main()
