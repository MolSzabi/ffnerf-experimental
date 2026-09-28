import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import numpy as np
import pyopencl as cl


def event_ms(evt):
    evt.wait()
    return (evt.profile.end - evt.profile.start) * 1e-6


device_preference_order = ['AMD', 'Intel'] # highest comes first

KERNEL_FILE_DEFAULT = str(Path(__file__).resolve().parent / "ffnerf_OG.cl")
KERNEL_NAME = "ffnerf_train_step"
KERNEL_DECAY = str(Path(__file__).resolve().parent / "decay_occ_grid_in-st.cl")
KERNEL_W_B_UPDATES = str(Path(__file__).resolve().parent / "weight_update.cl")

###############################################################
#   Must match kernel defines
###############################################################
TABLE_SIZE_ENTRIES = 2_260_992 # 983_040   # number of hash entries
F_PER_LEVEL = 4                 # features per entry
TABLE_ELEMS = TABLE_SIZE_ENTRIES * F_PER_LEVEL
L_POS = 16*F_PER_LEVEL
D_OUT = 16
L_DIR = 16
C_IN  = D_OUT + L_DIR  # 32
C_OUT = 3
M     = 64
S_MAX = 64 #256 #128
ND, NC = (2, 2,)
NDMAX, NCMAX = (3,4,) # fixed unless kernels are updated

TRANSFER_FMT = "fp32" #"fp16" # "fp32" | "fp16" | "fp8" #baseline fp16 
#               # performance diff -> results are clearer and sharper (in the good way) with fp32. fp8 barely learns.
##FVEC = 8 #NOTE: unsuccesful attempts

if TRANSFER_FMT == "fp32":
    transfer_nbytes = 4
    transfer_define = 32
elif TRANSFER_FMT == "fp16":
    transfer_nbytes = 2
    transfer_define = 16
elif TRANSFER_FMT == "fp8":
    transfer_nbytes = 1
    transfer_define = 8
else:
    raise ValueError("TRANSFER_FMT must be 'fp32', 'fp16', or 'fp8'")

# Adam hyperparams (MLP only)
ADAM_BETA1 = np.float32(0.9)
ADAM_BETA2 = np.float32(0.999)
ADAM_EPS   = np.float32(1e-8)


# Training related parameters
SIGMA_BIAS=-2.0
OCC_U_STR = 1 # 0.05     # occ_update_strength0
DECAY=0.9
RHO_MIN_DEF = 1.0 # 1.0
RHO_MAX_DEF = 64.0 #64.0
OCC_INIT_DENSITY = 0xEEEEEEEE # pick 0..F for each nibble
OCC_MAINTENANCE_PERIOD = 32




# DEVICE SPECS DEPENDENT!
# baseline: (16,16,1,1,236) # 30ms per 1024*64 sample on weight_update kernel (most optimal)
WU_WG_X_DEFAULT  = 16  #  13 #
WU_WG_Y_DEFAULT  = 16  #  13 #
WU_REG_X_DEFAULT = 1   #  1  #
WU_REG_Y_DEFAULT = 1   #  1  #
WU_TILEK_DEFAULT = 236 # 306 #
# (x+y)*T+y*x
# 248*2*16*4B+256*4B -> 31744+1024=2^15
# t*2*8+64 -> 8192 -> tmax=508
# NOTE: conclusions from tests: 
#   utilization of most local memory per WG is key, 
#   with large local memory usage WG per CU will be limited
#   -> utilizing every wavefront we can is the best (max 256 thread/WG) 
#   around max value of TILEK we need to find one where barrier's effect is minimal (2^k and modulos), the best I found here is 256
#       (16,16,1,1,248) 31.30016 31.25424 31.39312 31.34816     avg: 31.32392
#       (16,16,1,1,240) 34.73280 34.89552 34.74544 35.19680     avg: 34.89264 #eg of worst in scope
#       (16,16,1,1,236) 30.07536 29.92976 29.95408 29.90784     avg: 29.96676
#       (16,16,1,1,232) 31.09312 31.10304 30.97584 31.15632     avg: 31.08208
#       (16,16,1,1,128) 43.52304 43.33248 43.37968 43.70224     avg: 43.48436 # eg of smaller TILEK: underutilized memory
#       (8,8,1,1,64)    52.74256 51.91152 52.14704 52.51776         #underutilized tiling capacity (one CU can fit multiple of this)
#       (8,8,1,1,508)   53.28879 53.43584 53.46736 53.23039         # underutilized CU (only one of this can fit on CU), and tiling is 2x better with 4x the size
#       (11,11,1,1,366) 41.3 | 364 41.75 | 360 42.20 | 248 41.33 | 
#       (13,13,1,1,308) 35.35| 306 35.89 | 304 41.32 | 248 41.33 | 
#   furthermore we need to consider the size of the matrices. Here 64x64 dominates, so its 4x4 piece of 16x16 tiles
#   meaning with 32 CUs and one WG per CU, we can still have our results in one go
#       (16,16,2,2,118) --- # no data, but underutilized device capacity (4 WGs needed, will fit on 4CUs, but there are 32 available)
#   in case of barely larger matrices than the 32 CU can handle at once, 
#   we need to consider register tiling at cost of TILEK for least start to end time
#   in case of considerably larger matrices we minimize global read-write per result, keeping in mind caching



# ----------------------------
# OpenCL helpers
# ----------------------------
def pick_first_device():
    plats = cl.get_platforms()
    if not plats:
        raise RuntimeError("No OpenCL platforms found.")
    print("[I]: OpenCL platforms:", plats)
    for p in sorted(
        plats, 
        key=lambda p: sum([
            (len(device_preference_order)-i if name in p.get_info(cl.platform_info.NAME) else 0)
              for i, name in enumerate(device_preference_order)]), 
        reverse = True):
        devs = p.get_devices()
        print("[I]: platform devices on", p, ":", devs)
        if devs:
            return p, devs[0]
    raise RuntimeError("No OpenCL devices found on any platform.")


def make_queue(ctx, dev):
    """Try out-of-order queue; fall back to in-order if unsupported."""
    props = 0
    try:
        props |= cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE
        props |= cl.command_queue_properties.PROFILING_ENABLE
        q = cl.CommandQueue(ctx, properties=props)
        return q, True
    except Exception:
        print("Unsuccesful setup for out-of-order queue; fall back to default")
        q = cl.CommandQueue(ctx)
        return q, False


def f32_to_i32_bits(x_f32: np.ndarray) -> np.ndarray:
    x_f32 = np.asarray(x_f32, dtype=np.float32)
    return x_f32.view(np.int32)


def i32_bits_to_f32(x_i32: np.ndarray) -> np.ndarray:
    x_i32 = np.asarray(x_i32, dtype=np.int32)
    return x_i32.view(np.float32)


def event_is_complete(ev: cl.Event) -> bool:
    try:
        return ev.command_execution_status == cl.command_execution_status.COMPLETE
    except Exception:
        try:
            st = ev.get_info(cl.event_info.COMMAND_EXECUTION_STATUS)
            return st == cl.command_queue_properties.COMPLETE
        except Exception:
            return True


def prune_done(events):
    return [e for e in events if not event_is_complete(e)]



# ----------------------------
# Rays & targets
# ----------------------------
ray_dtype = np.dtype([("o", np.float32, 4), ("d", np.float32, 4)], align=True)
if ray_dtype.itemsize != 32:
    raise RuntimeError(f"Unexpected Ray dtype size: {ray_dtype.itemsize} (expected 32)")


def rays_targets_from_indices(im_idx, u_idx, v_idx, Ims, c2w_all, W, H, K=None):
    """
    im_idx: (n,) image indices into Ims/c2w_all
    u_idx, v_idx: (n,) pixel coords with u right, v down
    Ims: (N,H,W,3) float32 in [0,1]
    c2w_all: (N,4,4) camera-to-world
    K: (3,3) or (N,3,3) optional intrinsics
    """
    im_idx = np.asarray(im_idx, dtype=np.int32)
    u_idx  = np.asarray(u_idx,  dtype=np.int32)
    v_idx  = np.asarray(v_idx,  dtype=np.int32)
    n = im_idx.size

    assert Ims.ndim == 4 and Ims.shape[1] == H and Ims.shape[2] == W, (Ims.shape, H, W)
    assert c2w_all.shape[0] == Ims.shape[0]

    # --- camera-frame directions
    if K is not None:
        if K.ndim == 2:
            fx, fy = float(K[0,0]), float(K[1,1])
            cx, cy = float(K[0,2]), float(K[1,2])
            x = (u_idx.astype(np.float32) + 0.5 - cx) / fx
            y = (v_idx.astype(np.float32) + 0.5 - cy) / fy
        else:
            # per-image K
            Ki = K[im_idx].astype(np.float32)
            fx = Ki[:,0,0]; fy = Ki[:,1,1]
            cx = Ki[:,0,2]; cy = Ki[:,1,2]
            x = (u_idx.astype(np.float32) + 0.5 - cx) / fx
            y = (v_idx.astype(np.float32) + 0.5 - cy) / fy

        z = np.ones((n,), dtype=np.float32)
        rays_cam = np.stack([x, y, z], axis=1)  # OpenCV cam: x right, y down, z forward
    else:
        # your fallback heuristic (kept compatible)
        px = (u_idx.astype(np.float32) - (W / 2.0) + 0.5)
        py = (v_idx.astype(np.float32) - (H / 2.0) + 0.5)
        pz = np.full((n,), np.sqrt(H * W), dtype=np.float32)
        rays_cam = np.stack([px, py, pz], axis=1)

    # --- rotate to world
    R = c2w_all[im_idx, :3, :3].astype(np.float32)
    t = c2w_all[im_idx, :3, 3].astype(np.float32)

    dirs = np.einsum("nij,nj->ni", R, rays_cam)
    dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12).astype(np.float32)

    rays = np.zeros((n,), dtype=ray_dtype)
    rays["o"][:, :3] = t
    rays["o"][:, 3]  = 1.0
    rays["d"][:, :3] = dirs
    rays["d"][:, 3]  = 0.0

    rgb = Ims[im_idx, v_idx, u_idx, :3]
    if rgb.dtype != np.float32:
        rgb = rgb.astype(np.float32) / 255.0

    targets = np.zeros((n, 4), dtype=np.float32)
    targets[:, :3] = rgb
    return rays, targets


def rays_from_rt_indices(RT, idx, idy, W, H, K=None):
    RT = np.asarray(RT, dtype=np.float32)
    if RT.shape != (4, 4):
        raise ValueError(f"RT must be shape (4,4), got {RT.shape}")

    idx = np.asarray(idx, dtype=np.int32).reshape(-1)
    idy = np.asarray(idy, dtype=np.int32).reshape(-1)
    if idx.size != idy.size:
        raise ValueError("idx and idy must have the same length.")
    n = idx.size

    if K is None:
        px = idx.astype(np.float32) - W / 2.0 + 0.5
        py = idy.astype(np.float32) - H / 2.0 + 0.5
        pz = np.full(n, np.sqrt(H * W), dtype=np.float32)
    else:
        px = (idx.astype(np.float32) + 0.5 - K[0, 2]) / K[0, 0]
        py = (idy.astype(np.float32) + 0.5 - K[1, 2]) / K[1, 1]
        pz = np.ones(n, dtype=np.float32)
    rays_cam = np.stack([px, py, pz], axis=1)

    R = RT[:3, :3].astype(np.float32)
    t = RT[:3, 3].astype(np.float32)

    dirs = rays_cam @ R.T
    norms = np.linalg.norm(dirs, axis=1, keepdims=True).astype(np.float32)
    dirs = dirs / (norms + 1e-12)

    origins = np.broadcast_to(t[None, :], (n, 3)).copy()

    rays = np.zeros((n,), dtype=ray_dtype)
    rays["o"][:, :3] = origins
    rays["o"][:, 3] = 1.0
    rays["d"][:, :3] = dirs
    rays["d"][:, 3] = 0.0
    return rays


def default_ray_dtype() -> np.dtype:
    dt = np.dtype([("o", np.float32, 4), ("d", np.float32, 4)], align=True)
    if dt.itemsize != 32:
        raise RuntimeError(f"Unexpected ray dtype size: {dt.itemsize} (expected 32)")
    return dt

@dataclass
class RayBatch:
    """
    Detached IO bundle:
      - host arrays (optional but convenient)
      - device buffers for rays/targets/out
      - remembers num_rays + samples_per_ray so Nerf can size scratch and launch
    """
    ctx: cl.Context
    queue: cl.CommandQueue
    num_rays: int
    samples_per_ray: int
    ray_dtype: np.dtype = None

    ev_last: cl.Event | None = None
    # host-side (optional convenience)
    rays_host: Optional[np.ndarray] = None      # (num_rays,) ray_dtype
    targets_host: Optional[np.ndarray] = None   # (num_rays,4) float32
    out_host: Optional[np.ndarray] = None       # (num_rays,4) float32

    # device-side
    rays_buf: Optional[cl.Buffer] = None
    targets_buf: Optional[cl.Buffer] = None
    out_buf: Optional[cl.Buffer] = None

    def __post_init__(self):
        self.num_rays = int(self.num_rays)
        self.samples_per_ray = int(self.samples_per_ray)
        if self.num_rays <= 0 or self.samples_per_ray <= 0:
            raise ValueError("num_rays and samples_per_ray must be > 0")
        if self.ray_dtype is None:
            self.ray_dtype = default_ray_dtype()
        if self.ray_dtype.itemsize != 32:
            raise ValueError("ray_dtype must be 32 bytes (float4 o, float4 d)")

    @property
    def B_total(self) -> int:
        return self.num_rays * self.samples_per_ray

    @classmethod
    def init(
        cls,
        ctx: cl.Context,
        queue: cl.CommandQueue,
        num_rays: int,
        samples_per_ray: int,
        ray_dtype: np.dtype = None,
        with_host_arrays: bool = True,
        flags_rays=cl.mem_flags.READ_ONLY,
        flags_targets=cl.mem_flags.READ_ONLY,
        flags_out=cl.mem_flags.WRITE_ONLY,
    ) -> "RayBatch":
        rb = cls(ctx=ctx, queue=queue, num_rays=num_rays, samples_per_ray=samples_per_ray, ray_dtype=ray_dtype)

        mf = cl.mem_flags
        rays_bytes = rb.num_rays * rb.ray_dtype.itemsize
        tgt_bytes  = rb.num_rays * 16   # float4
        out_bytes  = rb.num_rays * 16

        rb.rays_buf    = cl.Buffer(ctx, flags_rays,    size=rays_bytes)
        rb.targets_buf = cl.Buffer(ctx, flags_targets, size=tgt_bytes)
        rb.out_buf     = cl.Buffer(ctx, flags_out,     size=out_bytes)

        if with_host_arrays:
            rb.rays_host = np.empty((rb.num_rays,), dtype=rb.ray_dtype)
            rb.targets_host = np.empty((rb.num_rays, 4), dtype=np.float32)
            rb.out_host = np.empty((rb.num_rays, 4), dtype=np.float32)

        return rb

    # ---------- Upload / download ----------
    def upload_rays(self, rays: np.ndarray, wait: bool = False) -> cl.Event:
        rays = np.asarray(rays)
        if rays.shape != (self.num_rays,):
            raise ValueError(f"rays must be shape ({self.num_rays},), got {rays.shape}")
        if rays.dtype.itemsize != 32:
            raise ValueError(f"rays dtype must be 32B, got {rays.dtype} (itemsize {rays.dtype.itemsize})")
        ev = cl.enqueue_copy(self.queue, self.rays_buf, rays)
        if wait:
            ev.wait()
        return ev

    def upload_targets(self, targets: np.ndarray, wait: bool = False) -> cl.Event:
        targets = np.asarray(targets, dtype=np.float32)
        if targets.shape != (self.num_rays, 4):
            raise ValueError(f"targets must be shape ({self.num_rays},4), got {targets.shape}")
        ev = cl.enqueue_copy(self.queue, self.targets_buf, targets)
        if wait:
            ev.wait()
        return ev

    def download_out(self, out: Optional[np.ndarray] = None, wait: bool = True, 
                     wait_for: List[cl.Event | None] = None) -> tuple[np.ndarray, cl.Event]:
        if out is None:
            out = self.out_host
        if out is None:
            out = np.empty((self.num_rays, 4), dtype=np.float32)
        else:
            out = np.asarray(out, dtype=np.float32)
            if out.shape != (self.num_rays, 4):
                raise ValueError(f"out must be shape ({self.num_rays},4), got {out.shape}")
        ev = cl.enqueue_copy(self.queue, out, self.out_buf, wait_for=wait_for)
        if wait:
            ev.wait()
        return out, ev

    def ensure_dummy_targets(self) -> np.ndarray:
        """
        Handy for forward-only calls where kernel ABI wants a targets buffer.
        """
        if self.targets_host is None:
            self.targets_host = np.zeros((self.num_rays, 4), dtype=np.float32)
        else:
            self.targets_host.fill(0.0)
        return self.targets_host


@dataclass
class OccParams:
    G_occ: int = 32
    box_min: np.ndarray = None   # (3,) float32
    box_max: np.ndarray = None   # (3,) float32
    rho_min: np.float32 = np.float32(RHO_MIN_DEF)
    rho_max: np.float32 = np.float32(RHO_MAX_DEF)
    dt_min:  np.float32 = np.float32(4e-3)
    dt_max:  np.float32 = np.float32(0.8)
    t_near_clip: np.float32 = np.float32(0.3)
    t_far_clip:  np.float32 = np.float32(1e9)
    occ_update_strength0: np.float32 = np.float32(OCC_U_STR)

    def finalize(self):
        if self.box_min is None:
            self.box_min = np.array([-3.0, -3.0, -3.0], dtype=np.float32)
        if self.box_max is None:
            self.box_max = np.array([+3.0, +3.0, +3.0], dtype=np.float32)
        self.box_min = np.asarray(self.box_min, dtype=np.float32).reshape(3)
        self.box_max = np.asarray(self.box_max, dtype=np.float32).reshape(3)
        self.t_near_clip = np.float32(self.t_near_clip)
        self.t_far_clip = np.float32(self.t_far_clip)
        if not np.isfinite(self.t_near_clip) or self.t_near_clip < 0:
            raise ValueError("t_near_clip must be finite and nonnegative")
        if not np.isfinite(self.t_far_clip) or self.t_far_clip <= self.t_near_clip:
            raise ValueError("t_far_clip must be finite and greater than t_near_clip")
        return self




class Nerf:
    """
    Fully fused NeRF host-side model container:
      - owns numpy state (int32 bit patterns)
      - owns OpenCL buffers (params + intermediates + IO)
      - owns save/load
      - step() runs ffnerf kernel, and optionally the weight_update kernel
    """

    # Adam defaults (MLP only) - keep here so trainer loop stays smaller
    ADAM_BETA1 = np.float32(0.9)
    ADAM_BETA2 = np.float32(0.999)
    ADAM_EPS   = np.float32(1e-6) # np.float32(1e-8)

    def __init__(
        self,
        ctx: cl.Context,
        queue: cl.CommandQueue,
        ffnerf_prg: cl.Program,
        ffnerf_suppress_mode_prg: cl.Program,
        decay_prg: cl.Program,
        wu_prg: cl.Program,
        ffnerf_kernel_name: str,
        decay_kernel_name: str,
        wu_kernel_name: str,
        training_dir: str | Path,
        model_id: str,

        # ordered dicts (normal dict ok in py3.7+; insertion order preserved)
        layers: dict[str, int],
        edges: dict[str, tuple[int, int]],

        transfer_fmt: str = "fp16",
        table_size_entries: int = TABLE_SIZE_ENTRIES,
        f_per_level: int = F_PER_LEVEL,

        occ: OccParams | None = None,
        load_id: str | None = None,
        rng_seed: int = 777,
    ):
        self.do_train = True # assume every time we can train, TODO adjust better logic
        # Needed to work with kernel
        self.M = M
        self.L_POS = L_POS
        self.L_DIR = L_DIR
        self.C_OUT = C_OUT
        self.D_D_OUT = D_OUT
        self.C_IN = D_OUT + L_DIR 

        self.ctx = ctx
        self.queue = queue
        self.ffnerf_prg = ffnerf_prg
        self.ffnerf_suppress_mode_prg = ffnerf_suppress_mode_prg
        self.decay_prg = decay_prg
        self.wu_prg = wu_prg
        self.ffnerf_knl = cl.Kernel(ffnerf_prg, ffnerf_kernel_name)
        self.ffnerf_suppress_mode_knl = cl.Kernel(ffnerf_suppress_mode_prg, ffnerf_kernel_name)
        self.decay_knl = cl.Kernel(decay_prg, decay_kernel_name)
        self.wu_knl = cl.Kernel(wu_prg, wu_kernel_name)

        self.training_dir = Path(training_dir)
        self.training_dir.mkdir(parents=True, exist_ok=True)
        self.model_id = model_id

        self.layers = dict(layers)     # insertion order preserved
        self.edges = dict(edges)

        # Weight-update tiling (must match what weight_update.cl was compiled with).
        # If you always compile with fixed defaults, you can hardcode these.
        # Better: store from meta / build config; simplest is to keep them here consistent with from_kernels defaults.
        self.wu_wg_x  = WU_WG_X_DEFAULT
        self.wu_wg_y  = WU_WG_Y_DEFAULT
        self.wu_reg_x = WU_REG_X_DEFAULT
        self.wu_reg_y = WU_REG_Y_DEFAULT

        # weights, activations, deltas = derive_nerf_weights(layers, edges)
        # self.weights_spec = dict(weights)
        # self.activations_spec = dict(activations)
        # self.deltas_spec = dict(deltas)
        # self.layer_names = list(self.layers.keys())
        # if len(self.layer_names) < 2:
        #     raise ValueError("layers must contain at least 2 entries (to define grads/acts ranges).")
        # self.weight_names = list(self.weights_spec.keys())
        # if not self.weight_names:
        #     raise ValueError("weights dict is empty.")
        # # infer biases from weights names: Wd0 -> bd0, Wc2 -> bc2, etc.
        # self.bias_names = ["b" + wn[1:] for wn in self.weight_names]

        # ---- Fixed-arg kernel arch (placeholders for missing layers) ----
        self.ND = int(ND)
        self.NC = int(NC)
        self.NDMAX = int(NDMAX)
        self.NCMAX = int(NCMAX)

        # Defines:
        #   self.weight_names (ALL weights, in canonical order)
        #   self.bias_names   (ALL biases,  in canonical order)
        #   self.active_weight_names / self.active_bias_names
        #   self.weights_spec / self.bias_spec
        #   self.activations_spec / self.deltas_spec (in *kernel arg order*)
        self._set_arch_from_layer_counts(self.ND, self.NC)

        # sanity
        if not self.weight_names or not self.bias_names:
            raise ValueError("Parameter name lists are empty (arch init failed).")


        # transfer format
        self.transfer_fmt = transfer_fmt
        if transfer_fmt == "fp32":
            self.transfer_nbytes = 4
            self.transfer_define = 32
        elif transfer_fmt == "fp16":
            self.transfer_nbytes = 2
            self.transfer_define = 16
        elif transfer_fmt == "fp8":
            self.transfer_nbytes = 1
            self.transfer_define = 8
        else:
            raise ValueError("transfer_fmt must be 'fp32', 'fp16', or 'fp8'")

        # hash table spec # TODO increase hash table size and/or decrease max resolution
        self.table_size_entries = int(table_size_entries)
        self.f_per_level = int(f_per_level)
        self.table_elems = self.table_size_entries * self.f_per_level

        # occupancy params
        self.occ = (occ or OccParams()).finalize()

        # rng (host-side for init + rng_u32 generation)
        self.rng = np.random.default_rng(int(rng_seed))

        # persistent numpy state (int32 bit patterns)
        self.steps_trained = 0
        self.state_i32: dict[str, np.ndarray] = {}   # "Wd0", "bd0", "mWd0", ...
        self.meta: dict[str, np.ndarray | int | str] = {}

        # OpenCL buffers (filled by setup_buffers)
        self.buf: dict[str, cl.Buffer] = {}
        self.io: dict[str, cl.Buffer] = {}
        self.tmp: dict[str, cl.Buffer] = {}

        # dynamic shapes: host-only
        self._shapes = {}  # e.g. {"Wd0": (M, L_POS), "bd0": (M,), ...}

        # load or init immediately (as requested)
        if load_id:
            self.load_latest(load_id)
        else:
            self._init_fresh()

    # -------------------------
    # Construction convenience
    # -------------------------
    @classmethod
    def from_kernels(
        cls,
        ctx: cl.Context,
        queue: cl.CommandQueue,
        kernel_file: str | Path,
        decay_file: str | Path,
        weight_update_file: str | Path,
        background: str, # "black", "white", "random"
        layers: dict[str, int],
        edges: dict[str, tuple[int, int]],
        training_dir: str | Path,
        model_id: str,
        load_id: str | None = None,
        transfer_fmt: str = "fp32",
        ffnerf_kernel_name: str = "ffnerf_train_step",
        decay_kernel_name: str = "occ_nibble_decay_write_both_q24",
        wu_kernel_name: str = "update_ffnerf_oneWG",
        build_options_extra: list[str] | None = None,
        rng_seed: int = 777,
        occ: OccParams | None = None,
        table_size_entries: int = TABLE_SIZE_ENTRIES,
        f_per_level: int = F_PER_LEVEL,
    ):
        kernel_file = Path(kernel_file)
        wu_file = Path(weight_update_file)
        src = kernel_file.read_text(encoding="utf-8") #.read_bytes() #
        dec_src = decay_file.read_text(encoding="utf-8") #.read_bytes() #
        wu_src = wu_file.read_text(encoding="utf-8") #.read_bytes() #

        # ---- weight-update tiling defaults (can be overridden via build_options_extra) ----
        # Pick something conservative and AMD-friendly by default:
        #   - 8x8 WG = 64 threads (1 wavefront on AMD)
        #   - REG 2x2 => tile is 16x16 (often reasonable register pressure)

        opts = [
            "-cl-std=CL1.2",
            f"-DND={ND}",    f"-DNC={NC}",
            f"-DNDMAX={NDMAX}", f"-DNCMAX={NCMAX}",
            f"-DTRANSFER_FMT={32 if transfer_fmt=='fp32' else 16 if transfer_fmt=='fp16' else 8}",
            # weight_update.cl tiling knobs (safe even if other kernels ignore them)
            f"-DWG_X={WU_WG_X_DEFAULT}",
            f"-DWG_Y={WU_WG_Y_DEFAULT}",
            f"-DREG_X={WU_REG_X_DEFAULT}",
            f"-DREG_Y={WU_REG_Y_DEFAULT}",
            # (optional) you can also expose TILEK if you want:
            f"-DTILEK={WU_TILEK_DEFAULT}",
        ]
        match background:
            case "black":
                opts.append(f"-DBACKGROUND_COLOR=0")
            case "white":
                opts.append(f"-DBACKGROUND_COLOR=1")
            case "random":
                opts.append(f"-DBACKGROUND_COLOR=2")

        # Allow caller to override *any* of the above.
        # For example: build_options_extra=["-DWG_X=16","-DWG_Y=16","-DREG_X=2","-DREG_Y=2"]
        if build_options_extra:
            opts += list(build_options_extra)

        cache_dir = str((Path(training_dir) / "_pyopencl_cache").resolve())

        ffnerf_prg = cl.Program(ctx, src).build(options=opts, cache_dir=cache_dir)
        ffnerf_suppress_mode_prg = cl.Program(ctx, src).build(options=opts+[f"-DSURPRESS_SAMPLES=1",], cache_dir=cache_dir)
        decay_prg  = cl.Program(ctx, dec_src).build(options=opts, cache_dir=cache_dir)
        wu_prg     = cl.Program(ctx, wu_src).build(options=opts, cache_dir=cache_dir)

        return cls(
            ctx=ctx,
            queue=queue,
            ffnerf_prg=ffnerf_prg,
            ffnerf_suppress_mode_prg=ffnerf_suppress_mode_prg,
            decay_prg=decay_prg,
            wu_prg=wu_prg,
            ffnerf_kernel_name=ffnerf_kernel_name,
            decay_kernel_name=decay_kernel_name,
            wu_kernel_name=wu_kernel_name,
            training_dir=training_dir,
            model_id=model_id,
            layers=layers,
            edges=edges,
            transfer_fmt=transfer_fmt,
            table_size_entries=table_size_entries,
            f_per_level=f_per_level,
            occ=occ,
            load_id=load_id,
            rng_seed=rng_seed,
        )

    def _set_arch_from_layer_counts(self, ND: int, NC: int):
        ND = int(ND); NC = int(NC)

        if not (1 <= ND <= self.NDMAX):
            raise ValueError(f"ND must be in [1, {self.NDMAX}], got {ND}")
        if not (1 <= NC <= self.NCMAX):
            raise ValueError(f"NC must be in [1, {self.NCMAX}], got {NC}")

        # ---- Parameter canonical names (match kernel arg list) ----
        self.density_w_all = ["Wd_in", "Wd_01", "Wd_12", "Wd_out"]
        self.density_b_all = ["bd_0", "bd_1", "bd_2", "bd_out"]

        self.color_w_all   = ["Wc_in", "Wc_01", "Wc_12", "Wc_23", "Wc_out"]
        self.color_b_all   = ["bc_0", "bc_1", "bc_2", "bc_3", "bc_out"]

        # These are the lists most of your host code iterates over.
        # IMPORTANT: we keep them as ALL params, because the kernel signature always needs them.
        self.weight_names = self.density_w_all + self.color_w_all
        self.bias_names   = self.density_b_all + self.color_b_all

        # ---- Which of the above are "real" given ND/NC ----
        # Density: Wd_in -> bd_0 always, then (ND-1) hidden blocks, then Wd_out/bd_out always
        active_dw = ["Wd_in", "Wd_out"]
        active_db = ["bd_0", "bd_out"]
        if ND >= 2:
            active_dw += ["Wd_01"]
            active_db += ["bd_1"]
        if ND >= 3:
            active_dw += ["Wd_12"]
            active_db += ["bd_2"]

        # Color: Wc_in/bc_0 always, (NC-1) hidden blocks, then Wc_out/bc_out always
        active_cw = ["Wc_in", "Wc_out"]
        active_cb = ["bc_0", "bc_out"]
        if NC >= 2:
            active_cw += ["Wc_01"]; active_cb += ["bc_1"]
        if NC >= 3:
            active_cw += ["Wc_12"]; active_cb += ["bc_2"]
        if NC >= 4:
            active_cw += ["Wc_23"]; active_cb += ["bc_3"]

        self.active_weight_names = set(active_dw + active_cw)
        self.active_bias_names   = set(active_db + active_cb)

        # ---- Shapes for params (inactive => placeholder tiny buffers) ----
        # Use (1,1)/(1,) placeholders so buffers exist but are minimal.
        W_PLACEHOLDER = (1, 1)
        B_PLACEHOLDER = (1,)

        # weights_spec is still used by _infer_shapes; keep it for ALL weights
        self.weights_spec = {
            "Wd_in":  (int(self.M), int(self.L_POS)),
            "Wd_01":  (int(self.M), int(self.M)),
            "Wd_12":  (int(self.M), int(self.M)),
            "Wd_out": (int(self.D_D_OUT), int(self.M)),
            "Wc_in":  (int(self.M), int(self.C_IN)),
            "Wc_01":  (int(self.M), int(self.M)),
            "Wc_12":  (int(self.M), int(self.M)),
            "Wc_23":  (int(self.M), int(self.M)),
            "Wc_out": (int(self.C_OUT), int(self.M)),
        }

        # biases are explicit now (no more name inference)
        self.bias_spec = {
            "bd_0":   (int(self.M),),
            "bd_1":   (int(self.M),),
            "bd_2":   (int(self.M),),
            "bd_out": (int(self.D_D_OUT),),
            "bc_0":   (int(self.M),),
            "bc_1":   (int(self.M),),
            "bc_2":   (int(self.M),),
            "bc_3":   (int(self.M),),
            "bc_out": (int(self.C_OUT),),
        }

        # Apply placeholder shapes to inactive params
        for wn in list(self.weights_spec.keys()):
            if wn not in self.active_weight_names:
                self.weights_spec[wn] = W_PLACEHOLDER
        for bn in list(self.bias_spec.keys()):
            if bn not in self.active_bias_names:
                self.bias_spec[bn] = B_PLACEHOLDER

        # ---- Scratch in *kernel arg order* (MUST match ffnerf_train_step signature) ----
        # Density hidden count = ND (1..NDMAX), so kernel carries dd1..ddNdmax and ad1..adNdmax.
        # Color hidden count   = NC (1..NCMAX), so kernel carries dc1..dcNcmax and ac1..acNcmax.
        # Placeholder policy:
        #   If a scratch corresponds to a layer that is inactive (e.g. dd3 when ND<3),
        #   we set its feature dim to None so it allocates no buffer.
        #   Kernels needs to never index inactive buffers (guarded by ND/NC).
        # --- Deltas (kernel order): dd1 dd2 dd3 dd_out dc1 dc2 dc3 dc4 dc_out ---
        deltas = {}
        # density hidden deltas: gl_dd1..gl_dd{NDMAX}
        for i in range(1, self.NDMAX + 1):
            key = f"gl_dd{i}"
            deltas[key] = int(self.M) if i <= ND else None

        # density output delta always real
        deltas["gl_dd_out"] = int(self.D_D_OUT)

        # color hidden deltas: gl_dc1..gl_dc{NCMAX}
        for i in range(1, self.NCMAX + 1):
            key = f"gl_dc{i}"
            deltas[key] = int(self.M) if i <= NC else None

        # color output delta always real
        deltas["gl_dc_out"] = int(self.C_OUT)

        self.deltas_spec = deltas

        # --- Activations (kernel order): apos ad1 ad2 ad3 ac_in ac1 ac2 ac3 ac4 ---
        acts = {}

        # pos encoding always real
        acts["gl_apos"] = int(self.L_POS)

        # density hidden activations: gl_ad1..gl_ad{NDMAX}
        for i in range(1, self.NDMAX + 1):
            key = f"gl_ad{i}"
            acts[key] = int(self.M) if i <= ND else None

        # color input always real
        acts["gl_ac_in"] = int(self.C_IN)

        # color hidden activations: gl_ac1..gl_ac{NCMAX}
        for i in range(1, self.NCMAX + 1):
            key = f"gl_ac{i}"
            acts[key] = int(self.M) if i <= NC else None

        self.activations_spec = acts

    # -------------------------
    # Init / shapes
    # -------------------------
    def _infer_shapes(self):
        self._shapes.clear()

        # weights (ALL, placeholders already baked in by _set_arch_from_layer_counts)
        for wn, sh in self.weights_spec.items():
            if len(sh) != 2:
                raise ValueError(f"Weight shape for {wn} must be 2D, got {sh}")
            self._shapes[wn] = (int(sh[0]), int(sh[1]))

        # biases (explicit, ALL)
        for bn, sh in self.bias_spec.items():
            if len(sh) != 1:
                raise ValueError(f"Bias shape for {bn} must be 1D, got {sh}")
            self._shapes[bn] = (int(sh[0]),)

        # Adam state for W and b (allocate for ALL; placeholders become tiny)
        for wn in self.weight_names:
            self._shapes["m" + wn] = self._shapes[wn]
            self._shapes["v" + wn] = self._shapes[wn]
        for bn in self.bias_names:
            self._shapes["m" + bn] = self._shapes[bn]
            self._shapes["v" + bn] = self._shapes[bn]

        # table
        self._shapes["table"] = (int(self.table_elems),)

        # occ grid
        self._shapes["occ_grid_u32"] = (int(self.occ.G_occ) ** 3,)

    def _init_fresh(self):
        self._infer_shapes()
        self.steps_trained = 0

        # table small init
        table_f = self.rng.uniform(-1e-4, 1e-4, size=self._shapes["table"]).astype(np.float32)
        self.state_i32["table"] = f32_to_i32_bits(table_f).reshape(-1)

        # He init helper
        def he(shape2):
            fan_in = int(shape2[1])
            return self.rng.normal(0.0, np.sqrt(2.0 / fan_in), size=shape2).astype(np.float32)

        # weights init (ALL weights; inactive are placeholders => zero)
        for wn in self.weight_names:
            sh = self._shapes[wn]
            if wn in self.active_weight_names:
                # heuristic: small init for tiny output heads
                if sh[0] <= 16:
                    w_f = self.rng.normal(0.0, 0.01, size=sh).astype(np.float32)
                else:
                    fan_in = int(sh[1])
                    w_f = self.rng.normal(0.0, np.sqrt(2.0 / fan_in), size=sh).astype(np.float32)
            else:
                w_f = np.zeros(sh, dtype=np.float32)

            self.state_i32[wn] = f32_to_i32_bits(w_f).reshape(-1)

        # biases init (ALL biases; inactive are placeholders => zero)
        for bn in self.bias_names:
            sh = self._shapes[bn]
            if bn in self.active_bias_names:
                b_f = np.zeros(sh, dtype=np.float32)

                # Density sigma is dout[0], so its initialization bias belongs
                # to the density output head, not to a hidden-layer bias.
                if bn == "bd_out":
                    if b_f.size > 0:
                        b_f[0] = SIGMA_BIAS
                        print(f"[I]: {bn}[0] = {SIGMA_BIAS}")

                # - your old "bc2 ~ N(-2.0, 0.05)" => now bc_2
                if bn == "bc_2":
                    b_f = self.rng.normal(-2.0, 0.05, size=sh).astype(np.float32)
            else:
                b_f = np.zeros(sh, dtype=np.float32)

            self.state_i32[bn] = f32_to_i32_bits(b_f).reshape(-1)

        # Adam state = 0
        for wn in self.weight_names:
            self.state_i32["m" + wn] = np.zeros(np.prod(self._shapes[wn]), dtype=np.int32)
            self.state_i32["v" + wn] = np.zeros(np.prod(self._shapes[wn]), dtype=np.int32)
        for bn in self.bias_names:
            self.state_i32["m" + bn] = np.zeros(np.prod(self._shapes[bn]), dtype=np.int32)
            self.state_i32["v" + bn] = np.zeros(np.prod(self._shapes[bn]), dtype=np.int32)

        # occ grid init = zeros
        self.state_i32["occ_grid_in_u32"] = np.full(self._shapes["occ_grid_u32"], OCC_INIT_DENSITY, dtype=np.uint32).view(np.int32)
        self.state_i32["occ_grid_st_u32"] = np.full(self._shapes["occ_grid_u32"], OCC_INIT_DENSITY, dtype=np.uint32).view(np.int32)

        # meta
        self.meta = self._build_meta()

    def _build_meta(self) -> dict:
        return dict(
            timestamp=datetime.now().strftime("%Y%m%d-%H%M%S"),
            model_id=self.model_id,
            steps_trained=int(self.steps_trained),
            transfer_fmt=self.transfer_fmt,
            table_size_entries=int(self.table_size_entries),
            f_per_level=int(self.f_per_level),
            layers=self.layers,  # will be saved as object? we avoid pickles later; store separately
            # occ meta (saved as arrays/scalars)
            G_occ=np.int32(self.occ.G_occ),
            box_min=np.asarray(self.occ.box_min, dtype=np.float32),
            box_max=np.asarray(self.occ.box_max, dtype=np.float32),
            rho_min=np.float32(self.occ.rho_min),
            rho_max=np.float32(self.occ.rho_max),
            dt_min=np.float32(self.occ.dt_min),
            dt_max=np.float32(self.occ.dt_max),
            t_near_clip=np.float32(self.occ.t_near_clip),
            t_far_clip=np.float32(self.occ.t_far_clip),
            ND=np.int32(self.ND),
            NC=np.int32(self.NC),
        )

    # -------------------------
    # Buffers
    # ------------------------- # TODO - implemeent a threaded ray fetcher to update RayBatches
    def setup_buffers(self, num_rays: int, samples_per_ray: int, do_train=None):
        """
        Allocate device buffers for parameters + scratch for a given (num_rays, samples_per_ray).
        Rays/targets/out are NOT owned by Nerf anymore; they come from RayBatch.
        """
        if do_train is not None:
            self.do_train = do_train
        self.num_rays = int(num_rays)
        self.samples_per_ray = int(samples_per_ray)
        self.B = self.num_rays * self.samples_per_ray if self.do_train else None

        mf = cl.mem_flags

        # --- persistent parameter buffers ---
        def make_rw_from_host(name: str, host_i32: np.ndarray):
            host_i32 = np.ascontiguousarray(host_i32.reshape(-1).astype(np.int32, copy=False))
            self.buf[name] = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=host_i32)

        make_rw_from_host("table", self.state_i32["table"])

        for wn in self.weight_names:
            make_rw_from_host(wn, self.state_i32[wn])
        for bn in self.bias_names:
            make_rw_from_host(bn, self.state_i32[bn])

        for wn in self.weight_names:
            make_rw_from_host("m" + wn, self.state_i32["m" + wn])
            make_rw_from_host("v" + wn, self.state_i32["v" + wn])
        for bn in self.bias_names:
            make_rw_from_host("m" + bn, self.state_i32["m" + bn])
            make_rw_from_host("v" + bn, self.state_i32["v" + bn])

        # occ grid as uint32
        occ_in_u32 = self.state_i32["occ_grid_in_u32"].view(np.uint32)
        occ_in_u32 = np.ascontiguousarray(occ_in_u32.reshape(-1))
        self.buf["occ_grid_in_u32"] = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=occ_in_u32)
        occ_st_u32 = self.state_i32["occ_grid_st_u32"].view(np.uint32)
        occ_st_u32 = np.ascontiguousarray(occ_st_u32.reshape(-1))
        self.buf["occ_grid_st_u32"] = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=occ_st_u32) if self.do_train else None

        # --- scratch derived from layers ---
        self.tmp.clear()

        def alloc_tmp(key: str, n_elems: int):
            self.tmp[key] = (
                cl.Buffer(self.ctx, mf.READ_WRITE, size=int(n_elems) * self.transfer_nbytes) if (self.do_train and n_elems is not None) else None
            )

        for lname, lsize in self.deltas_spec.items():
            alloc_tmp(lname, lsize * self.B if (self.do_train and lsize is not None) else None)
        for lname, lsize in self.activations_spec.items():
            alloc_tmp(lname, lsize * self.B if (self.do_train and lsize is not None) else None)

    # -------------------------
    # Save / Load
    # -------------------------
    def _model_paths(self, save_id: str) -> tuple[Path, list[Path]]:
        td = self.training_dir
        td.mkdir(parents=True, exist_ok=True)
        # same convention as you had: timestamped + rolling
        pats = sorted(td.glob(f"{save_id}_weights_*.npz"))
        return td / f"{save_id}_weights_latest.npz", pats

    def save(self, min_new_file_seconds: int = 300) -> Path:
        """
        Copies device -> host for all persistent buffers, then writes .npz.
        """
        # ---- small bugfix in save() payload ----
        # in save(): you had t_far_clip=np.float32(self.occ.t_near_clip)
        # it should be:
        #   t_far_clip=np.float32(self.occ.t_far_clip)
        self.steps_trained = int(self.steps_trained)
        self.meta = self._build_meta()

        # pull from device (authoritative) into state_i32
        q = self.queue

        def pull_i32(name: str, dtype=np.int32):
            host = np.empty_like(self.state_i32[name], dtype=dtype)
            cl.enqueue_copy(q, host, self.buf[name]).wait()
            self.state_i32[name] = host.view(np.int32)

        pull_i32("table", np.int32)

        for wn in self.weight_names:
            pull_i32(wn, np.int32)
        for bn in self.bias_names:
            pull_i32(bn, np.int32)

        for wn in self.weight_names:
            pull_i32("m" + wn, np.int32)
            pull_i32("v" + wn, np.int32)
        for bn in self.bias_names:
            pull_i32("m" + bn, np.int32)
            pull_i32("v" + bn, np.int32)

        # occ grid as uint32
        occ_in_host = np.empty((int(self.occ.G_occ) ** 3,), dtype=np.uint32)
        cl.enqueue_copy(q, occ_in_host, self.buf["occ_grid_in_u32"]).wait()
        occ_st_host = np.empty((int(self.occ.G_occ) ** 3,), dtype=np.uint32)
        cl.enqueue_copy(q, occ_st_host, self.buf["occ_grid_st_u32"]).wait()

        # pick filename
        latest_path, pats = self._model_paths(self.model_id)
        now = time.time()
        make_new = True
        if latest_path.exists():
            try:
                age = now - latest_path.stat().st_mtime
                make_new = age >= float(min_new_file_seconds)
            except OSError:
                make_new = True

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        fname = (self.training_dir / f"{self.model_id}_weights_{ts}.npz") if make_new else latest_path

        # payload: avoid pickles -> store layers/weights in flat arrays
        payload = dict(
            timestamp=ts,
            model_id=self.model_id,
            steps_trained=np.int32(self.steps_trained),
            transfer_fmt=np.array(self.transfer_fmt),

            # store how many layers were active during training
            ND=np.int32(self.ND),
            NC=np.int32(self.NC),

            table_i=np.asarray(self.state_i32["table"], dtype=np.int32),
            table_size_entries=np.int32(self.table_size_entries),
            table_f_per_level=np.int32(self.f_per_level),

            occ_grid_in_u32=np.asarray(occ_in_host, dtype=np.uint32),
            occ_grid_st_u32=np.asarray(occ_st_host, dtype=np.uint32),

            # occ meta
            G_occ=np.int32(self.occ.G_occ),
            box_min=np.asarray(self.occ.box_min, dtype=np.float32),
            box_max=np.asarray(self.occ.box_max, dtype=np.float32),
            rho_min=np.float32(self.occ.rho_min),
            rho_max=np.float32(self.occ.rho_max),
            dt_min=np.float32(self.occ.dt_min),
            dt_max=np.float32(self.occ.dt_max),
            t_near_clip=np.float32(self.occ.t_near_clip),
            t_far_clip=np.float32(self.occ.t_far_clip),
        )

        # Save ONLY active params (no placeholders)
        active_params = sorted(list(self.active_weight_names | self.active_bias_names))

        payload["active_params"] = np.array(active_params, dtype="U32")

        for name in active_params:
            payload[name + "_i"] = np.asarray(self.state_i32[name], dtype=np.int32)
            payload["m" + name + "_i"] = np.asarray(self.state_i32["m" + name], dtype=np.int32)
            payload["v" + name + "_i"] = np.asarray(self.state_i32["v" + name], dtype=np.int32)

        np.savez_compressed(fname, **payload)
        if make_new:
            np.savez_compressed(latest_path, **payload)

        return fname

    def load_latest(self, load_id: str):
        """
        Load the most recently modified checkpoint, including the latest alias.
        """
        td = self.training_dir
        pats = sorted(td.glob(f"{load_id}_weights_*.npz"))
        path = max(pats, key=lambda p: p.stat().st_mtime_ns) if pats else None
        if path is None:
            raise FileNotFoundError(f"No saved model found for id='{load_id}' in {td.resolve()}")

        data = np.load(path, allow_pickle=False)

        # ---- restore ND/NC and rebuild arch (placeholders depend on kernel, not checkpoint) ----
        self.ND = int(data["ND"])
        self.NC = int(data["NC"])
        self._set_arch_from_layer_counts(self.ND, self.NC)

        self.transfer_fmt = str(data["transfer_fmt"].item())
        if self.transfer_fmt == "fp32":
            self.transfer_nbytes = 4
            self.transfer_define = 32
        elif self.transfer_fmt == "fp16":
            self.transfer_nbytes = 2
            self.transfer_define = 16
        elif self.transfer_fmt == "fp8":
            self.transfer_nbytes = 1
            self.transfer_define = 8
        else:
            raise ValueError(f"Bad transfer_fmt in checkpoint: {self.transfer_fmt}")

        self.table_size_entries = int(data["table_size_entries"])
        self.f_per_level = int(data["table_f_per_level"])
        self.table_elems = self.table_size_entries * self.f_per_level

        # occ meta
        self.occ = OccParams(
            G_occ=int(data["G_occ"]),
            box_min=np.asarray(data["box_min"], dtype=np.float32),
            box_max=np.asarray(data["box_max"], dtype=np.float32),
            rho_min=np.float32(data["rho_min"]),
            rho_max=np.float32(data["rho_max"]),
            dt_min=np.float32(data["dt_min"]),
            dt_max=np.float32(data["dt_max"]),
            t_near_clip=np.float32(data["t_near_clip"]),
            t_far_clip=np.float32(data["t_far_clip"]),
        ).finalize()

        self._infer_shapes()
        self.steps_trained = int(data.get("steps_trained", 0))

        # table
        self.state_i32["table"] = np.asarray(data["table_i"], dtype=np.int32).reshape(-1)

        # Active params list from checkpoint
        active_params = [str(x) for x in data["active_params"].tolist()]

        # Initialize ALL params (including placeholders) so setup_buffers can always run
        self._infer_shapes()

        # Fill all params with zeros first (placeholders and inactive)
        for wn in self.weight_names:
            self.state_i32[wn] = np.zeros(np.prod(self._shapes[wn]), dtype=np.int32)
            self.state_i32["m" + wn] = np.zeros(np.prod(self._shapes[wn]), dtype=np.int32)
            self.state_i32["v" + wn] = np.zeros(np.prod(self._shapes[wn]), dtype=np.int32)
        for bn in self.bias_names:
            self.state_i32[bn] = np.zeros(np.prod(self._shapes[bn]), dtype=np.int32)
            self.state_i32["m" + bn] = np.zeros(np.prod(self._shapes[bn]), dtype=np.int32)
            self.state_i32["v" + bn] = np.zeros(np.prod(self._shapes[bn]), dtype=np.int32)

        # Load only the active ones from file
        for name in active_params:
            self.state_i32[name] = np.asarray(data[name + "_i"], dtype=np.int32).reshape(-1)
            self.state_i32["m" + name] = np.asarray(data["m" + name + "_i"], dtype=np.int32).reshape(-1)
            self.state_i32["v" + name] = np.asarray(data["v" + name + "_i"], dtype=np.int32).reshape(-1)

        occ_u32 = np.asarray(data["occ_grid_in_u32"], dtype=np.uint32).reshape(-1)
        self.state_i32["occ_grid_in_u32"] = occ_u32.view(np.int32)
        occ_u32 = np.asarray(data["occ_grid_st_u32"], dtype=np.uint32).reshape(-1)
        self.state_i32["occ_grid_st_u32"] = occ_u32.view(np.int32)

        self.meta = self._build_meta()

        # If buffers already exist, refresh them (optional convenience)
        if self.buf:
            self._push_params_to_device()

    def _push_params_to_device(self):
        q = self.queue
        cl.enqueue_copy(q, self.buf["table"], self.state_i32["table"]).wait()
        for wn in self.weight_names:
            cl.enqueue_copy(q, self.buf[wn], self.state_i32[wn]).wait()
            cl.enqueue_copy(q, self.buf["m" + wn], self.state_i32["m" + wn]).wait()
            cl.enqueue_copy(q, self.buf["v" + wn], self.state_i32["v" + wn]).wait()
        for bn in self.bias_names:
            cl.enqueue_copy(q, self.buf[bn], self.state_i32[bn]).wait()
            cl.enqueue_copy(q, self.buf["m" + bn], self.state_i32["m" + bn]).wait()
            cl.enqueue_copy(q, self.buf["v" + bn], self.state_i32["v" + bn]).wait()

        occ_in_u32 = self.state_i32["occ_grid_in_u32"].view(np.uint32)
        cl.enqueue_copy(q, self.buf["occ_grid_in_u32"], occ_in_u32).wait()
        occ_st_u32 = self.state_i32["occ_grid_st_u32"].view(np.uint32)
        cl.enqueue_copy(q, self.buf["occ_grid_st_u32"], occ_st_u32).wait()

    # -------------------------
    # Kernel arg packing
    # -------------------------
    def _get_rng_u32(self) -> np.uint32:
        return np.uint32(self.rng.integers(0, 2**32, dtype=np.uint32))

    def _pack_ffnerf_args(
        self,
        io: RayBatch,
        lr_f: np.float32,
        do_train: int,
        sh_multiplier: np.float32,
        occ_update_strength: np.float32,
        rng_u32: np.uint32,
    ):
        args = []

        # ---- EXACT kernel arg order ----
        # Density weights
        args.append(self.buf["Wd_in"])
        args.append(self.buf["Wd_01"])
        args.append(self.buf["Wd_12"])
        args.append(self.buf["Wd_out"])
        # Density biases
        args.append(self.buf["bd_0"])
        args.append(self.buf["bd_1"])
        args.append(self.buf["bd_2"])
        args.append(self.buf["bd_out"])

        # Color weights
        args.append(self.buf["Wc_in"])
        args.append(self.buf["Wc_01"])
        args.append(self.buf["Wc_12"])
        args.append(self.buf["Wc_23"])
        args.append(self.buf["Wc_out"])
        # Color biases
        args.append(self.buf["bc_0"])
        args.append(self.buf["bc_1"])
        args.append(self.buf["bc_2"])
        args.append(self.buf["bc_3"])
        args.append(self.buf["bc_out"])

        # Hash-grid
        args.append(self.buf["table"])

        # Rays and targets
        args.append(io.rays_buf)
        args.append(io.targets_buf)

        # Deltas & Activations (kernel order)
        for lname in self.deltas_spec.keys():
            args.append(self.tmp[lname])
        for lname in self.activations_spec.keys():
            args.append(self.tmp[lname])

        # Output
        args.append(io.out_buf)

        # Scalars
        args.append(np.float32(lr_f))
        args.append(np.uint32(io.num_rays))
        args.append(np.uint32(io.samples_per_ray))
        args.append(np.int32(do_train))
        args.append(np.float32(sh_multiplier))

        # Occupancy params
        occ = self.occ
        box_min_v = cl.cltypes.make_float3(*occ.box_min.tolist())
        box_max_v = cl.cltypes.make_float3(*occ.box_max.tolist())

        args.append(self.buf["occ_grid_in_u32"])
        args.append(self.buf["occ_grid_st_u32"])
        args.append(box_min_v)
        args.append(box_max_v)
        args.append(np.int32(occ.G_occ))
        args.append(np.float32(occ.rho_min))
        args.append(np.float32(occ.rho_max))
        args.append(np.float32(occ.dt_min))
        args.append(np.float32(occ.dt_max))
        args.append(np.float32(occ.t_near_clip))
        args.append(np.float32(occ.t_far_clip))
        args.append(np.uint32(rng_u32))
        args.append(np.float32(occ_update_strength))

        return args


    def _pack_wu_args(self, B_total: int, L_POS: int, M: int, lr_t: np.float32, color_MLP_rel_lr: np.float32):
        args = []

        # ---- Canonical param order (ALL, incl placeholders) ----
        # weights then biases, both in the same "kernel grouping" order
        w_order = ["Wd_in","Wd_01","Wd_12","Wd_out","Wc_in","Wc_01","Wc_12","Wc_23","Wc_out"]
        b_order = ["bd_0","bd_1","bd_2","bd_out","bc_0","bc_1","bc_2","bc_3","bc_out"]

        for wn in w_order:
            args.append(self.buf[wn])
        for bn in b_order:
            args.append(self.buf[bn])

        # mW + vW (same order as weights)
        for wn in w_order:
            args.append(self.buf["m" + wn])
        for wn in w_order:
            args.append(self.buf["v" + wn])

        # mb + vb (same order as biases)
        for bn in b_order:
            args.append(self.buf["m" + bn])
        for bn in b_order:
            args.append(self.buf["v" + bn])

        # grads then acts (kernel order used by ff kernel)
        for lname in self.deltas_spec.keys():
            args.append(self.tmp[lname])
        for lname in self.activations_spec.keys():
            args.append(self.tmp[lname])

        args.append(np.int32(B_total))
        args.append(np.int32(L_POS))
        args.append(np.int32(M))
        args.append(np.float32(lr_t))
        args.append(np.float32(color_MLP_rel_lr))
        args.append(np.float32(self.ADAM_BETA1))
        args.append(np.float32(self.ADAM_BETA2))
        args.append(np.float32(self.ADAM_EPS))
        return args

    # -------------------------
    # Public API
    # -------------------------
    def _require_buffers_ready(self):
        if not self.io or not self.buf or not self.tmp:
            raise RuntimeError(
                "Buffers not set up. Call setup_buffers(batch_rays, samples_per_ray) first."
            )
        if not hasattr(self, "batch_rays") or not hasattr(self, "samples_per_ray"):
            raise RuntimeError("Internal batch params missing; call setup_buffers().")

    def _validate_io(self, rays_host: np.ndarray, targets_host: np.ndarray | None):
        rays_host = np.asarray(rays_host)
        if rays_host.dtype != np.dtype([("o", np.float32, 4), ("d", np.float32, 4)], align=True):
            # allow user to pass equivalent dtype; just enforce itemsize/fields
            if rays_host.dtype.itemsize != 32:
                raise ValueError(f"rays_host must have itemsize 32 (float4 o,d). Got {rays_host.dtype}.")
        if rays_host.ndim != 1:
            raise ValueError(f"rays_host must be shape (batch_rays,), got {rays_host.shape}")
        if rays_host.shape[0] != self.batch_rays:
            raise ValueError(
                f"rays_host length={rays_host.shape[0]} does not match setup batch_rays={self.batch_rays}. "
                "Call setup_buffers() again if you changed batch size."
            )

        if targets_host is not None:
            targets_host = np.asarray(targets_host, dtype=np.float32)
            if targets_host.shape != (self.batch_rays, 4):
                raise ValueError(
                    f"targets_host must be shape (batch_rays,4) = ({self.batch_rays},4), got {targets_host.shape}"
                )
        return rays_host, targets_host

    def _ensure_dummy_targets(self):
        if not hasattr(self, "_dummy_targets_host") or self._dummy_targets_host is None:
            self._dummy_targets_host = np.zeros((self.batch_rays, 4), dtype=np.float32)

        
    def step(
        self,
        io: RayBatch,
        lr_f: float,
        do_train: bool,
        return_output: bool,
        sh_multiplier: float = float(1.0),
        color_MLP_rel_lr: float = float(1.0),
        occ_update_strength: float | None = None,
        rng_u32: np.uint32 | None = None,
        wait: bool = False,
        global_size: tuple[int, ...] | None = None,
        local_size: tuple[int, ...] | None = None,
        wait_for: list | None = None,
    ) -> tuple[np.ndarray | None, cl.Event]:
        """
        Detached stepping:
        - io owns rays/targets/out buffers
        - caller is responsible for uploading rays/targets beforehand
        - Nerf owns params + scratch
        """

        # Ensure scratch matches this io shape
        if ((not hasattr(self, "num_rays")) or (self.num_rays != io.num_rays) or 
                (self.samples_per_ray != io.samples_per_ray)
            ) and do_train :
            # simplest policy: re-setup scratch whenever io shape changes
            # (params buffers are re-created too in this simple version)
            self.setup_buffers(io.num_rays, io.samples_per_ray)

        if occ_update_strength is None:
            occ_update_strength = float(self.occ.occ_update_strength0 if do_train else 0.0)

        if rng_u32 is None:
            rng_u32 = self._get_rng_u32()

        # If forward-only, you still need targets_buf to contain something valid
        if not do_train:
            # nothing required if caller already uploaded something;
            # but you can optionally ensure zeros exist:
            # (this uses host zeros then uploads)
            pass
        
        if do_train:
            self.steps_trained = int(self.steps_trained) + 1

        q = self.queue

        if global_size is None:
            global_size = (io.B_total,)
        if local_size is None:
            local_size = (io.samples_per_ray,)

        ff_args = self._pack_ffnerf_args(
            io=io,
            lr_f=np.float32(lr_f),
            do_train=int(1 if do_train else 0),
            sh_multiplier=np.float32(sh_multiplier),
            occ_update_strength=np.float32(occ_update_strength),
            rng_u32=np.uint32(rng_u32),
        )
        # Occupancy updates are deliberately disabled during the early training
        # warm-up (the caller passes occ_update_strength=0 until that point).
        # Do not run maintenance before occupancy learning is active: the packed
        # grid is still initialized near-full then, so the old suppress-mode
        # empty-cell rejection sampler has no useful targets and can become an
        # extremely long GPU kernel on Windows.
        do_occ_maintenance = (
            do_train
            and float(occ_update_strength) > 0.0
            and self.steps_trained % OCC_MAINTENANCE_PERIOD == 0
        )

        # Core training always uses the normal ray/compositing kernel.  The
        # periodic maintenance pass below merges/decays occ_grid_st -> occ_grid_in.
        # Keep the separately compiled suppress kernel available for later work,
        # but do not substitute it for a normal training batch here: that path was
        # previously unreachable and needs a dedicated, bounded empty-cell sampler
        # before it is safe to enable.
        ffnerf_kernel_to_run = self.ffnerf_knl
        ffnerf_kernel_to_run.set_args(*ff_args)

        ev_ff = cl.enqueue_nd_range_kernel(
            queue=q,
            kernel=ffnerf_kernel_to_run,
            global_work_size=global_size,
            local_work_size=local_size,
            wait_for=wait_for,
        )
        # ev_ff.wait()
        ev_last = ev_ff

        if do_train:
            #print("     [DI]: doing train")
            # ----- occupancy grid (more like density grid) decay if applicable -----
            if do_occ_maintenance:
                n_words = int(self.occ.G_occ) ** 3
                decay = DECAY
                d = 1 - occ_update_strength * (1 - decay)
                decay_q24 = int(round(d * (1 << 24)))
                decay_q24 = max(0, min(decay_q24, (1 << 24)))

                local = S_MAX
                global_ = ((n_words + local - 1) // local) * local

                self.decay_knl.set_args(
                    self.buf["occ_grid_in_u32"],
                    self.buf["occ_grid_st_u32"],
                    np.uint32(n_words),
                    np.uint32(decay_q24),
                    np.uint32(rng_u32),
                )
                evt_decay = cl.enqueue_nd_range_kernel(
                    queue=q,
                    kernel=self.decay_knl,
                    global_work_size=(global_,),
                    local_work_size=(local,),
                    wait_for=[ev_ff],
                )
                ev_last = evt_decay

            # ----- Weight updates with AdaM ----- //TODO: later try Muon j4f
            t_adam = float(self.steps_trained)
            b1t = self.ADAM_BETA1 ** np.float32(t_adam)
            b2t = self.ADAM_BETA2 ** np.float32(t_adam)
            lr_t = np.float32(np.float32(lr_f) * np.sqrt(1.0 - b2t) / (1.0 - b1t))

            wu_args = self._pack_wu_args(
                B_total=int(io.B_total),
                L_POS=int(self.L_POS),
                M=int(self.M),
                lr_t=lr_t,
                color_MLP_rel_lr=color_MLP_rel_lr,
            )
            self.wu_knl.set_args(*wu_args)
            # ---- Multi-WG tiling launch for weight update ----
            # These must match compile-time values in weight_update.cl:
            wg_x, wg_y  = int(self.wu_wg_x),  int(self.wu_wg_y)
            reg_x, reg_y = int(self.wu_reg_x), int(self.wu_reg_y)

            INT  = wg_x * reg_x
            OUTT = wg_y * reg_y

            # Cover the *largest* layer dimensions used inside update_ffnerf_oneWG.
            # OUT dimension can be M (most layers), D_OUT (16), or C_OUT (3) => max is M.
            # IN dimension can be M, C_IN (=D_OUT+L_DIR), or L_POS => max is max(M, C_IN, L_POS).
            max_out = int(self.M)
            max_in  = int(max(self.M, self.C_IN, self.L_POS))

            num_groups_x = (max_in  + INT  - 1) // INT
            num_groups_y = (max_out + OUTT - 1) // OUTT

            global_wu = (num_groups_x * wg_x, num_groups_y * wg_y)
            local_wu  = (wg_x, wg_y)

            ev_wu = cl.enqueue_nd_range_kernel(
                queue=q,
                kernel=self.wu_knl,
                global_work_size=global_wu,
                local_work_size=local_wu,
                wait_for=[ev_last],
            )
            ev_last = ev_wu
        
        if self.steps_trained < 4 and (do_train if True else False): #TODO rething this staetement  #((self.steps_trained+1) % 32) == 0:
            out, ev_dwn = io.download_out(wait=True, wait_for=[ev_last])
            debug_vals = out[0:4,3] # x,y,z,w = r,g,b + debug val
            # print("ev_ff command_type:", ev_ff.get_info(cl.event_info.COMMAND_TYPE))
            print(f"[I]: ev_ff   {'do_train' if do_train else '        '}  [ms]:", event_ms(ev_ff))
            print("          debug vals -", list(debug_vals))
            if "evt_decay" in locals():
                print("[I]: evt_decay         [ms]:", event_ms(evt_decay))
            if "ev_wu" in locals():
                print("[I]: ev_wu             [ms]:", event_ms(ev_wu))

        out = None
        if return_output:
            out, ev_dwn = io.download_out(wait=False, wait_for=[ev_last])
            # download_out enqueues copy; wait below handles it if requested
            # but download_out(wait=False) returns the array immediately; the copy may still be in-flight
            # so if wait=True, we force completion here:
            if wait:
                ev_dwn.wait()
            return out, ev_dwn

        if wait:
            ev_last.wait()
        return None, ev_last


    # -------------------------
    # Convenience: sync params without saving
    # -------------------------
    def sync_params_from_device(self): 
        """
        Pull device buffers into self.state_i32 (table, weights, biases, adam, occ).
        Useful for debugging/inspection without writing a file.
        """
        self._require_buffers_ready()
        q = self.queue

        def pull_i32(name: str):
            host = np.empty_like(self.state_i32[name], dtype=np.int32)
            cl.enqueue_copy(q, host, self.buf[name]).wait()
            self.state_i32[name] = host

        pull_i32("table")
        for wn in self.weight_names:
            pull_i32(wn); pull_i32("m" + wn); pull_i32("v" + wn)
        for bn in self.bias_names:
            pull_i32(bn); pull_i32("m" + bn); pull_i32("v" + bn)

        occ_in_host = np.empty((int(self.occ.G_occ) ** 3,), dtype=np.uint32)
        cl.enqueue_copy(q, occ_in_host, self.buf["occ_grid_in_u32"]).wait()
        self.state_i32["occ_grid_in_u32"] = occ_in_host.view(np.int32)
        occ_st_host = np.empty((int(self.occ.G_occ) ** 3,), dtype=np.uint32)
        cl.enqueue_copy(q, occ_st_host, self.buf["occ_grid_st_u32"]).wait()
        self.state_i32["occ_grid_st_u32"] = occ_st_host.view(np.int32)

