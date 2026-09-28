// ffnerf_OG.cl — toy fused NeRF-like step with back-to-front compositing
// + occupancy-grid guided sampling (DDA) + atomic nibble exchange updates.
//
// OpenCL C 1.2 compatible.
//
// Mapping:
//   group_id(0)  = ray index
//   local_id(0)  = sample index along ray
//   local_size   = samples_per_ray (<= S_MAX)

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics  : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable

// --- instant-ngp NeRF MLP dims ---
#ifndef NDMAX 
#define NDMAX 3
#endif
#ifndef NCMAX 
#define NCMAX 4
#endif

#if (NDMAX != 3 || NCMAX != 4)
#error NDMAX is 3, NCMAX is 4 in kernels
#endif

#ifndef ND
#define ND 2
#endif
#ifndef NC
#define NC 2
#endif
#if (ND < 1) || (ND > 3)
#error ND must be 1..3
#endif
#if (NC < 1) || (NC > 4)
#error NC must be 1..4
#endif

#define L_DIR   16              // SH(dir) (deg 0..3)
#define D_OUT   16              // density MLP output (log_sigma + 15 features)
#define C_IN    (D_OUT + L_DIR) // 32
#define C_OUT   3               // RGB
#define M       64              // width for ALL hidden layers

#define S_MAX 64//256//128   // max samples per ray that this kernel supports

// Toggle: keep ReLU masks in regs (recommended to try)
// #ifndef USE_RELU_MASK
// #define USE_RELU_MASK 1
// #endif


#define RGB_SIGMOID_SCALE 0.5f

#define LAMBDA_FOCUS 0.0f   //1.0e-3f   // [1.2 to 1.8]     brings forward the samples eg [2,3,2,5,4,3,5,16,8,6,16,8,4,6,] 
                            // to [0.02, 0.04, 0.02, 0.07, 0.05, 0.03, 0.06, 0.21, 0.09, 0.06, 0.16, 0.06, 0.02, 0.04]  at 1.0e-2
#define DENSITY_PENALTY_LMD 5e-3f // [1e-6 to something small]      subtracted from weight before its processed
#define GAMMA_W 1.3f        // [0.8 to 1.8]     increases contrast with w[s]^gamma/sum(w^gamma)
#define UNIFORM_PART 0.3f  // [0.1 to 0.5]     the ratio of unifrom PDF in the total weighting PDF

#define SIGMA_LR_RATE 1.0f // 0.2f
#define SxDT_CAP_FOR_ALPHA -5.0f //old: -18.0f
#define HASH_LR_RATE 0.5f //0.05f  // idk, keep .<1 so network can adapt, and its not just noise
#define HASH_W_LVL_SCALE_EXPONENT 0.1f //1.2f // try 1.4f //-0.75f //0.05f // TODO try out [0.0f,1.2f, etc] // lr_lvl_rel = 2^(-lvl * lvlsc_exp)


#ifndef SURPRESS_SAMPLES
#define SURPRESS_SAMPLES 0
#endif

// Strength of sigma->0 suppress pass (tune: 1e-3 .. 1e-1 typically)
#ifndef SUPPRESS_SIGMA_LAMBDA
#define SUPPRESS_SIGMA_LAMBDA 1.0e-2f
#endif

// Max tries for rejection sampling occ4==0
#ifndef SUPPRESS_MAX_TRIES
#define SUPPRESS_MAX_TRIES 64
#endif



// ============================================================
// Transfer format shim for inter-kernel comm buffers (a*, d*)
// ============================================================
//
// Select exactly one at build time (host passes -DTRANSFER_FMT=...):
//   32 = fp32 (float)
//   16 = fp16 (half)
//    8 = fp8  (custom e4m3-ish, saturating)
//
// If not provided, default to fp32.
#ifndef TRANSFER_FMT
#define TRANSFER_FMT 32
#endif

#if (TRANSFER_FMT == 16)
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

// -------------------- FP8 (custom) helpers --------------------
// A small, fast-ish saturating e4m3-like pack/unpack.
// Range target: approximately [-15.5, +15.5] (max finite ~15.5).
// Notes:
// - This is NOT IEEE; it’s a pragmatic “minifloat-like” encoding.
// - We flush tiny values to 0 (no subnormals).
// - Good enough for bandwidth experiments, not for “numerical fidelity”.

#if (TRANSFER_FMT == 8)
    typedef uchar TRANSFER_T;
    // Clamp to the representable finite range for this fp8: ~[-15.5, +15.5]
    static inline float fp8_sat(float x) { if (!isfinite(x)) return copysign(15.5f, x); return clamp(x, -15.5f, 15.5f); }
    // Pack float -> fp8 bits: [7]=sign, [6:3]=exp (bias 7), [2:0]=mant (3 bits)    exp==0 => zero (we flush)
    static inline uchar pack_fp8_e4m3(float x_in) {
        float x = fp8_sat(x_in);
        uint s = (x < 0.0f) ? 0x80u : 0u; float ax = fabs(x);
        if (ax < 1.0e-8f) return (uchar)s; // flush-to-zero
        int e; float m = frexp(ax, &e);  // frexp: ax = m * 2^e, m in [0.5,1)
        // Convert to normalized form: m2 in [1,2), e2 is unbiased exponent
        float m2 = m * 2.0f; int e2 = e - 1;
        const int bias = 7;  int ef = e2 + bias;
        // flush underflow (no subnormals here)  // clamp exponent to max normal (14), reserve 15 as inf/nan if desired
        if (ef <= 0) return (uchar)s;
        if (ef > 14) { ef = 10; return (uchar)(s | ((uint)ef << 3) | 7u); }
        // mantissa: 3 bits from fractional part (m2 - 1) in [0,1)        // handle rounding carry        // If exponent bumped out of range, saturate to max finite
        float frac = m2 - 1.0f; int mant = (int)rint(frac * 8.0f); // 0..8
        if (mant >= 8) { mant = 0; ef += 1; }
        if (ef > 14) { ef = 10; mant = 7; }
        return (uchar)(s | ((uint)ef << 3) | ((uint)mant & 7u));
    }
    static inline float unpack_fp8_e4m3(uchar b) {
        uint s  = (b & 0x80u); uint ef = (b >> 3) & 0x0Fu; uint mf = (b & 0x07u);
        if (ef == 0u) return s ? -0.0f : 0.0f; // zero (we used flush-to-zero)
        const int bias = 7; int e2 = (int)ef - bias;
        float mant = (float)mf * (1.0f / 8.0f);
        float v = ldexp(1.0f + mant, e2); // (1+mant)*2^e2
        return s ? -v : v;
    }
    static inline float tload(const __global TRANSFER_T* p, int idx) { return unpack_fp8_e4m3(p[idx]); }
    static inline void tstore(__global TRANSFER_T* p, int idx, float x) { p[idx] = pack_fp8_e4m3(x); }

#elif (TRANSFER_FMT == 16)
    typedef half TRANSFER_T;
    // Saturate to half finite range to avoid infs on store.
    static inline float fp16_sat(float x) { if (!isfinite(x)) return copysign(65504.0f, x); return clamp(x, -65504.0f, 65504.0f); }
    static inline float tload(const __global TRANSFER_T* p, int idx) { return convert_float(p[idx]); }
    static inline void tstore(__global TRANSFER_T* p, int idx, float x) { p[idx] = convert_half_rte(fp16_sat(x)); }

#else // TRANSFER_FMT == 32
    typedef float TRANSFER_T;
    static inline float tload(const __global TRANSFER_T* p, int idx) { return p[idx]; }
    static inline void  tstore(__global TRANSFER_T* p, int idx, float x) { p[idx] = x; }

#endif




// -------------------- Hash table settings -----------
// ---- hash config (per-level partitioned tables) ----
#define RESOLUTION_LVL0 16.0f
#define RESOLUTION_LVLMAX 1024.0f // 2048.0f;

#define N_LEVELS 16
#ifndef F_PER_LEVEL
#define F_PER_LEVEL 4 // tuneable (2 -> 4 -> 8, etc.)
#endif
#define L_POS   (F_PER_LEVEL*N_LEVELS)              // hashgrid encoding

// Example A sizing (~~ 1.0M entries total):
//   lv 0..3  : 2^14    lv 4..7  : 2^15
//   lv 8..11 : 2^16    lv 12..15: 2^17
// Total entries = 983,040.
//
// Layout: levels are concatenated in one big table[] buffer. Each level uses its own mask and base offset (in ENTRIES).
// Host must allocate: TABLE_TOTAL_ENTRIES * F_PER_LEVEL ints.
#define TABLE_TOTAL_ENTRIES 2260992u

// Per-level log2 table sizes:
//   lv 0..3   : 2^13
//   lv 4..7   : 2^15
//   lv 8..11  : 2^17
//   lv 12..13 : 2^18
//   lv 14..15 : 2^19
__constant uint LEVEL_LOG2[N_LEVELS] = {
    13u,13u,13u,13u,
    15u,15u,15u,15u,
    17u,17u,17u,17u,
    18u,18u,19u,19u
};

__constant uint LEVEL_OFFSET_ENTRIES[N_LEVELS] = {
    0u, 8192u, 16384u, 24576u, // 0..3 (2^13 each = 8192)
    32768u, 65536u, 98304u, 131072u, // 4..7 (2^15 each = 32768)
    163840u, 294912u, 425984u, 557056u, // 8..11 (2^17 each = 131072)
    688128u, 950272u, // 12..13 (2^18 each = 262144)
    1212416u, 1736704u // 14..15 (2^19 each = 524288)
};

static inline uint level_mask(int lvl) {
    return (1u << LEVEL_LOG2[lvl]) - 1u;
}
// Convert (level-local entry index) -> flat FEATURE index in table[].
static inline int table_feat_index(int lvl, uint entry_in_level, uint f) {
    // base entry in global (concatenated) space
    uint base_entry = LEVEL_OFFSET_ENTRIES[lvl] + entry_in_level;
    // convert to feature index
    return (int)(base_entry * (uint)F_PER_LEVEL + f);
}


static inline float relu(float x) { return x > 0.0f ? x : 0.0f; }
static inline float relu_prime_from_pre(float z) { return z > 0.0f ? 1.0f : 0.0f; }
static inline float relu_prime_from_act(float a) { return a > 0.0f ? 1.0f : 0.0f; }
static inline float clamp01(float x) { return fmin(1.0f, fmax(0.0f, x)); }
static inline float softplus(float x) {
    if (x > 20.0f)  return x;                 // log(1+exp(x)) ~ x
    if (x < -20.0f) return exp(x);           // log(1+exp(x)) ~ exp(x)
    return log1p(exp(x));
}
static inline float softplus_prime_from_pre(float x) {
    if (x >= 0.0f)  return 1.0f / (1.0f + exp(-x));
    else            return exp(x) / (1.0f + exp(x));
}
// static inline float sigmoid(float x) {
//     if (x >= 0.0f)  return 1.0f / (1.0f + exp(-x));
//     else            return exp(x) / (1.0f + exp(x));
// }
// static inline float sigmoid_prime_from_sigmoid(float s) { 
//     return s * (1.0f - s); // use already computed s = sigmoid(x)
// }
static inline float sigmoid_soft(float x) {
    x *= RGB_SIGMOID_SCALE;
    x = clamp(x, -18.0f, 18.0f);
    if (x >= 0.0f)  return 1.0f / (1.0f + exp(-x));
    else            return exp(x) / (1.0f + exp(x));
}
// derivative w.r.t. ORIGINAL x (pre-scale), not the scaled one
static inline float sigmoid_soft_prime_from_sigmoid(float s_soft) {
    // ds/dx = ds/d(x*k) * d(x*k)/dx = (s*(1-s)) * k
    return (s_soft * (1.0f - s_soft)) * RGB_SIGMOID_SCALE;
}
// Robust loss: Charbonnier (smooth L1)
// phi(x) = sqrt(x^2 + eps^2) // dphi/dx = x / sqrt(x^2 + eps^2)
#define RGB_CHARB_EPS 1.0e-3f  // try 1e-3 .. 1e-2
static inline float charbonnier(float x) {
    return sqrt(x*x + RGB_CHARB_EPS*RGB_CHARB_EPS);
}
static inline float charbonnier_prime(float x) {
    return x * rsqrt(x*x + RGB_CHARB_EPS*RGB_CHARB_EPS);
}





static inline float i32_to_f(int u) { return as_float(u); }
static inline int   f_to_i32(float f) { return as_int(as_uint(f)); }

// -------------------------
// Float atomic add via CAS
// -------------------------
static inline void atomic_add_f32_global_i32(volatile __global int *addr, float val){
    int old_i = *addr;
    for (;;) {
        float old_f = as_float(old_i);
        float new_f = old_f + val;
        int new_i = as_int(as_uint(new_f));
        int prev_i = atomic_cmpxchg(addr, old_i, new_i);
        if (prev_i == old_i) break;
        old_i = prev_i;
    }
}

static inline void atomic_add_f32_local_i32(volatile __local int *addr, float val)
{
    int old_i = *addr;
    for (;;) {
        float old_f = as_float(old_i);
        float new_f = old_f + val;
        int new_i = as_int(as_uint(new_f));
        int prev_i = atomic_cmpxchg(addr, old_i, new_i);
        if (prev_i == old_i) break;
        old_i = prev_i;
    }
}

// -------------------------
// RNG (xorshift32 + hash)
// -------------------------
static inline uint hash_u32(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static inline float rand01(uint* state)
{
    uint x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return (float)(x & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

// -------------------------
// Ray + AABB intersect
// -------------------------
typedef struct Ray
{
    float4 o;   // origin
    float4 d;   // direction
} Ray;

static inline int ray_aabb_intersect(float3 o, float3 d, float3 bmin, float3 bmax, float* t0, float* t1)
{
    float3 invd = (float3)(
        (d.x != 0.0f) ? (1.0f / d.x) : INFINITY,
        (d.y != 0.0f) ? (1.0f / d.y) : INFINITY,
        (d.z != 0.0f) ? (1.0f / d.z) : INFINITY
    );

    float3 tbot = (bmin - o) * invd;
    float3 ttop = (bmax - o) * invd;

    float3 tmin3 = fmin(tbot, ttop);
    float3 tmax3 = fmax(tbot, ttop);

    float tmin = fmax(tmin3.x, fmax(tmin3.y, tmin3.z));
    float tmax = fmin(tmax3.x, fmin(tmax3.y, tmax3.z));

    if (tmax < tmin) return 0;
    *t0 = tmin;
    *t1 = tmax;
    return 1;
}

// -------------------------
// -------------------------
// Occupancy grid nibble update (packed uint holds 8 nibbles)
// Nibble encoding: occ4 in [0..15] represents occ = (occ4 + 1) / 16.
// So 0 => 1/16 (minimum), 15 => 16/16 (maximum).
//
// Update rule in float:
//   old_f = (old4+1)/16
//   decayed = old_f * decay
//   pred_f = pred_q/16  (where pred_q is 1..16)
//   new_f = max(decayed, pred_f)
// Quantize back to q in [1..16], store new4 = q-1.
// -------------------------
static inline void occ_nibble_atomic_decay_max_u16_stoch(
    __global uint* occ_grid_rw,
    int cell_index,
    uint subcell,     // 0..7
    uint pred_q,      // 1..16
    float decay,      // 0..1
    float occ_update_strength,
    uint rng_seed     // caller-provided seed (e.g. hash_u32(ray_id ^ (s_id<<16) ^ cell_index))
){      
    // NOTE: with the logic "if p>o -> +1 else o*delta" at o~~3 d=0.8 is good to keep 1:1 noisy there
    // other logic tries to dampen the effect
    // WARNING! with occl_update_strength we need proper o*d not just a single step down
    const uint shift = 4u * subcell;
    const uint mask  = 0xFu << shift;

    volatile __global int* addr = (volatile __global int*)(&occ_grid_rw[cell_index]);
    int old_i = *addr;

    // clamp inputs (cheap + avoids weirdness)
    float d = clamp((1.0f - occ_update_strength * (1.0f - decay)), 0.0f, 1.0f); // step down probability decreased by update strength

    for (;;)
    {
        uint old_u = as_uint(old_i);
        uint old4  = (old_u >> shift) & 0xFu;   // 0..15
        uint q     = old4 + 1u;                 // 1..16

        uint pq = (uint)(
                (float)pred_q * occ_update_strength + (float)q * (1.0f-occ_update_strength)
            ); // occ_update_strength holds back pred from updating
        if (pq < 1u) pq = 1u;  if (pq > 16u) pq = 16u;

        // --- stochastic 1-step decay in q-space ---
        uint q_after = q;
        if (q > 1u)
        {
            // desired decayed value in [0..16]  q*d * os + q * (1-os)
            float qd = (float)q * d;
            // first step in the range of [floor,floor+1]
            q_after = (uint)floor(qd)+1u; 
            // then step down with probability p_dec = q - qd (in (0,1))
            float p_dec = ((float)q_after - qd);

            // 24-bit uniform integer
            uint r = hash_u32(rng_seed ^ (uint)cell_index * 9781u ^ subcell * 6271u ^ old_u);
            uint r24 = r & 0x00FFFFFFu;
            uint thresh = (uint)(p_dec * 16777216.0f); // 2^24

            if (r24 < thresh) q_after -= 1u;
        }

        // combine with evidence (max)
        uint q_new = (q+1u <= pq) ? q + 1 : ((pq <= q_after) ? q_after : q); //(q_after > pq) ? q_after : q + 1;// pq;

        // store as nibble
        uint new4 = q_new - 1u; // 0..15
        uint new_u = (old_u & ~mask) | ((new4 & 0xFu) << shift);

        int new_i = as_int(new_u);
        int prev_i = atomic_cmpxchg(addr, old_i, new_i);
        if (prev_i == old_i) break;
        old_i = prev_i;
    }
}

// ============================
// occ helpers (above kernel)
// ============================

#define CHG_WORDS ((S_MAX + 31) / 32)

// local bitflag helpers
static inline void chg_clear_all(__local uint* f) {
    for (uint i = 0; i < (uint)CHG_WORDS; ++i) f[i] = 0u;
}
static inline void chg_set(__local uint* f, uint idx) {
    f[idx >> 5] |= (1u << (idx & 31u));
}
static inline uint chg_get(__local const uint* f, uint idx) {
    return (f[idx >> 5] >> (idx & 31u)) & 1u;
}

static inline uint ctz_u32(uint x) { // Works for x != 0
    uint lsb = x & (0u - x);
    return popcount(lsb - 1u);
}

// find next set bit after 'idx' (exclusive). returns WG if none.
static inline uint chg_find_next(__local const uint* f, uint idx, uint WG)
{
    uint i = idx + 1u;
    if (i >= WG) return WG;

    uint w = i >> 5;
    uint b = i & 31u;

    // mask bits >= b
    uint mask = f[w] & (~0u << b);

    while (1) {
        if (mask) {
            // mask!=0 => safe. NOTE: ctz built in did not compile. clz exists, ctz easy to implement.
            uint off = (uint)ctz_u32(mask);
            uint res = (w << 5) + off;
            return (res < WG) ? res : WG;
        }
        w++;
        if ((w << 5) >= WG) break;
        mask = f[w];
    }
    return WG;
}

// pack (cell,sub) into one key for change detection in lane0
static inline uint pack_cellsub(int cell, uint sub) {
    // cell is non-negative; sub in [0..7]
    return (((uint)cell) << 3) | (sub & 7u);
}

// read q in [1..16] from packed nibble grid
static inline uint occ_nibble_read_q16(__global const uint* occ_grid, int cell_index, uint subcell)
{
    uint packed = occ_grid[cell_index];
    uint occ4 = (packed >> (4u * subcell)) & 0xFu;
    return occ4 + 1u; // 1..16
}

// atomic max on one nibble (q in 1..16)
static inline void occ_nibble_atomic_max_u16(
    __global uint* occ_grid_rw,
    int cell_index,
    uint subcell,
    uint proposed_q   // 1..16
){
    const uint shift = 4u * subcell;
    const uint mask  = 0xFu << shift;

    volatile __global int* addr = (volatile __global int*)(&occ_grid_rw[cell_index]);
    int old_i = *addr;

    for (;;) {
        uint old_u = as_uint(old_i);
        uint old4  = (old_u >> shift) & 0xFu;  // 0..15
        uint old_q = old4 + 1u;                // 1..16
        if (proposed_q <= old_q) return;

        uint new4 = (proposed_q - 1u) & 0xFu;
        uint new_u = (old_u & ~mask) | (new4 << shift);

        int new_i = as_int(new_u);
        int prev_i = atomic_cmpxchg(addr, old_i, new_i);
        if (prev_i == old_i) return;
        old_i = prev_i;
    }
}

static inline float3 random_point_in_empty_nibble(
    uint* rng,
    __global const uint* occ_grid_in,
    float3 box_min,
    float3 box_max,
    int G,
    int* out_cell,      // optional, can be 0
    uint* out_sub       // optional, can be 0
){
    float3 box_size  = box_max - box_min;
    float3 cell_size = box_size * (1.0f / (float)G);

    int cell = 0;
    uint sub = 0u;

    // rejection sample: pick random cell+sub until occ4==0
    // bounded to avoid infinite loops if grid has no empties
    for (int it = 0; it < SUPPRESS_MAX_TRIES; ++it) {
        // cell in [0, G^3)
        uint ucell = (uint)((float)(G*G*G) * rand01(rng));
        if (ucell >= (uint)(G*G*G)) ucell = (uint)(G*G*G - 1);
        cell = (int)ucell;

        sub = (uint)(8.0f * rand01(rng)); // 0..7
        if (sub > 7u) sub = 7u;

        uint packed = occ_grid_in[cell];
        uint occ4 = (packed >> (4u * sub)) & 0xFu;
        if (occ4 == 0u) break;
    }

    // decode cell -> ix,iy,iz
    int ix = cell % G;
    int iy = (cell / G) % G;
    int iz = (cell / (G*G));

    // choose random point inside that subcell octant (each axis halfc-cell)
    float3 cell_min = box_min + (float3)((float)ix, (float)iy, (float)iz) * cell_size;

    uint sx = (sub & 1u);
    uint sy = (sub & 2u) >> 1;
    uint sz = (sub & 4u) >> 2;

    float3 halfc = cell_size * 0.5f;

    float rx = rand01(rng);
    float ry = rand01(rng);
    float rz = rand01(rng);

    float3 p0 = cell_min + halfc * (float3)((float)sx, (float)sy, (float)sz);
    float3 x  = p0 + (float3)(rx, ry, rz) * halfc;

    if (out_cell) *out_cell = cell;
    if (out_sub)  *out_sub  = sub;
    return x;
}


#define SPHERICAL_H_DG 3 // 0, 1, 2 or 3
// if (SPHERICAL_H_DG+1)^2 != L_DIR it will result in garbage. later
static inline void directional_encode(float3 d, float* sh){
    float x = d.x; 
    float y = d.y; 
    float z = d.z; 
    // Input: normalized direction (x,y,z)
    sh[0]  = 0.28209479177387814f;
    if (SPHERICAL_H_DG == 0) return;

    sh[1]  = 0.4886025119029199f  * y;
    sh[2]  = 0.4886025119029199f  * z;
    sh[3]  = 0.4886025119029199f  * x;
    if (SPHERICAL_H_DG == 1) return;

    sh[4]  = 1.0925484305920792f  * x * y;
    sh[5]  = 1.0925484305920792f  * y * z;
    sh[6]  = 0.31539156525252005f * (3.0f*z*z - 1.0f);
    sh[7]  = 1.0925484305920792f  * x * z;
    sh[8]  = 0.5462742152960396f  * (x*x - y*y);
    if (SPHERICAL_H_DG == 2) return;

    sh[9]  = 0.5900435899266435f  * y * (3.0f*x*x - y*y);
    sh[10] = 2.890611442640554f   * x * y * z;
    sh[11] = 0.4570457994644658f  * y * (5.0f*z*z - 1.0f);
    sh[12] = 0.3731763325901154f  * (5.0f*z*z*z - 3.0f*z);
    sh[13] = 0.4570457994644658f  * x * (5.0f*z*z - 1.0f);
    sh[14] = 1.445305721320277f   * z * (x*x - y*y);
    sh[15] = 0.5900435899266435f  * x * (x*x - 3.0f*y*y);
    if (SPHERICAL_H_DG == 3) return;
}


// -----------------------------
// Forward scan helpers
// -----------------------------
// Workgroup-parallel compositing + loss + per-sample grads
// Requirements:
//   - WG = get_local_size(0) is power-of-two (32/64/128/256)
//   - WG == samples_per_ray
//   - WG <= S_MAX
// ------------------------------------------------------------
// ============================================================
// WG-parallel sampler helpers (float scan/reduce + CDF lookup)
// Assumes WG is power-of-two (same as your compositing scan).
// ============================================================
// Blelloch exclusive scan (in-place) for float
static inline void scan_excl_f_inplace(__local float* a, uint lid, uint n) {
    for (uint stride = 1u; stride < n; stride <<= 1u) {
        uint idx = ((lid + 1u) * (stride << 1u)) - 1u;
        if (idx < n) a[idx] += a[idx - stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0u) a[n - 1u] = 0.0f; // identity for exclusive scan
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = (n >> 1u); stride >= 1u; stride >>= 1u) {
        uint idx = ((lid + 1u) * (stride << 1u)) - 1u;
        if (idx < n) {
            float t = a[idx - stride];
            a[idx - stride] = a[idx];
            a[idx] += t;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (stride == 1u) break;
    }
}
// In-place reduction sum. Returns sum in all lanes (after it finishes).
static inline float reduce_sum_f_inplace(__local float* a, uint lid, uint n) {
    for (uint stride = (n >> 1u); stride >= 1u; stride >>= 1u) {
        if (lid < stride) a[lid] += a[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (stride == 1u) break;
    }
    return a[0];
}
// lower_bound on inclusive CDF (monotone nondecreasing), returns first idx with cdf[idx] >= u
static inline uint lower_bound_cdf_local(__local const float* cdf, uint n, float u) {
    uint lo = 0u, hi = n - 1u;
    while (lo < hi) {
        uint mid = (lo + hi) >> 1;
        if (cdf[mid] >= u) hi = mid;
        else lo = mid + 1u;
    }
    return lo;
}
// Atomic bit-set for chg_flag (needed because multiple lanes set bits concurrently)
static inline void chg_set_atomic(__local uint* f, uint idx) {
    const uint w = idx >> 5;
    const uint b = idx & 31u;
    // local atomics enabled via your pragma extensions
    atomic_or((volatile __local int*)(&f[w]), (int)(1u << b));
}

static inline float safe_pow_pos(float x, float gamma){
    // x should be >= 0. For x==0 -> 0.
    // Use powr for non-negative base.
    x = fmax(x, 0.0f);
    if (x <= 0.0f) return 0.0f;
    return powr(x, gamma);
}
static inline float focus_gain_unitprefix(float pi, float lambda_focus){
    // pi is in [0,1] ideally; clamp for safety
    pi = clamp(pi, 0.0f, 1.0f);
    // exp(-lambda*pi) is dimensionless + scale invariant
    return exp(-lambda_focus * pi);
}
static inline int stoch_round(float x, uint* rng){
    int n = (int)floor(x);
    float frac = x - (float)n;
    float u = rand01(rng);
    return n + ((u < frac) ? 1 : 0);
}


typedef struct { float a; float3 v; } Seg;
inline Seg seg_combine(Seg L, Seg R) { Seg O; O.a = L.a * R.a; O.v = L.v + L.a * R.v; return O; }

typedef struct { float a; float b; } Aff;
// combine_rev(L,R) = R ° L
inline Aff aff_combine_rev(Aff L, Aff R) { Aff O; O.a = R.a * L.a; O.b = R.a * L.b + R.b; return O; }

inline void scan_excl_seg(__local Seg* segs, uint lid, uint WG, Seg* total_out) {
    for (uint stride = 1; stride < WG; stride <<= 1) {
        uint idx = ((lid + 1) * (stride << 1)) - 1;
        if (idx < WG) segs[idx] = seg_combine(segs[idx - stride], segs[idx]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) *total_out = segs[WG - 1];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) { segs[WG - 1].a = 1.0f; segs[WG - 1].v = (float3)(0.0f); }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = WG >> 1; stride >= 1; stride >>= 1) {
        uint idx = ((lid + 1) * (stride << 1)) - 1;
        if (idx < WG) {
            Seg t = segs[idx - stride];
            segs[idx - stride] = segs[idx];
            segs[idx] = seg_combine(t, segs[idx]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (stride == 1) break;
    }
}

inline void scan_excl_aff_rev(__local Aff* affs, uint lid, uint WG, Aff* total_out) {
    for (uint stride = 1; stride < WG; stride <<= 1) {
        uint idx = ((lid + 1) * (stride << 1)) - 1;
        if (idx < WG) affs[idx] = aff_combine_rev(affs[idx - stride], affs[idx]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) *total_out = affs[WG - 1];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) { affs[WG - 1].a = 1.0f; affs[WG - 1].b = 0.0f; } // identity
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = WG >> 1; stride >= 1; stride >>= 1) {
        uint idx = ((lid + 1) * (stride << 1)) - 1;
        if (idx < WG) {
            Aff t = affs[idx - stride];
            affs[idx - stride] = affs[idx];
            affs[idx] = aff_combine_rev(t, affs[idx]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (stride == 1) break;
    }
}


// ------------------------------------
// ReLU mask helpers (M <= 64)
// ------------------------------------
static inline ulong relu_mask_setbit(ulong m, int j, int on) {
    // on is 0/1
    return on ? (m | (1ul << (uint)j)) : m;
}
static inline float relu_mask_apply(ulong m, int j, float x) {
    // multiply by 0 or 1 based on bit
    return ((m >> (uint)j) & 1ul) ? x : 0.0f;
}

// ------------------------------------
// FC + ReLU: out[M] = relu(W[M,in]*in + b[M])
// also builds a 64-bit mask of which neurons are active
// ------------------------------------
static inline void fc_relu_in(
    float* outM, //outM[M]
    ulong* mask_out,
    const float* in,
    int in_dim,
    __global const int* W,   // [M * in_dim]
    __global const int* b    // [M] or null
){
    ulong m = 0ul;
    for (int i = 0; i < M; ++i) {
        float sum = b ? i32_to_f(b[i]) : 0.0f;
        // If in_dim is compile-time constant, compiler will unroll well.
        for (int j = 0; j < in_dim; ++j) sum += i32_to_f(W[i*in_dim + j]) * in[j];
        float a = relu(sum);
        outM[i] = a;
        m = relu_mask_setbit(m, i, (a > 0.0f));
    }
    *mask_out = m;
}

// FC + ReLU: out[M] = relu(W[M,M]*inM + b[M])
static inline void fc_relu_mm(
    float* outM, //outM[M]
    ulong* mask_out,
    const float* inM, //inM[M]
    __global const int* W,   // [M*M]
    __global const int* b    // [M] or null
){
    ulong m = 0ul;
    for (int i = 0; i < M; ++i) {
        float sum = b ? i32_to_f(b[i]) : 0.0f;
        for (int j = 0; j < M; ++j) sum += i32_to_f(W[i*M + j]) * inM[j];
        float a = relu(sum);
        outM[i] = a;
        m = relu_mask_setbit(m, i, (a > 0.0f));
    }
    *mask_out = m;
}

// FC linear out: out[out_dim] = W[out_dim,M]*inM + b[out_dim]
static inline void fc_linear_out(
    float* out,
    int out_dim,
    const float* inM, //inM[M]
    __global const int* W,   // [out_dim * M]
    __global const int* b    // [out_dim] or null
){
    for (int i = 0; i < out_dim; ++i) {
        float sum = b ? i32_to_f(b[i]) : 0.0f;
        for (int j = 0; j < M; ++j) sum += i32_to_f(W[i*M + j]) * inM[j];
        out[i] = sum;
    }
}

// ------------------------------------
// Backprop through linear: delta_in[M] = W[out_dim,M]^T * delta_out[out_dim]
// ------------------------------------
static inline void bwd_linear_to_M(
    float* delta_in, //delta_in[M]
    const float* delta_out,
    int out_dim,
    __global const int* W   // [out_dim * M]
){
    for (int j = 0; j < M; ++j) {
        float s = 0.0f;
        for (int i = 0; i < out_dim; ++i) s += i32_to_f(W[i*M + j]) * delta_out[i];
        delta_in[j] = s;
    }
}

// Backprop through M->M: delta_in[M] = W[M,M]^T * delta_out[M]
static inline void bwd_mm_to_M(
    float* delta_in, //delta_in[M]
    const float* delta_out, //delta_out[M]
    __global const int* W   // [M*M]
){
    for (int j = 0; j < M; ++j) {
        float s = 0.0f;
        for (int i = 0; i < M; ++i) s += i32_to_f(W[i*M + j]) * delta_out[i];
        delta_in[j] = s;
    }
}


__kernel void ffnerf_train_step(
    // -------------------------
    // Weights and biases (float bits in int)
    // -------------------------
    // Density MLP: 32 -> 64 -> 16
    __global int* Wd_in,    // M×L_POS  row-major (M rows, L_POS cols)
    __global int* Wd_01,    // M×M (only if ND>=2)
    __global int* Wd_12,    // M×M (only if ND>=3)
    __global int* Wd_out,   // D_OUT×M  row-major (D_OUT rows, M cols)
    __global int* bd_0,     // M
    __global int* bd_1,     // M (only if ND>=2)
    __global int* bd_2,     // M (only if ND>=3)
    __global int* bd_out,   // D_OUT
    // Color args (max)
    __global int* Wc_in,    // M×C_IN   //C_IN=(D_OUT+L_POS)
    __global int* Wc_01,    // M×M (if NC>=2)
    __global int* Wc_12,    // M×M (if NC>=3)
    __global int* Wc_23,    // M×M (if NC>=4)
    __global int* Wc_out,   // C_OUT×M
    __global int* bc_0,     // M
    __global int* bc_1,     // M (if NC>=2)
    __global int* bc_2,     // M (if NC>=3)
    __global int* bc_3,     // M (if NC>=4)
    __global int* bc_out,   // C_OUT   row-major (C_OUT rows, M cols)

    // Hash-grid
    __global int*  table,           // big table

    // Rays and targets
    __global const Ray*   rays,        // [num_rays]
    __global const float4* target_rgb, // [num_rays] target rgb in xyz; w ignored

    // -------------------------
    // Saved activations / errors for your update kernels // (not-last activations, not-first errors)
    // -------------------------
    // Errors (deltas)
    // WARNING I SHUFFLED THESE UP BADY, BUT WE CAN ADJUST TO THE NEW CONVENTION
    //__global TRANSFER_T*  gl_dpos,     // [L_POS * Btot]   grad wrt pos-encoding (for debugging; optional)
    __global TRANSFER_T*  gl_dd1,      // [M * Btot]       delta at density hidden (z_d1)
    __global TRANSFER_T*  gl_dd2,      // [M * Btot]       delta at density hidden (z_d1)
    __global TRANSFER_T*  gl_dd3,      // [M * Btot]       delta at density hidden (z_d1)
    __global TRANSFER_T*  gl_dd_out,   // [D_OUT * Btot]   delta at density output preact (z_d_out)
    __global TRANSFER_T*  gl_dc1,      // [M * Btot]       delta at color hidden1 (z_c1)
    __global TRANSFER_T*  gl_dc2,      // [M * Btot]       delta at color hidden2 (z_c2)
    __global TRANSFER_T*  gl_dc3,      // [M * Btot]       delta at color hidden2 (z_c2)
    __global TRANSFER_T*  gl_dc4,      // [M * Btot]       delta at color hidden2 (z_c2)
    __global TRANSFER_T*  gl_dc_out,   // [C_OUT * Btot]   delta at RGB preact (z_c_out)
    // Activations (not-last)
    __global TRANSFER_T*  gl_apos,     // [L_POS * Btot]   pos encoding (a0)
    __global TRANSFER_T*  gl_ad1,      // [M * Btot]       density hidden activation
    __global TRANSFER_T*  gl_ad2,      // [M * Btot]       density hidden activation
    __global TRANSFER_T*  gl_ad3,      // [M * Btot]       density hidden activation
    __global TRANSFER_T*  gl_ac_in,    // [C_IN * Btot]    color input (a_d_out + SH)
    __global TRANSFER_T*  gl_ac1,      // [M * Btot]       color hidden1 activation
    __global TRANSFER_T*  gl_ac2,      // [M * Btot]       color hidden2 activation
    __global TRANSFER_T*  gl_ac3,      // [M * Btot]       color hidden2 activation
    __global TRANSFER_T*  gl_ac4,      // [M * Btot]       color hidden2 activation

    // Output (for inspection)
    __global float4* out_rgb,          // [num_rays] predicted rgb in xyz

    // Params
    float lr,
    uint  num_rays,
    uint  samples_per_ray,
    int   do_train,

    // spherical harmonics can get a multiplier, to mash their effect until later in training.
    float sh_multiplier,

    // --- Occupancy-guided sampling inputs ---
    __global uint* occ_grid_in,    // uint[G*G*G] packed: 8 subcells x 4-bit - for marching and proposal baseline
    __global uint* occ_grid_st,    // uint[G*G*G] 8 nibbles, for storing max proposals // TODO update host
    float3 box_min,
    float3 box_max,
    int    G,
    float  rho_min,
    float  rho_max,
    float  dt_min,
    float  dt_max,
    float  t_near_clip,
    float  t_far_clip,
    uint   rng_seed_base,
    float  occ_update_strength
){
    
    const uint ray_id = get_group_id(0);
    const uint s_id   = get_local_id(0);
    const uint lsize  = get_local_size(0);

    if (samples_per_ray > S_MAX) return;
    if (s_id >= samples_per_ray) return;
    // ------------------------------------------------------------
    // Indices for global staging buffers (safe to compute early)
    // ------------------------------------------------------------
    const int Btot  = (int)(num_rays * samples_per_ray);
    const int batch = (int)(ray_id * samples_per_ray + s_id);

    // Helper: only write staging buffers during training
    #define IF_TRAINING if (do_train != 0)

    // -------------------------
    // Local shared arrays
    // -------------------------
    __local float CO[S_MAX * 4];        // [s][0]=sigma, [s][1..3]=rgb
    __local float Cin[S_MAX * 3];       // C_in before applying sample s (rgb)
    __local float dL_dc[S_MAX * 3];     // per-sample grad wrt color c_s (rgb)
    __local float dL_dsigma[S_MAX];     // per-sample grad wrt sigma_s
    // __local int   temp_store[M];        // slice accumulation buffer

    __local float Tr[S_MAX];            // T accumulator
    __local float Ts[S_MAX];            // per-sample t
    __local float DTs[S_MAX];           // per-sample dt
    __local uint chg_flag[CHG_WORDS];   // 256 bits when S_MAX=256
    
    __local Aff aff_scan[S_MAX];        // fwd and backprop
    __local Aff aff_elem[S_MAX];        // fwd and backprop
    __local Seg segs[S_MAX];
    __local float3 gr_l;
    __local float  gT_end_l;

    // forward pass helpers. Note: Tr, DTs, CO, aff_elem are 'reused' in forward scan.
    __local float t0_l, t1_l;
    __local float dt_base_l;
    __local float u0_l;
    __local int   hit_l;


    // Per-thread registers
    Ray r = rays[ray_id];
    
    
    float A[M];
    float B[M];
    // --- inputs ---
    float a0[L_POS];    // hashgrid position encoding - keep
    // --- density MLP: 32 -> 64 -> 16 ---
    // --- color MLP: 32 -> 64 -> 64 -> 3 ---
    float a_cin[C_IN];          // concat [a_dout (16), sh (16)] // TODO: if we know CIN<M, eliminate this buffer
    // --- outputs ---
    float sigma;
    float3 rgb;
    // --- deltas ---
    float delta_cout[C_OUT];
    float delta_dout[D_OUT];

    // solid angle per ray (for now: global constant)
    // const float solid_angle = 3.662459967e-6f;
    // const float angle_alpha = sqrt(solid_angle); // so solid_angle * t*(t+dt) = (angle_alpha*t)*(angle_alpha*(t+dt))

#if (SURPRESS_SAMPLES == 0)
    // ------------------------------------------------------------
    // (A') WG-parallel sampler (fixed cost ~O(WG log WG), no DDA)
    // Produces:
    //   Ts[0..WG-1]   sorted (nondecreasing)
    //   DTs[0..WG-1]  dt_base (clamped)
    //   chg_flag      run-start flags based on (cell,sub) changes
    // Requirements: WG == samples_per_ray, WG <= S_MAX, WG power-of-two
    // Reuses locals:
    //   CO[0..S_MAX-1]      as cand_t
    //   CO[S_MAX..2S_MAX-1] as w_pow
    //   Tr[0..S_MAX-1]      as scan/reduce/CDF buffer
    //   aff_elem memory     as key_l (cast to uint*)
    // ------------------------------------------------------------
    {
        const uint lid = s_id;
        const uint WG  = (uint)get_local_size(0);

        // If host ever launches mismatch, fail safe (uniform across WG -> no divergence issues)
        if (WG != (uint)samples_per_ray || WG > (uint)S_MAX) {
            // produce trivial outputs so later code is benign
            if (lid < (uint)CHG_WORDS) chg_flag[lid] = 0u;
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lid == 0u) {
                for (uint i = 0u; i < WG; ++i) { Ts[i] = 0.0f; DTs[i] = 0.0f; }
                // set all run starts
                for (uint i = 0u; i < WG; ++i) chg_set(chg_flag, i);
            }
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        } else {

            // alias scratch buffers (reuse existing locals)
            __local float* cand_t = CO;             // size >= WG
            __local float* w_pow  = CO + S_MAX;     // size >= WG
            __local float* cdf_l  = Tr;             // size >= WG
            __local uint*  key_l  = (__local uint*)aff_elem; // reuse aff_elem storage

            // Clear change flags (parallel clear by first CHG_WORDS lanes)
            if (lid < (uint)CHG_WORDS) chg_flag[lid] = 0u;

            // lane0 computes hit interval once
            if (lid == 0u) {
                float t0, t1;
                int hit = ray_aabb_intersect(r.o.xyz, r.d.xyz, box_min, box_max, &t0, &t1);

                if (!hit) {
                    hit_l = 0; t0_l = 0.0f; t1_l = 0.0f;
                } else {
                    t0 = fmax(t0, t_near_clip);
                    t1 = fmin(t1, t_far_clip);
                    if (t1 <= t0) { hit_l = 0; t0_l = 0.0f; t1_l = 0.0f; }
                    else { hit_l = 1; t0_l = t0 + 1.0e-6f; t1_l = t1; }
                }

                // shared dt_base (clamped)
                float dtb = (hit_l != 0) ? ((t1_l - t0_l) * (1.0f / (float)WG)) : 0.0f;
                dtb = fmin(dt_max, fmax(dt_min, dtb));
                dt_base_l = (hit_l != 0) ? dtb : 0.0f;

                // shared u0 for systematic resampling
                if (hit_l != 0) {
                    uint rng0 = hash_u32(rng_seed_base ^ (uint)ray_id * 1597334677u);
                    u0_l = rand01(&rng0) * (1.0f / (float)WG);
                } else {
                    u0_l = 0.0f;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            // no-hit: lane0 fills outputs (avoid racy bit-sets)
            if (hit_l == 0) {
                if (lid == 0u) {
                    for (uint i = 0u; i < WG; ++i) { Ts[i] = 0.0f; DTs[i] = 0.0f; }
                    for (uint i = 0u; i < WG; ++i) chg_set(chg_flag, i);
                }
                barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            } else {
                const float t0 = t0_l, t1 = t1_l;
                const float dt_base = dt_base_l;

                // constants (your tunables)
                const float lambda_focus = LAMBDA_FOCUS;
                const float gamma_w      = GAMMA_W;
                const float uniform_part = UNIFORM_PART;
                const float EPS_W        = 1.0e-12f;

                // stratified candidate time (sorted by lid by construction)
                uint rng = hash_u32(rng_seed_base ^ (uint)ray_id * 747796405u ^ lid * 2891336453u);
                float u  = ((float)lid + rand01(&rng)) * (1.0f / (float)WG); // [0,1)
                float tk = t0 + (t1 - t0) * u;

                cand_t[lid] = tk;

                // occupancy read at tk
                float3 box_size = box_max - box_min;
                float3 inv_box  = (float3)(1.0f/box_size.x, 1.0f/box_size.y, 1.0f/box_size.z);

                float3 xk = r.o.xyz + tk * r.d.xyz;
                float3 gm = (xk - box_min) * inv_box * (float)G;

                float occ = 0.0f;
                if (gm.x >= 0.0f && gm.x < (float)G &&
                    gm.y >= 0.0f && gm.y < (float)G &&
                    gm.z >= 0.0f && gm.z < (float)G)
                {
                    int ixm = clamp((int)floor(gm.x), 0, G-1);
                    int iym = clamp((int)floor(gm.y), 0, G-1);
                    int izm = clamp((int)floor(gm.z), 0, G-1);
                    int cell = ixm + G*(iym + G*izm);
                    uint packed = occ_grid_in[cell];

                    float fx = gm.x - floor(gm.x);
                    float fy = gm.y - floor(gm.y);
                    float fz = gm.z - floor(gm.z);

                    // avoid edge weirdness
                    fx = clamp(fx, 0.0f, 1.0f - 1e-7f);
                    fy = clamp(fy, 0.0f, 1.0f - 1e-7f);
                    fz = clamp(fz, 0.0f, 1.0f - 1e-7f);

                    uint sx = (fx > 0.5f);
                    uint sy = (fy > 0.5f);
                    uint sz = (fz > 0.5f);
                    uint sub = sx + 2u*sy + 4u*sz;

                    // occ = (float)(occ4 + 1u) * (1.0f/16.0f);
                    uint occ4 = (packed >> (4u*sub)) & 0xFu;   // 0..15
                    uint q_train = occ4 + 1u;
                    uint q_infer = (occ4 > 2u) ? (occ4 - 2u) : 0u;
                    uint q_eff = (do_train != 0) ? q_train : q_infer;
                    occ = (float)q_eff * (1.0f/16.0f);

                }
                // --------------------------
                // Weight construction
                // --------------------------
                float infer_f = (do_train == 0) ? 1.0f : 0.0f;

                // active=1 iff q_eff>0 (only matters in inference; in train, q_eff>=1 always)
                float active = (occ > 0.0f) ? 1.0f : 0.0f;

                // Base rho from occ
                float rho_base = rho_min + (rho_max - rho_min) * occ;

                // In inference: truly zero-out rho when inactive so these cells have wp=0
                // scale = 1 for training; = active for inference
                float rho = rho_base * (1.0f - infer_f + infer_f * active);

                float w_raw = rho * dt_base;
                float wp    = safe_pow_pos(w_raw, gamma_w);
                w_pow[lid]  = wp;

                // --------------------------
                // Prefix sums etc (unchanged)
                // --------------------------
                barrier(CLK_LOCAL_MEM_FENCE);

                // prefix sum of w_pow -> pref
                cdf_l[lid] = wp;
                barrier(CLK_LOCAL_MEM_FENCE);
                scan_excl_f_inplace(cdf_l, lid, WG);
                float pref = cdf_l[lid];

                // Wtot = sum(w_pow)
                cdf_l[lid] = wp;
                barrier(CLK_LOCAL_MEM_FENCE);
                float Wtot = reduce_sum_f_inplace(cdf_l, lid, WG);
                Wtot = fmax(Wtot, EPS_W);
                barrier(CLK_LOCAL_MEM_FENCE);

                float pi = pref / Wtot;
                float g  = wp * focus_gain_unitprefix(pi, lambda_focus);

                // gtot = sum(g)
                cdf_l[lid] = g;
                barrier(CLK_LOCAL_MEM_FENCE);
                float gtot = reduce_sum_f_inplace(cdf_l, lid, WG);
                gtot = fmax(gtot, EPS_W);
                barrier(CLK_LOCAL_MEM_FENCE);

                // --------------------------
                // Uniform-part masking (ignore inactive when do_train==0)
                // --------------------------
                // active_count = sum(active) across WG
                cdf_l[lid] = active;
                barrier(CLK_LOCAL_MEM_FENCE);
                float active_count = reduce_sum_f_inplace(cdf_l, lid, WG);
                barrier(CLK_LOCAL_MEM_FENCE);

                // any_active = 1 if there is at least one active cell, else 0
                float any_active = (active_count > 0.5f) ? 1.0f : 0.0f;

                // uniform_term:
                //   if any_active: active / active_count  (inactive get 0 prob)
                //   else:          1/WG (fallback to avoid all-zero CDF)
                float inv_active = 1.0f / fmax(active_count, 1.0f);
                float uniform_term_active = active * inv_active;
                float uniform_term = mix(1.0f / (float)WG, uniform_term_active, any_active);

                // proposal term (g/gtot) is already 0 for inactive since wp=0,
                // but if no active, kill it to avoid numerical weirdness
                float proposal_term = any_active * (g / gtot);

                // mixture probability p_i
                float p = uniform_part * uniform_term + (1.0f - uniform_part) * proposal_term;

                // build inclusive CDF in cdf_l (unchanged)
                cdf_l[lid] = p;
                barrier(CLK_LOCAL_MEM_FENCE);
                scan_excl_f_inplace(cdf_l, lid, WG);
                float cdf = cdf_l[lid] + p;  // inclusive
                cdf_l[lid] = cdf;
                barrier(CLK_LOCAL_MEM_FENCE);

                // systematic resampling (uk increasing => idx nondecreasing => Ts sorted)
                float uk = u0_l + (float)lid * (1.0f / (float)WG);
                uk = fmin(uk, 0.99999994f);

                uint idx = lower_bound_cdf_local(cdf_l, WG, uk);

                float t_out = cand_t[idx];
                Ts[lid]  = t_out;
                DTs[lid] = dt_base;

                // compute packed (cell,sub) key for run detection
                float3 xo = r.o.xyz + t_out * r.d.xyz;
                float3 go = (xo - box_min) * inv_box * (float)G;

                uint key = 0xFFFFFFFEu;
                int ixo = (int)floor(go.x);
                int iyo = (int)floor(go.y);
                int izo = (int)floor(go.z);

                if ((uint)ixo < (uint)G && (uint)iyo < (uint)G && (uint)izo < (uint)G) {
                    float fx = go.x - floor(go.x);
                    float fy = go.y - floor(go.y);
                    float fz = go.z - floor(go.z);

                    fx = clamp(fx, 0.0f, 1.0f - 1e-7f);
                    fy = clamp(fy, 0.0f, 1.0f - 1e-7f);
                    fz = clamp(fz, 0.0f, 1.0f - 1e-7f);

                    uint sx = (fx >= 0.5f);
                    uint sy = (fy >= 0.5f);
                    uint sz = (fz >= 0.5f);
                    uint sub = sx + 2u*sy + 4u*sz;

                    int cell = ixo + G*(iyo + G*izo);
                    key = pack_cellsub(cell, sub);
                }

                key_l[lid] = key;
                barrier(CLK_LOCAL_MEM_FENCE);

                // run-start flags (use atomic set to avoid word races)
                if (lid == 0u || key_l[lid] != key_l[lid - 1u]) {
                    chg_set_atomic(chg_flag, lid);
                }

                barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    // ------------------------------------------------------------
    // 1) Sample position along ray (now from Ts/DTs)
    // ------------------------------------------------------------
    float t = Ts[s_id];
    float dt_here = DTs[s_id];
    float3 x = r.o.xyz + t * r.d.xyz;

#else

    // --- suppress mode: ignore rays for sampling ---
    // We still keep r for direction encoding convenience, but position is sampled independently.
    uint rng = hash_u32(rng_seed_base ^ (uint)ray_id * 1597334677u ^ (uint)s_id * 3812015801u ^ 0xBADC0FFEu);

    int  cell_s = 0;
    uint sub_s  = 0u;
    float3 x = random_point_in_empty_nibble(&rng, occ_grid_in, box_min, box_max, G, &cell_s, &sub_s);

    // dt_here is only used for alpha/compositing in normal mode; in suppress mode we don't composite.
    // Keep it >0 to avoid accidental sigma=0 forcing etc.
    float dt_here = 1.0f;

    // keep Ts/DTs benign if later code accidentally reads them
    Ts[s_id]  = 0.0f;
    DTs[s_id] = dt_here;

#endif

    // ------------------------------------------------------------
    // 2) Instant-NGP style multi-level hash-grid encoding -> a0[L]
    //    WITHOUT storing feature_idx/feature_w across levels.
    //    We only keep per-level corner cache (8 weights + 8 base indices) in registers.
    // ------------------------------------------------------------
    {
        const float R0_f   = RESOLUTION_LVL0;
        const float Rmax_f = RESOLUTION_LVLMAX;

        // ---- normalize x into [0,1]^3 using the same AABB you already have ----
        float3 box_size = box_max - box_min;
        float3 p = (x - box_min) / box_size;

        // clamp to [0, 1-eps] so we don't hit the top boundary
        const float eps = 1.0e-6f;
        p = clamp(p, 0.0f, 1.0f - eps);

        // Hash entry macro (same as yours)
        #define HASH_ENTRY(ix,iy,iz,lvl) \
            (hash_u32( \
                (uint)(ix) * 73856093u ^ \
                (uint)(iy) * 19349663u ^ \
                (uint)(iz) * 83492791u ^ \
                (uint)(lvl) * 2654435761u) & level_mask(lvl))

        for (int lvl = 0; lvl < N_LEVELS; ++lvl) {
            // --- same resolution schedule ---
            float t_lvl = (float)lvl * (1.0f / (float)(N_LEVELS - 1));
            float Rf = R0_f * pow(Rmax_f / R0_f, t_lvl);
            int R = (int)rint(Rf);
            if (R < 2) R = 2;

            // grid coordinate in [0, R)
            float3 u = p * (float)R;

            int ix0 = (int)floor(u.x);          int iy0 = (int)floor(u.y);          int iz0 = (int)floor(u.z);
            ix0 = clamp(ix0, 0, R - 2);         iy0 = clamp(iy0, 0, R - 2);         iz0 = clamp(iz0, 0, R - 2);

            float fx = u.x - (float)ix0;        float fy = u.y - (float)iy0;        float fz = u.z - (float)iz0;
            float wx0 = 1.0f - fx, wx1 = fx;    float wy0 = 1.0f - fy, wy1 = fy;    float wz0 = 1.0f - fz, wz1 = fz;

            // Per-level tiny cache (registers only, does NOT grow with N_LEVELS):
            float w_corner[8];
            int   base_feat_corner[8]; // base index for f=0 (so idx = base + f)

            #pragma unroll
            for (int c = 0; c < 8; ++c) {
                int dx = (c & 1) ? 1 : 0;   int dy = (c & 2) ? 1 : 0;   int dz = (c & 4) ? 1 : 0;

                float w = (dx ? wx1 : wx0) * (dy ? wy1 : wy0) * (dz ? wz1 : wz0);
                w_corner[c] = w;

                uint entry = HASH_ENTRY(ix0 + dx, iy0 + dy, iz0 + dz, lvl);

                // base feature index in table[] (for f=0)
                uint base_entry = LEVEL_OFFSET_ENTRIES[lvl] + entry;
                base_feat_corner[c] = (int)(base_entry * (uint)F_PER_LEVEL);
            }

            float acc[F_PER_LEVEL];
            #pragma unroll
            for (int f = 0; f < F_PER_LEVEL; ++f) acc[f] = 0.0f;
            if (table) {
                #pragma unroll
                for (int c = 0; c < 8; ++c) {
                    float w = w_corner[c];
                    int base = base_feat_corner[c];
                    #pragma unroll
                    for (int f = 0; f < F_PER_LEVEL; ++f) {
                        acc[f] += w * i32_to_f(table[base + f]);
                    }
                }
            }
            #pragma unroll
            for (int f = 0; f < F_PER_LEVEL; ++f) {
                a0[lvl * F_PER_LEVEL + f] = acc[f];
            }
        }
        #undef HASH_ENTRY
    }
    IF_TRAINING {
        #pragma unroll
        for (int i = 0; i < L_POS; ++i) tstore(gl_apos, i*Btot + batch, a0[i]);
    }

    // ------------------------------------------------------------
    // 3) Forward pass (instant-ngp NeRF split MLPs)
    //    Density: 32->64->16, Color: (16+16)->64->64->3
    // ------------------------------------------------------------
    // ------------------------------
    // Forward: Density MLP (ND hidden layers, width M)
    // ------------------------------
    float* cur = A;
    float* nxt = B;

    ulong dmask1=0ul, dmask2=0ul, dmask3=0ul;

    {
        // layer 0: a0[L_POS] -> h1[M]
        fc_relu_in(cur, &dmask1, a0, L_POS, Wd_in, bd_0);
        IF_TRAINING {
            for (int i = 0; i < M; ++i) tstore(gl_ad1, i*Btot + batch, cur[i]);
        }

    #if (ND >= 2)
        // h1 -> h2
        fc_relu_mm(nxt, &dmask2, cur, Wd_01, bd_1);
        { float* tmp = cur; cur = nxt; nxt = tmp; }
        IF_TRAINING {
            for (int i = 0; i < M; ++i) tstore(gl_ad2, i*Btot + batch, cur[i]);
        }
    #endif

    #if (ND >= 3)
        // h2 -> h3
        fc_relu_mm(nxt, &dmask3, cur, Wd_12, bd_2);
        { float* tmp = cur; cur = nxt; nxt = tmp; }
        IF_TRAINING {
            for (int i = 0; i < M; ++i) tstore(gl_ad3, i*Btot + batch, cur[i]);
        }
    #endif

        // out: hLast -> dout[D_OUT] (preact)
        float dout[D_OUT];
        fc_linear_out(dout, D_OUT, cur, Wd_out, bd_out);

        // stash into first D_OUT of a_cin (this matches your current convention)
        for (int i = 0; i < D_OUT; ++i) a_cin[i] = dout[i];

        // sigma from density head (dim 0)
        sigma = softplus(clamp(dout[0], -10.0f, 10.0f));
    }

#if (SURPRESS_SAMPLES == 0)
    // ------------------------------
    // Forward: Color MLP (NC hidden layers, width M)
    // input is a_cin[C_IN] = [dout(16), SH(16)]
    // ------------------------------
    ulong cmask1=0ul, cmask2=0ul, cmask3=0ul, cmask4=0ul;

    {
        // SH(dir) into a_cin[D_OUT..]
        directional_encode(r.d.xyz, &a_cin[D_OUT]);
        for (int i = 0; i < L_DIR; ++i) a_cin[D_OUT+i] *= sh_multiplier;

        IF_TRAINING {
            for (int i = 0; i < C_IN; ++i) tstore(gl_ac_in, i*Btot + batch, a_cin[i]);
        }

        // layer0: a_cin[C_IN] -> h1[M]
        fc_relu_in(cur, &cmask1, a_cin, C_IN, Wc_in, bc_0);
        IF_TRAINING {
            for (int i = 0; i < M; ++i) tstore(gl_ac1, i*Btot + batch, cur[i]);
        }

    #if (NC >= 2)
        fc_relu_mm(nxt, &cmask2, cur, Wc_01, bc_1);
        { float* tmp = cur; cur = nxt; nxt = tmp; }
        IF_TRAINING {
            for (int i = 0; i < M; ++i) tstore(gl_ac2, i*Btot + batch, cur[i]);
        }
    #endif

    #if (NC >= 3)
        fc_relu_mm(nxt, &cmask3, cur, Wc_12, bc_2);
        { float* tmp = cur; cur = nxt; nxt = tmp; }
        IF_TRAINING {
            for (int i = 0; i < M; ++i) tstore(gl_ac3, i*Btot + batch, cur[i]);
        }
    #endif

    #if (NC >= 4)
        fc_relu_mm(nxt, &cmask4, cur, Wc_23, bc_3);
        { float* tmp = cur; cur = nxt; nxt = tmp; }
        IF_TRAINING {
            for (int i = 0; i < M; ++i) tstore(gl_ac4, i*Btot + batch, cur[i]);
        }
    #endif

        // out: hLast -> rgb preacts -> sigmoid_soft
        float cout[C_OUT];
        fc_linear_out(cout, C_OUT, cur, Wc_out, bc_out);
        rgb.x = sigmoid_soft(cout[0]);
        rgb.y = sigmoid_soft(cout[1]);
        rgb.z = sigmoid_soft(cout[2]);
    }

#else
    // suppress mode: no color network evaluation
    rgb = (float3)(0.0f, 0.0f, 0.0f);

    // Optional: if you want activations buffers to be consistent, you can write zeros.
    // Not strictly required if your update kernels only use deltas.
    IF_TRAINING {
        // gl_ac_in (C_IN), gl_ac1..4 (M) -> zero
        for (int i = 0; i < C_IN; ++i) tstore(gl_ac_in, i*Btot + batch, 0.0f);
        for (int i = 0; i < M; ++i) {
            tstore(gl_ac1, i*Btot + batch, 0.0f);
            tstore(gl_ac2, i*Btot + batch, 0.0f);
            tstore(gl_ac3, i*Btot + batch, 0.0f);
            tstore(gl_ac4, i*Btot + batch, 0.0f);
        }
    }
#endif

    // ------------------------------------------------------------
    // (B) STORE per-sample outputs for compositing/backward
    // ------------------------------------------------------------
    // dt==0 => no-op sample (force alpha to 0 later)
    if (dt_here <= 0.0f) sigma = 0.0f;

    const int base = (int)s_id * 4;
    CO[base + 0] = sigma;
    CO[base + 1] = rgb.x;
    CO[base + 2] = rgb.y;
    CO[base + 3] = rgb.z;

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

#if (SURPRESS_SAMPLES == 0)
    // ------------------------------------------------------------
    // 4) Workgroup-parallel compositing + loss + per-sample grads
    //    Works for WG size power-of-two up to WG_SIZE eg 256.
    // ------------------------------------------------------------
    {
        const uint lid = s_id;
        const uint WG  = (uint)lsize;

        // fail-fast / safety (remove if you want max speed and trust host)
        if (WG > (uint)S_MAX) return;
        if (WG != (uint)samples_per_ray) return;
        // Optional power-of-two check:
        // if ((WG & (WG - 1u)) != 0u) return;

        // ---- per-lane fetch ----
        float3 c_s = rgb; //(float3)(CO[lid*4 + 1], CO[lid*4 + 2], CO[lid*4 + 3]);
        float  dt_s = DTs[lid];

        float alpha = 0.0f;
        if (dt_s > 0.0f) { // TODO use SxDT_CAP_FOR_ALPHA to discourage very large density
            float x = clamp(- CO[lid*4 + 0] * dt_s, SxDT_CAP_FOR_ALPHA, 18.0f);
            alpha = 1.0f - exp(x); // x <= 0 usually
        }
        float one_minus_alpha = 1.0f - alpha;

        // ---- forward scan (segment) ----
        Seg my; my.a = one_minus_alpha; my.v = alpha * c_s;
        segs[lid] = my;
        barrier(CLK_LOCAL_MEM_FENCE);

        Seg total_seg;
        scan_excl_seg(segs, lid, WG, &total_seg);

        float  T_s = segs[lid].a;
        float3 Cin_s = segs[lid].v;

        Tr[lid] = T_s; // here is Tr, Tr[s+1]=Tr[s]*(1-alpha_s) -> alpha_s = 1-T[s+1]/T[s] easy
        Cin[lid*3 + 0] = Cin_s.x;
        Cin[lid*3 + 1] = Cin_s.y;
        Cin[lid*3 + 2] = Cin_s.z;

        float  T_final = total_seg.a;
        float3 C_in    = total_seg.v;

        // ---- use shared loss grad ---- gr_l gT_end_l


        if (lid == 0) { // --- add background contribution ---
            // black bg baked in OR random background (per-ray)
            float3 bg = (float3)(0.0f,0.0f,0.0f);
            if (do_train){
                #ifdef BACKGROUND_COLOR
                    #if (BACKGROUND_COLOR==0)
                        bg = (float3)(0.0f,0.0f,0.0f);
                    #elif (BACKGROUND_COLOR==1)
                        bg = (float3)(1.0f,1.0f,1.0f);
                    #elif (BACKGROUND_COLOR==2)
                        uint rng_bg = hash_u32(rng_seed_base ^ (uint)ray_id * 747796405u ^ 0xA511E9B3u);
                        bg = (float3)(rand01(&rng_bg), rand01(&rng_bg), rand01(&rng_bg));
                    #else
                        bg = (float3)(0.0f,0.0f,0.0f);
                    #endif
                #endif
            }
            else {
                bg = (float3)(0.0f,0.0f,0.0f);
            }

            float3 C_final = C_in + T_final * bg;

            out_rgb[ray_id] = (float4)(
                clamp(C_final.x, 0.0f, 1.0f),
                clamp(C_final.y, 0.0f, 1.0f),
                clamp(C_final.z, 0.0f, 1.0f),
                0.0f
            );

            float3 tgt  = target_rgb[ray_id].xyz;
            // charbonnier_prime(x)
            float3 diff = C_final - tgt;
            gr_l.x = charbonnier_prime(diff.x); // L2:diff for 0.5*||diff||^2; charbonnier:charbonnier_prime(diff) for charbonnier(diff)
            gr_l.y = charbonnier_prime(diff.y); // L2:diff for 0.5*||diff||^2; charbonnier:charbonnier_prime(diff) for charbonnier(diff)
            gr_l.z = charbonnier_prime(diff.z); // L2:diff for 0.5*||diff||^2; charbonnier:charbonnier_prime(diff) for charbonnier(diff)

            // gT_N = dL/dT_final = gr · bg  (0 for black)
            gT_end_l = gr_l.x*bg.x + gr_l.y*bg.y + gr_l.z*bg.z;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        float3 gr = gr_l;
        float  gT_end = gT_end_l;

        // ---- per-lane dL/dc ----
        float w_s = T_s * alpha;
        float3 dcs = w_s * gr;
        dL_dc[lid*3 + 0] = dcs.x;
        dL_dc[lid*3 + 1] = dcs.y;
        dL_dc[lid*3 + 2] = dcs.z;

        // ---- backward via reversed affine scan ----
        // B_s = gr · (alpha*c_s)
        float B_s = gr.x*(alpha*c_s.x) + gr.y*(alpha*c_s.y) + gr.z*(alpha*c_s.z);

        Aff f; f.a = one_minus_alpha; f.b = B_s;

        uint k = (WG - 1u) - lid;  // reversed index
        aff_elem[k] = f;
        aff_scan[k] = f;
        barrier(CLK_LOCAL_MEM_FENCE);

        Aff total_aff;
        scan_excl_aff_rev(aff_scan, lid, WG, &total_aff);
        barrier(CLK_LOCAL_MEM_FENCE);

        // gT_{s+1} = (f_{s+1} ° ... ° f_{WG-1})(gT_end)
        float gT_next = gT_end;
        if (k >= 1u) {
            uint m = k - 1u;
            // inclusive prefix on r[0..m] in rev-composition
            Aff incl = aff_combine_rev(aff_scan[m], aff_elem[m]);
            gT_next = incl.a * gT_end + incl.b;
        }

        // gAlpha and dL/dsigma
        float gAlpha_from_C = gr.x*(T_s*c_s.x) + gr.y*(T_s*c_s.y) + gr.z*(T_s*c_s.z);
        float gAlpha = gAlpha_from_C - gT_next * T_s;

        // dalpha/dsigma = dt * exp(-sigma dt) = dt*(1-alpha)
        float dalpha_dsigma = (dt_s > 0.0f) ? (dt_s * one_minus_alpha) : 0.0f;
        // density regularizer:  punish large density*dt contributions to discourage sparkles and blobs that are not related to actual surfaces
        // total loss += \lambda * \sum_s (alpha_s) /samples_per_ray   aka mean alpha, hence each dL/d_alpha = \lambda / samples_per_ray,
        //   so dL_dsigma += \lambda * dt_s / samples_per_ray
        float regu = (dt_s > 0.0f) ? (DENSITY_PENALTY_LMD * dt_s / samples_per_ray) : 0.0f;  // invWG = 1.0f / WG
        dL_dsigma[lid] = gAlpha * dalpha_dsigma + regu; 

        // ------------------------------------------------------------
        // Occupancy grid update (representative per run)
        // Evidence per cell-run: sum_s (T_s * alpha_s) = Tr[start] - Tr[end]
        // ------------------------------------------------------------
        if (occ_update_strength > 0.0f && dt_s > 0.0f && do_train){
            // representative = first sample of a (cell,sub) run
            uint is_rep = chg_get(chg_flag, lid);

            if (is_rep){ // TODO: experiment with proposed value vunctions. instead of alpha=1-T'/T use ALPHA=(T_start-T_next_cell)/T_start
                // map this sample position to (cell, sub) (same as before)
                float3 box_size = box_max - box_min;
                float3 inv_box  = (float3)(1.0f/box_size.x, 1.0f/box_size.y, 1.0f/box_size.z);
                float3 gpos = (x - box_min) * inv_box * (float)G;

                int ix = (int)floor(gpos.x);
                int iy = (int)floor(gpos.y);
                int iz = (int)floor(gpos.z);

                if ((uint)ix < (uint)G && (uint)iy < (uint)G && (uint)iz < (uint)G){
                    float fx = gpos.x - floor(gpos.x);
                    float fy = gpos.y - floor(gpos.y);
                    float fz = gpos.z - floor(gpos.z);

                    uint sx = (fx >= 0.5f);
                    uint sy = (fy >= 0.5f);
                    uint sz = (fz >= 0.5f);
                    uint sub = sx + 2u*sy + 4u*sz;

                    int cell = ix + G*(iy + G*iz);

                    // run evidence /*sumW = sum(T_s*alpha_s)*/ over this run,
                    // new: ALPHAW = (T_start-T_end)/T_start, analog to T[s+1]=T[s]*(1-alpha_s)
                    // find next representative (or run end)
                    uint next = chg_find_next(chg_flag, lid, WG);

                    /*
                    // get contribution of cell from T_start-T_end. If next==WG, use T_final.
                    float T_start = Tr[lid];
                    float T_end   = (next < WG) ? Tr[next] : T_final;

                    // pred in [0,1]
                    float invTs = 1.0f / fmax(T_start, 1.0e-12f);
                    float pred  = clamp01(1.0f * (T_start - T_end) * invTs); // ALPHA, the relative opacity of the cell
                    */

                    // -----------------------------------------------
                    // Opacity correction: use cell boundary t_enter/t_exit
                    // and interpolate Tr[] to those boundary times.
                    // -----------------------------------------------
                    float pred = 0.0f;

                    {
                        // Cell AABB in world space (one grid level, edge length lambda = box_size/G)
                        float3 box_size = box_max - box_min;
                        float3 cell_size = box_size * (1.0f / (float)G);

                        float3 cell_min = box_min + (float3)((float)ix, (float)iy, (float)iz) * cell_size;
                        float3 cell_max = cell_min + cell_size;

                        float t_enter, t_exit;
                        int hit_cell = ray_aabb_intersect(r.o.xyz, r.d.xyz, cell_min, cell_max, &t_enter, &t_exit);

                        if (hit_cell) {
                            // (optional) clip to your global clip range
                            t_enter = fmax(t_enter, t_near_clip);
                            t_exit  = fmin(t_exit,  t_far_clip);

                            if (t_exit > t_enter) {
                                // We will interpolate T(t) using two anchors:
                                // - start side: (t_prev, Tr[prev_end])  and (t_cur, Tr[lid])
                                // - end side:   (t_last, Tr[last])      and (t_next, Tr[next_start] or T_final)

                                const float EPS_T = 1.0e-12f;
                                const float EPS_DT = 1.0e-8f;

                                // ---- START interpolation (use prev_end and lid) ----
                                uint prev_end = (lid > 0u) ? (lid - 1u) : 0u;

                                float t_prev = Ts[prev_end];
                                float T_prev = Tr[prev_end];

                                float t_cur  = Ts[lid];
                                float T_cur  = Tr[lid];

                                float denom0 = t_cur - t_prev;
                                float u0 = (fabs(denom0) > EPS_DT) ? (t_enter - t_prev) / denom0 : 0.0f;
                                u0 = clamp(u0, 0.0f, 1.0f);

                                float T_enter = mix(T_prev, T_cur, u0);

                                // ---- END interpolation (use last in run and next_start / T_final) ----
                                uint next_start = next; // from chg_find_next(chg_flag, lid, WG)

                                uint last = (next_start < WG) ? (next_start - 1u) : (WG - 1u);

                                float t_last = Ts[last];
                                float T_last = Tr[last];

                                float t_next = (next_start < WG) ? Ts[next_start] : t_exit;
                                float T_next = (next_start < WG) ? Tr[next_start] : T_final;

                                float denom1 = t_next - t_last;
                                float u1 = (fabs(denom1) > EPS_DT) ? (t_exit - t_last) / denom1 : 1.0f;
                                u1 = clamp(u1, 0.0f, 1.0f);

                                float T_exit_interp = mix(T_last, T_next, u1);

                                // ---- cell opacity estimate in [0,1] ----
                                pred = clamp01((T_enter - T_exit_interp) / fmax(T_enter, EPS_T));
                            }
                        }
                    } // pred now replaces old "pred = (T_start - T_end)/T_start"

                    // old from input grid (const)
                    uint old_q = occ_nibble_read_q16(occ_grid_in, cell, sub);

                    // --- probability-based step-up in q-space ---
                    uint proposed_q = old_q;
                    
                    // optional alt: lerp but still cap jump (uncomment if wanted)
                    /*
                    float s = clamp(occ_update_strength, 0.0f, 1.0f);
                    uint lerp_q = (uint)floor((1.0f - s)*(float)old_q + s*(float)pred_q + (1.0f - s));
                    if (lerp_q < old_q) lerp_q = old_q;
                    uint proposed_q = min(old_q + 1u, lerp_q); // hard cap
                    */

                    float pred16 = pred * 16.0f; // in [0..16]
                    if (pred16 > (float)old_q && old_q < 16u)
                    {
                        float f0   = floor(pred16);
                        float frac = pred16 - f0;          // in [0,1)
                        uint  q0   = (uint)f0;             // 0..16 (but pred16>old_q>=1 => q0>=1)

                        // RNG: deterministic per (ray, lid, cell, sub) and seed base
                        uint r = hash_u32(
                            rng_seed_base ^
                            (uint)ray_id * 747796405u ^
                            (uint)lid    * 2891336453u ^
                            (uint)cell   * 9781u ^
                            (uint)sub    * 6271u
                        );
                        uint r24 = r & 0x00FFFFFFu;
                        uint thresh = (uint)(frac * 16777216.0f); // 2^24

                        // floor(pred16) with probability 1-frac, ceil with probability frac
                        uint q_stoch = (r24 < thresh) ? (q0 + 1u) : q0;

                        // if need to ensure it actually steps up
                        //if (q_stoch < old_q + 1u) q_stoch = old_q + 1u;
                        if (q_stoch > 16u) q_stoch = 16u;

                        proposed_q = q_stoch;
                    }

                    // do we send the signal? probabilistic compensation for distance 
                    // --- probabilistic keep to counter near-ray over-density ---
                    // lambda = occlusion cell edge length (assumes uniform grid; uses x extent)
                    float occlusion_lattice_constant = (box_max.x - box_min.x) * (1.0f / (float)G);
                    float D = 0.5f * length(box_max - box_min);

                    float tl = t / fmax(occlusion_lattice_constant, 1.0e-12f);
                    float Dl = D / fmax(occlusion_lattice_constant, 1.0e-12f);

                    float num = tl*tl + tl + (1.0f/3.0f);
                    float den = Dl*Dl + Dl + (1.0f/3.0f);
                    float f_keep = num / fmax(den, 1.0e-12f);

                    float keep_p = fmin(f_keep, 1.0f);
                    if (keep_p >= 1.0f) {
                        occ_nibble_atomic_max_u16(occ_grid_st, cell, sub, proposed_q);
                    } else {
                        uint rng_keep = hash_u32(
                            rng_seed_base ^
                            (uint)ray_id * 747796405u ^
                            (uint)lid    * 2891336453u ^
                            (uint)cell   * 9781u ^
                            (uint)sub    * 6271u
                        );
                        if (rand01(&rng_keep) <= keep_p) {
                            occ_nibble_atomic_max_u16(occ_grid_st, cell, sub, proposed_q);
                        }
                    }
                }
            }
        }
    }

#else

    // suppress mode: sigma-only “make empty” loss
    // Use L = 0.5 * lambda * sigma^2  -> dL/dsigma = lambda * sigma
    // sigma is your post-softplus sigma (>=0)
    float dsd = SUPPRESS_SIGMA_LAMBDA * sigma;
    dL_dsigma[s_id] = dsd;

    // ensure dL_dc is zero (so color path produces zero deltas)
    dL_dc[s_id*3 + 0] = 0.0f;
    dL_dc[s_id*3 + 1] = 0.0f;
    dL_dc[s_id*3 + 2] = 0.0f;

    // out_rgb is not meaningful in suppress mode; you can leave it unchanged or set something debuggy
    if (s_id == 0) {
        // optional debug
        // out_rgb[ray_id].w = 1234.0f;
    }

#endif

    //if (s_id == 0) out_rgb[ray_id].w = (float)2*do_train;  // or += 1000*do_train
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    if (do_train == 0) return;

    // ------------------------------------------------------------
    // 5-7) Deltas + Backprop + Save (in-place reuse of activations)
    // ------------------------------------------------------------
    float* dcur = A;
    float* dnxt = B;

IF_TRAINING
{
    // ---------- 5) output deltas (small) ----------
    float delta_cout[C_OUT];
    float delta_dout[D_OUT];

#if (SURPRESS_SAMPLES == 0)
    {
        float t  = Ts[s_id];
        float dtl = DTs[s_id];
        float D = 0.5f * length(box_max - box_min);

        float num = t*t + t*dtl + (dtl*dtl/3.0f);
        float den = D*D + D*dtl + (dtl*dtl/3.0f);

        float x = t / fmax(D, 1.0e-12f);
        float s = (x >= 1.0f) ? 0.0f : 1.0f;
        float gate = (1.0f - s) * 1.0f + s * (pow(fmax(x, 1.0e-12f), 0.2f));
        float f_vol = gate * num / fmax(den, 1.0e-12f);

        float d_sigma = f_vol * dL_dsigma[s_id];
        float d_r = f_vol * dL_dc[s_id*3 + 0];
        float d_g = f_vol * dL_dc[s_id*3 + 1];
        float d_b = f_vol * dL_dc[s_id*3 + 2];

        // dL/d(z_cout)
        delta_cout[0] = d_r * sigmoid_soft_prime_from_sigmoid(rgb.x);
        delta_cout[1] = d_g * sigmoid_soft_prime_from_sigmoid(rgb.y);
        delta_cout[2] = d_b * sigmoid_soft_prime_from_sigmoid(rgb.z);

        // density head only on dim0 initially
        delta_dout[0] = SIGMA_LR_RATE * d_sigma *
                        softplus_prime_from_pre(clamp(a_cin[0], -18.0f, 18.0f));
        #pragma unroll
        for (int i = 1; i < D_OUT; ++i) delta_dout[i] = 0.0f;
    }

#else
    // suppress mode: no color learning signal
    delta_cout[0] = 0.0f;
    delta_cout[1] = 0.0f;
    delta_cout[2] = 0.0f;

    // sigma-only learning signal:
    // dL/dsigma already computed as SUPPRESS_SIGMA_LAMBDA * sigma
    float d_sigma = dL_dsigma[s_id];

    // z_sigma_pre = a_cin[0] (your dout[0] pre-softplus, clamped earlier in forward)
    float z = clamp(a_cin[0], -18.0f, 18.0f);

    delta_dout[0] = SIGMA_LR_RATE * d_sigma * softplus_prime_from_pre(z);
    #pragma unroll
    for (int i = 1; i < D_OUT; ++i) delta_dout[i] = 0.0f;

#endif

    // Save color output delta (always written; zeros in suppress mode)
    #pragma unroll
    for (int i = 0; i < C_OUT; ++i) tstore(gl_dc_out, i*Btot + batch, delta_cout[i]);

    // ---------- 6) backprop, TODO reusing activation arrays ----------
    // --------------------------------------------------
    // Backprop: Color MLP
    // Inputs:
    //   delta_cout[C_OUT]
    //   cmask1..cmask4 (if USE_RELU_MASK)
    // Outputs:
    //   stores gl_dcK for K=1..NC (each M)
    //   accumulates into delta_dout[0..D_OUT-1]
    // --------------------------------------------------
    // float dA[M];
    // float dB[M];
    // float* dcur = dA;
    // float* dnxt = dB;
#if (SURPRESS_SAMPLES == 0)
    // 6.1) delta at last hidden = Wc_out^T * delta_cout, then * relu'(z_last)
    bwd_linear_to_M(dcur, delta_cout, C_OUT, Wc_out);

    #if (NC == 1)
        for (int j=0;j<M;++j) dcur[j] = relu_mask_apply(cmask1, j, dcur[j]);
    #elif (NC == 2)
        for (int j=0;j<M;++j) dcur[j] = relu_mask_apply(cmask2, j, dcur[j]);
    #elif (NC == 3)
        for (int j=0;j<M;++j) dcur[j] = relu_mask_apply(cmask3, j, dcur[j]);
    #else // NC==4
        for (int j=0;j<M;++j) dcur[j] = relu_mask_apply(cmask4, j, dcur[j]);
    #endif

    // store delta for last hidden as gl_dcNC
    #if (NC == 1)
        for (int i=0;i<M;++i) tstore(gl_dc1, i*Btot + batch, dcur[i]);
    #elif (NC == 2)
        for (int i=0;i<M;++i) tstore(gl_dc2, i*Btot + batch, dcur[i]);
    #elif (NC == 3)
        for (int i=0;i<M;++i) tstore(gl_dc3, i*Btot + batch, dcur[i]);
    #else
        for (int i=0;i<M;++i) tstore(gl_dc4, i*Btot + batch, dcur[i]);
    #endif

    // 6.2) step back through M->M layers until we reach hidden1
    #if (NC >= 2)
        // from hidden NC to hidden NC-1 via Wc_(NC-2)?? (see mapping below)
        // NC=2: hidden2 came from Wc_01
        // NC=3: hidden3 came from Wc_12
        // NC=4: hidden4 came from Wc_23

        #if (NC == 2)
            bwd_mm_to_M(dnxt, dcur, Wc_01);
            for (int j=0;j<M;++j) dnxt[j] = relu_mask_apply(cmask1, j, dnxt[j]);
            { float* tmp=dcur; dcur=dnxt; dnxt=tmp; }
            for (int i=0;i<M;++i) tstore(gl_dc1, i*Btot + batch, dcur[i]);
        #elif (NC == 3)
            // hidden3 -> hidden2
            bwd_mm_to_M(dnxt, dcur, Wc_12);
            for (int j=0;j<M;++j) dnxt[j] = relu_mask_apply(cmask2, j, dnxt[j]);
            { float* tmp=dcur; dcur=dnxt; dnxt=tmp; }
            for (int i=0;i<M;++i) tstore(gl_dc2, i*Btot + batch, dcur[i]);

            // hidden2 -> hidden1
            bwd_mm_to_M(dnxt, dcur, Wc_01);
            for (int j=0;j<M;++j) dnxt[j] = relu_mask_apply(cmask1, j, dnxt[j]);
            { float* tmp=dcur; dcur=dnxt; dnxt=tmp; }
            for (int i=0;i<M;++i) tstore(gl_dc1, i*Btot + batch, dcur[i]);

        #else // NC == 4
            // hidden4 -> hidden3
            bwd_mm_to_M(dnxt, dcur, Wc_23);
            for (int j=0;j<M;++j) dnxt[j] = relu_mask_apply(cmask3, j, dnxt[j]);
            { float* tmp=dcur; dcur=dnxt; dnxt=tmp; }
            for (int i=0;i<M;++i) tstore(gl_dc3, i*Btot + batch, dcur[i]);

            // hidden3 -> hidden2
            bwd_mm_to_M(dnxt, dcur, Wc_12);
            for (int j=0;j<M;++j) dnxt[j] = relu_mask_apply(cmask2, j, dnxt[j]);
            { float* tmp=dcur; dcur=dnxt; dnxt=tmp; }
            for (int i=0;i<M;++i) tstore(gl_dc2, i*Btot + batch, dcur[i]);

            // hidden2 -> hidden1
            bwd_mm_to_M(dnxt, dcur, Wc_01);
            for (int j=0;j<M;++j) dnxt[j] = relu_mask_apply(cmask1, j, dnxt[j]);
            { float* tmp=dcur; dcur=dnxt; dnxt=tmp; }
            for (int i=0;i<M;++i) tstore(gl_dc1, i*Btot + batch, dcur[i]);
        #endif
    #endif

#else
    // suppress mode: zero deltas for all color hidden layers
    for (int i = 0; i < M; ++i) {
        tstore(gl_dc1, i*Btot + batch, 0.0f);
        tstore(gl_dc2, i*Btot + batch, 0.0f);
        tstore(gl_dc3, i*Btot + batch, 0.0f);
        tstore(gl_dc4, i*Btot + batch, 0.0f);
    }
    // do NOT add color contribution into delta_dout[]
#endif

#if (SURPRESS_SAMPLES == 0)
    // 6.3) Now dcur == delta at hidden1.
    // Accumulate into delta_dout[i] for i in 0..D_OUT-1 from the first layer Wc_in (M x C_IN).
    // We only take the first D_OUT inputs (skip SH part).
    for (int i = 0; i < D_OUT; ++i) {
        float s = 0.0f;
        for (int j = 0; j < M; ++j) s += i32_to_f(Wc_in[j*C_IN + i]) * dcur[j];
        delta_dout[i] += s;
    }
#endif

    // --------------------------------------------------
    // Backprop: Density MLP
    // Inputs:
    //   delta_dout[D_OUT] (already includes sigma head + color contribution)
    //   dmask1..dmask3 (if USE_RELU_MASK)
    // Outputs:
    //   stores gl_dd_out (D_OUT) and gl_ddK for K=1..ND (each M)
    //   leaves delta_h1 in dcur (for hash update)
    // --------------------------------------------------

    // store delta_dout now (matches your current behavior)
    for (int i = 0; i < D_OUT; ++i) tstore(gl_dd_out, i*Btot + batch, delta_dout[i]);

    // start from last hidden: dcur = Wd_out^T * delta_dout
    bwd_linear_to_M(dcur, delta_dout, D_OUT, Wd_out);

    // apply relu' for last hidden
    #if (ND == 1)
        for (int j=0;j<M;++j) dcur[j] = relu_mask_apply(dmask1, j, dcur[j]);
    #elif (ND == 2)
        for (int j=0;j<M;++j) dcur[j] = relu_mask_apply(dmask2, j, dcur[j]);
    #else // ND==3
        for (int j=0;j<M;++j) dcur[j] = relu_mask_apply(dmask3, j, dcur[j]);
    #endif

    // store delta for last hidden as gl_ddND
    #if (ND == 1)
        for (int i=0;i<M;++i) tstore(gl_dd1, i*Btot + batch, dcur[i]);
    #elif (ND == 2)
        for (int i=0;i<M;++i) tstore(gl_dd2, i*Btot + batch, dcur[i]);
    #else
        for (int i=0;i<M;++i) tstore(gl_dd3, i*Btot + batch, dcur[i]);
    #endif

    #if (ND >= 2)
        #if (ND == 2)
            // hidden2 -> hidden1 via Wd_01
            bwd_mm_to_M(dnxt, dcur, Wd_01);
            for (int j=0;j<M;++j) dnxt[j] = relu_mask_apply(dmask1, j, dnxt[j]);
            { float* tmp=dcur; dcur=dnxt; dnxt=tmp; }
            for (int i=0;i<M;++i) tstore(gl_dd1, i*Btot + batch, dcur[i]);

        #else // ND == 3
            // hidden3 -> hidden2 via Wd_12
            bwd_mm_to_M(dnxt, dcur, Wd_12);
            for (int j=0;j<M;++j) dnxt[j] = relu_mask_apply(dmask2, j, dnxt[j]);
            { float* tmp=dcur; dcur=dnxt; dnxt=tmp; }
            for (int i=0;i<M;++i) tstore(gl_dd2, i*Btot + batch, dcur[i]);

            // hidden2 -> hidden1 via Wd_01
            bwd_mm_to_M(dnxt, dcur, Wd_01);
            for (int j=0;j<M;++j) dnxt[j] = relu_mask_apply(dmask1, j, dnxt[j]);
            { float* tmp=dcur; dcur=dnxt; dnxt=tmp; }
            for (int i=0;i<M;++i) tstore(gl_dd1, i*Btot + batch, dcur[i]);
        #endif
    #endif

    // Now dcur is delta at density hidden1.
    // Use dcur[j] in hash update as delta_d1.
}

    // ------------------------------------------------------------
    // 8) Update hash table entries calculating delta_pos and using feature_w/feature_idx
    //    lr_lvl sets the per-level LR rolloff. The sample-volume correction is
    //    already carried by the backpropagated deltas via f_vol in section 5.
    //    Distribute the encoding gradient to 8 corners.
    // ------------------------------------------------------------
    if (table) {
        if (s_id == 0) out_rgb[ray_id].w += (table != 0) ? 4.0f : 0.0f;

        const float R0_f   = RESOLUTION_LVL0;
        const float Rmax_f = RESOLUTION_LVLMAX;

        float3 box_size = box_max - box_min;
        float3 p = (x - box_min) / box_size;
        const float eps = 1.0e-6f;
        p = clamp(p, 0.0f, 1.0f - eps);

        #define HASH_ENTRY(ix,iy,iz,lvl) \
            (hash_u32( \
                (uint)(ix) * 73856093u ^ \
                (uint)(iy) * 19349663u ^ \
                (uint)(iz) * 83492791u ^ \
                (uint)(lvl) * 2654435761u) & level_mask(lvl))

        for (int lvl = 0; lvl < N_LEVELS; ++lvl) {

            float e = HASH_W_LVL_SCALE_EXPONENT;
            float lr_lvl = lr * pow(2.0f, -(float)lvl * e);

            // --- same per-level coordinate math as forward ---
            float t_lvl = (float)lvl * (1.0f / (float)(N_LEVELS - 1));
            float Rf = R0_f * pow(Rmax_f / R0_f, t_lvl);
            int R = (int)rint(Rf);
            if (R < 2) R = 2;

            float3 u = p * (float)R;

            int ix0 = (int)floor(u.x);
            int iy0 = (int)floor(u.y);
            int iz0 = (int)floor(u.z);

            ix0 = clamp(ix0, 0, R - 2);
            iy0 = clamp(iy0, 0, R - 2);
            iz0 = clamp(iz0, 0, R - 2);

            float fx = u.x - (float)ix0;
            float fy = u.y - (float)iy0;
            float fz = u.z - (float)iz0;

            float wx0 = 1.0f - fx, wx1 = fx;
            float wy0 = 1.0f - fy, wy1 = fy;
            float wz0 = 1.0f - fz, wz1 = fz;

            // Tiny per-level cache
            float w_corner[8];
            int   base_feat_corner[8];

            #pragma unroll
            for (int c = 0; c < 8; ++c) {
                int dx = (c & 1) ? 1 : 0;
                int dy = (c & 2) ? 1 : 0;
                int dz = (c & 4) ? 1 : 0;

                float w = (dx ? wx1 : wx0) * (dy ? wy1 : wy0) * (dz ? wz1 : wz0);
                w_corner[c] = w;

                uint entry = HASH_ENTRY(ix0 + dx, iy0 + dy, iz0 + dz, lvl);
                uint base_entry = LEVEL_OFFSET_ENTRIES[lvl] + entry;
                base_feat_corner[c] = (int)(base_entry * (uint)F_PER_LEVEL);
            }

            #pragma unroll
            for (int f = 0; f < F_PER_LEVEL; ++f) {
                int k = lvl * F_PER_LEVEL + f;
                float gf = 0.0f;
                for (int j = 0; j < M; ++j) 
                    gf += i32_to_f(Wd_in[j*L_POS + k]) * dcur[j]; // contains delta_d1

                if (isfinite(gf)) {
                    #pragma unroll
                    for (int c = 0; c < 8; ++c) {
                        float w = w_corner[c];
                        int   i = base_feat_corner[c] + f;
                        atomic_add_f32_global_i32(&table[i], -lr_lvl * HASH_LR_RATE * gf * w);
                    }
                }
            }
        }
        #undef HASH_ENTRY
    }
}

