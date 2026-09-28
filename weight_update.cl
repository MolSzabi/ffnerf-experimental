// weight_update.cl (OpenCL C 1.2)
//
// Multi work-group weight update kernel.
// Each work-group computes exactly one (OUT tile, IN tile) for EACH layer, using:
//   tile_in  = get_group_id(0)
//   tile_out = get_group_id(1)
//   in_base  = tile_in  * INT
//   out_base = tile_out * OUTT
//
// Weights/biases are float bits in int buffers (__global int*).
// Adam state is float buffers.
//
// Launch with a 2D NDRange:
//   local  = (WG_X, WG_Y)
//   global = (num_groups_x * WG_X, num_groups_y * WG_Y)
//
// Where num_groups_x/num_groups_y cover the max (IN, OUT) across layers.
// Work-groups that fall outside a particular layer’s bounds early-return for that layer.

#ifndef WG_X
#define WG_X 16
#endif
#ifndef WG_Y
#define WG_Y 16
#endif

#ifndef REG_X
#define REG_X 2   // per-thread columns
#endif
#ifndef REG_Y
#define REG_Y 2   // per-thread rows
#endif

#ifndef TILEK
#define TILEK 64
#endif

#define INT  (WG_X * REG_X)
#define OUTT (WG_Y * REG_Y)

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

#define L_DIR   16
#define D_OUT   16
#define C_IN    (D_OUT + L_DIR) // 32
#define C_OUT   3

// Scratch sizes for one tile
#define SCRATCH0_SIZE (OUTT*TILEK + OUTT*INT) // lD + lC
#define SCRATCH1_SIZE (INT*TILEK)             // lA
// (x+y)*T+y*x

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

#if (TRANSFER_FMT == 8)
    typedef uchar TRANSFER_T;

    static inline float fp8_sat(float x) {
        if (!isfinite(x)) return copysign(15.5f, x);
        return clamp(x, -15.5f, 15.5f);
    }

    static inline uchar pack_fp8_e4m3(float x_in) {
        float x = fp8_sat(x_in);
        uint s = (x < 0.0f) ? 0x80u : 0u;
        float ax = fabs(x);
        if (ax < 1.0e-8f) return (uchar)s;

        int e;
        float m = frexp(ax, &e);      // ax = m * 2^e, m in [0.5,1)
        float m2 = m * 2.0f;          // in [1,2)
        int e2 = e - 1;

        const int bias = 7;
        int ef = e2 + bias;

        if (ef <= 0) return (uchar)s;
        if (ef > 14) { ef = 10; return (uchar)(s | ((uint)ef << 3) | 7u); }

        float frac = m2 - 1.0f;
        int mant = (int)rint(frac * 8.0f); // 0..8
        if (mant >= 8) { mant = 0; ef += 1; }
        if (ef > 14) { ef = 10; mant = 7; }

        return (uchar)(s | ((uint)ef << 3) | ((uint)mant & 7u));
    }

    static inline float unpack_fp8_e4m3(uchar b) {
        uint s  = (b & 0x80u);
        uint ef = (b >> 3) & 0x0Fu;
        uint mf = (b & 0x07u);

        if (ef == 0u) return s ? -0.0f : 0.0f;

        const int bias = 7;
        int e2 = (int)ef - bias;
        float mant = (float)mf * (1.0f / 8.0f);
        float v = ldexp(1.0f + mant, e2);
        return s ? -v : v;
    }

    static inline float tload(const __global TRANSFER_T* p, int idx) { return unpack_fp8_e4m3(p[idx]); }
    static inline void  tstore(__global TRANSFER_T* p, int idx, float x) { p[idx] = pack_fp8_e4m3(x); }

#elif (TRANSFER_FMT == 16)
    typedef half TRANSFER_T;

    static inline float fp16_sat(float x) {
        if (!isfinite(x)) return copysign(65504.0f, x);
        return clamp(x, -65504.0f, 65504.0f);
    }

    static inline float tload(const __global TRANSFER_T* p, int idx) { return convert_float(p[idx]); }
    static inline void  tstore(__global TRANSFER_T* p, int idx, float x) { p[idx] = convert_half_rte(fp16_sat(x)); }

#else  // TRANSFER_FMT == 32
    typedef float TRANSFER_T;
    static inline float tload(const __global TRANSFER_T* p, int idx) { return p[idx]; }
    static inline void  tstore(__global TRANSFER_T* p, int idx, float x) { p[idx] = x; }
#endif

// ============================================================
// Param storage: weights/biases are float bits in int buffers
// ============================================================
static inline float load_fbits(__global const int* p, int idx) {
    return as_float(p[idx]);
}
static inline void store_fbits(__global int* p, int idx, float x) {
    // store float bits into int
    p[idx] = as_int(as_uint(x));
}

// ------------------------------------------------------------
// Adam update for a single parameter (param in int-fbits buffer).
// g is raw gradient (NOT negated).
// lr_t can be bias-corrected by host (recommended).
// ------------------------------------------------------------
static inline void adam_update_1_i32(
    __global int* p_bits,
    __global float* m,
    __global float* v,
    int idx,
    float g,
    float lr_t,
    float beta1,
    float beta2,
    float eps
){
    float mi = m[idx];
    float vi = v[idx];

    mi = fma((1.0f - beta1), g, beta1 * mi);
    vi = fma((1.0f - beta2), g*g, beta2 * vi);

    float p = load_fbits(p_bits, idx);
    float denom = native_sqrt(vi) + eps;
    p -= lr_t * (mi / denom);
    store_fbits(p_bits, idx, p);

    m[idx] = mi;
    v[idx] = vi;
}

// ------------------------------------------------------------
// One-tile update for OUTxIN.
// This WG computes gradients for the tile (out_base..out_base+OUTT-1,
// in_base..in_base+INT-1), sums over B, then applies Adam updates.
//
// Bias update: only when (in_base == 0 && tx == 0), so exactly one WG per out tile.
// ------------------------------------------------------------
static inline void update_layer_tile_adam_i32(
    __global int* W_bits,         // [OUT*IN] row-major float-bits
    __global int* b_bits,         // [OUT] or NULL (float-bits)

    __global float* mW,           // [OUT*IN]
    __global float* vW,           // [OUT*IN]
    __global float* mb,           // [OUT] or NULL
    __global float* vb,           // [OUT] or NULL

    __global const TRANSFER_T* d, // [OUT*B] SoA: d[out*B + batch]
    __global const TRANSFER_T* a, // [IN *B] SoA: a[in *B + batch]
    int OUT,
    int IN,
    int B,

    int out_base,
    int in_base,

    float lr_t,
    float beta1,
    float beta2,
    float eps,

    __local float* scratch0,      // OUTT*TILEK + OUTT*INT
    __local float* scratch1       // INT*TILEK
){
    const int tx  = (int)get_local_id(0);
    const int ty  = (int)get_local_id(1);
    const int lid = ty * WG_X + tx;
    const int lsize = WG_X * WG_Y;

    // NOTE: out_base/in_base are uniform across the WG.
    // Early return is safe because either all threads return or none do.
    if (out_base >= OUT || in_base >= IN) return;

    __local float* lD = scratch0;                      // OUTT*TILEK
    __local float* lC = scratch0 + (OUTT * TILEK);     // OUTT*INT
    __local float* lA = scratch1;                      // INT*TILEK

    const int out_l0 = ty * REG_Y;
    const int in_l0  = tx * REG_X;

    // zero C tile
    for (int idx = lid; idx < OUTT * INT; idx += lsize) lC[idx] = 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    // bias accumulators (only used by threads tx==0; still harmless for others)
    float bacc[REG_Y];
    for (int ry = 0; ry < REG_Y; ++ry) bacc[ry] = 0.0f;

    for (int k_base = 0; k_base < B; k_base += TILEK)
    {
        // load D tile (OUTT x TILEK)
        for (int idx = lid; idx < OUTT * TILEK; idx += lsize) {
            int r = idx / TILEK;
            int c = idx - r*TILEK;
            int go = out_base + r;
            int gk = k_base + c;
            lD[idx] = (go < OUT && gk < B) ? tload(d, go*B + gk) : 0.0f;
        }

        // load A tile (INT x TILEK)
        for (int idx = lid; idx < INT * TILEK; idx += lsize) {
            int r = idx / TILEK;
            int c = idx - r*TILEK;
            int gi = in_base + r;
            int gk = k_base + c;
            lA[idx] = (gi < IN && gk < B) ? tload(a, gi*B + gk) : 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        float acc[REG_Y * REG_X];
        for (int i = 0; i < REG_Y * REG_X; ++i) acc[i] = 0.0f;

        for (int k = 0; k < TILEK; ++k) {
            float av[REG_X];
            for (int rx = 0; rx < REG_X; ++rx) {
                int ii = in_l0 + rx;
                av[rx] = lA[ii * TILEK + k];
            }

            for (int ry = 0; ry < REG_Y; ++ry) {
                int oo = out_l0 + ry;
                float dv = lD[oo * TILEK + k];

                if (tx == 0) bacc[ry] += dv;

                for (int rx = 0; rx < REG_X; ++rx) {
                    acc[ry*REG_X + rx] += dv * av[rx];
                }
            }
        }

        // write into C tile (unique ownership)
        for (int ry = 0; ry < REG_Y; ++ry) {
            int oo = out_l0 + ry;
            int base = oo * INT + in_l0;
            for (int rx = 0; rx < REG_X; ++rx) {
                lC[base + rx] += acc[ry*REG_X + rx];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Update weights
    for (int ry = 0; ry < REG_Y; ++ry) {
        int go = out_base + out_l0 + ry;
        if (go >= OUT) continue;

        int cbase = (out_l0 + ry) * INT + in_l0;

        for (int rx = 0; rx < REG_X; ++rx) {
            int gi = in_base + in_l0 + rx;
            if (gi >= IN) continue;

            int widx = go * IN + gi;
            float g  = lC[cbase + rx];
            adam_update_1_i32(W_bits, mW, vW, widx, g, lr_t, beta1, beta2, eps);
        }
    }

    // Update biases once per OUT row: only when in_base==0 and tx==0
    if (b_bits && mb && vb && tx == 0 && in_base == 0)
    {
        for (int ry = 0; ry < REG_Y; ++ry) {
            int go = out_base + out_l0 + ry;
            if (go < OUT) adam_update_1_i32(b_bits, mb, vb, go, bacc[ry], lr_t, beta1, beta2, eps);
        }
    }
}

// ============================================================
// Kernel entry (matches your host set_args ordering exactly)
// ============================================================
__kernel void update_ffnerf_oneWG(
    // Weights (float bits in int) -- fixed args (max)
    __global int* Wd_in,    // [M * L]         (M×L_POS)
    __global int* Wd_01,    // [M * M]         (if ND>=2 else placeholder)
    __global int* Wd_12,    // [M * M]         (if ND>=3 else placeholder)
    __global int* Wd_out,   // [D_OUT * M]

    __global int* Wc_in,    // [M * C_IN]
    __global int* Wc_01,    // [M * M]         (if NC>=2 else placeholder)
    __global int* Wc_12,    // [M * M]         (if NC>=3 else placeholder)
    __global int* Wc_23,    // [M * M]         (if NC>=4 else placeholder)
    __global int* Wc_out,   // [C_OUT * M]

    // Biases (float bits in int) -- fixed args (max)
    __global int* bd_0,     // [M]
    __global int* bd_1,     // [M]             (if ND>=2 else placeholder)
    __global int* bd_2,     // [M]             (if ND>=3 else placeholder)
    __global int* bd_out,   // [D_OUT]

    __global int* bc_0,     // [M]
    __global int* bc_1,     // [M]             (if NC>=2 else placeholder)
    __global int* bc_2,     // [M]             (if NC>=3 else placeholder)
    __global int* bc_3,     // [M]             (if NC>=4 else placeholder)
    __global int* bc_out,   // [C_OUT]

    // Adam m for weights (same order as weights)
    __global float* mWd_in,
    __global float* mWd_01,
    __global float* mWd_12,
    __global float* mWd_out,

    __global float* mWc_in,
    __global float* mWc_01,
    __global float* mWc_12,
    __global float* mWc_23,
    __global float* mWc_out,

    // Adam v for weights (same order as weights)
    __global float* vWd_in,
    __global float* vWd_01,
    __global float* vWd_12,
    __global float* vWd_out,

    __global float* vWc_in,
    __global float* vWc_01,
    __global float* vWc_12,
    __global float* vWc_23,
    __global float* vWc_out,

    // Adam m for biases (same order as biases)
    __global float* mbd_0,
    __global float* mbd_1,
    __global float* mbd_2,
    __global float* mbd_out,

    __global float* mbc_0,
    __global float* mbc_1,
    __global float* mbc_2,
    __global float* mbc_3,
    __global float* mbc_out,

    // Adam v for biases (same order as biases)
    __global float* vbd_0,
    __global float* vbd_1,
    __global float* vbd_2,
    __global float* vbd_out,

    __global float* vbc_0,
    __global float* vbc_1,
    __global float* vbc_2,
    __global float* vbc_3,
    __global float* vbc_out,

    // Deltas (SoA: node-major then batch) -- MUST match ffnerf scratch order
    __global TRANSFER_T* dd1,      // [M * Btot]   (if ND>=1)
    __global TRANSFER_T* dd2,      // [M * Btot]   (if ND>=2 else placeholder)
    __global TRANSFER_T* dd3,      // [M * Btot]   (if ND>=3 else placeholder)
    __global TRANSFER_T* dd_out,   // [D_OUT * Btot]

    __global TRANSFER_T* dc1,      // [M * Btot]   (if NC>=1)
    __global TRANSFER_T* dc2,      // [M * Btot]   (if NC>=2 else placeholder)
    __global TRANSFER_T* dc3,      // [M * Btot]   (if NC>=3 else placeholder)
    __global TRANSFER_T* dc4,      // [M * Btot]   (if NC>=4 else placeholder)
    __global TRANSFER_T* dc_out,   // [C_OUT * Btot]

    // Activations (not-last) -- MUST match ffnerf scratch order
    __global TRANSFER_T* apos,     // [L * Btot]
    __global TRANSFER_T* ad1,      // [M * Btot]   (if ND>=1)
    __global TRANSFER_T* ad2,      // [M * Btot]   (if ND>=2 else placeholder)
    __global TRANSFER_T* ad3,      // [M * Btot]   (if ND>=3 else placeholder)

    __global TRANSFER_T* ac_in,    // [C_IN * Btot]
    __global TRANSFER_T* ac1,      // [M * Btot]   (if NC>=1)
    __global TRANSFER_T* ac2,      // [M * Btot]   (if NC>=2 else placeholder)
    __global TRANSFER_T* ac3,      // [M * Btot]   (if NC>=3 else placeholder)
    __global TRANSFER_T* ac4,      // [M * Btot]   (if NC>=4 else placeholder)

    // Sizes
    int B,
    int L,
    int M,

    // Adam hyperparams
    float lr_t,
    float color_MLP_rel_lr,
    float beta1,
    float beta2,
    float eps
){
    __local float scratch0[SCRATCH0_SIZE];
    __local float scratch1[SCRATCH1_SIZE];

    int tile_in  = (int)get_group_id(0);
    int tile_out = (int)get_group_id(1);

    int in_base  = tile_in  * INT;
    int out_base = tile_out * OUTT;

    // ================================
    // --------------- Color MLP ---------------
    // ================================
    // Pick last hidden activation for output heads
    // -------------------------
    #if (NC >= 4)
        #define AC_LAST ac4
    #elif (NC == 3)
        #define AC_LAST ac3
    #elif (NC == 2)
        #define AC_LAST ac2
    #else
        #define AC_LAST ac1
    #endif

    #if (ND >= 3)
        #define AD_LAST ad3
    #elif (ND == 2)
        #define AD_LAST ad2
    #else
        #define AD_LAST ad1
    #endif

    update_layer_tile_adam_i32(
        Wc_out, bc_out, mWc_out, vWc_out, mbc_out, vbc_out,
        dc_out, AC_LAST, C_OUT, M, B,
        out_base, in_base,
        lr_t*color_MLP_rel_lr, beta1, beta2, eps,
        scratch0, scratch1
    );

    #if (NC >= 4)
        update_layer_tile_adam_i32(
            Wc_23, bc_3, mWc_23, vWc_23, mbc_3, vbc_3,
            dc4, ac3, M, M, B,
            out_base, in_base,
            lr_t*color_MLP_rel_lr, beta1, beta2, eps,
            scratch0, scratch1
        );
    #endif

    #if (NC >= 3)
        update_layer_tile_adam_i32(
            Wc_12, bc_2, mWc_12, vWc_12, mbc_2, vbc_2,
            dc3, ac2, M, M, B,
            out_base, in_base,
            lr_t*color_MLP_rel_lr, beta1, beta2, eps,
            scratch0, scratch1
        );
    #endif

    #if (NC >= 2)
        update_layer_tile_adam_i32(
            Wc_01, bc_1, mWc_01, vWc_01, mbc_1, vbc_1,
            dc2, ac1, M, M, B,
            out_base, in_base,
            lr_t*color_MLP_rel_lr, beta1, beta2, eps,
            scratch0, scratch1
        );
    #endif

    #if (NC >= 1)
        update_layer_tile_adam_i32(
            Wc_in, bc_0, mWc_in, vWc_in, mbc_0, vbc_0,
            dc1, ac_in, M, C_IN, B,
            out_base, in_base,
            lr_t*color_MLP_rel_lr, beta1, beta2, eps,
            scratch0, scratch1
        );
    #endif


        // ---------------- Density MLP ----------------
        update_layer_tile_adam_i32(
            Wd_out, bd_out, mWd_out, vWd_out, mbd_out, vbd_out,
            dd_out, AD_LAST, D_OUT, M, B,
            out_base, in_base,
            lr_t, beta1, beta2, eps,
            scratch0, scratch1
        );

    #if (ND >= 3)
        update_layer_tile_adam_i32(
            Wd_12, bd_2, mWd_12, vWd_12, mbd_2, vbd_2,
            dd3, ad2, M, M, B,
            out_base, in_base,
            lr_t, beta1, beta2, eps,
            scratch0, scratch1
        );
    #endif

    #if (ND >= 2)
        update_layer_tile_adam_i32(
            Wd_01, bd_1, mWd_01, vWd_01, mbd_1, vbd_1,
            dd2, ad1, M, M, B,
            out_base, in_base,
            lr_t, beta1, beta2, eps,
            scratch0, scratch1
        );
    #endif

    #if (ND >= 1)
        update_layer_tile_adam_i32(
            Wd_in, bd_0, mWd_in, vWd_in, mbd_0, vbd_0,
            dd1, apos, M, L, B,
            out_base, in_base,
            lr_t, beta1, beta2, eps,
            scratch0, scratch1
        );
    #endif
    #undef AC_LAST
    #undef AD_LAST

}
