// decay_occ_grid_in-st.cl
static inline uint hash_u32(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// decay_q24 = round(decay * 2^24), clamped to [0..2^24]
__kernel void occ_nibble_decay_write_both_q24(
    __global uint* __restrict occ_grid_in,   // in/out (old values in, new values out)
    __global uint* __restrict occ_grid_st,   // in/out
    const uint n_words,
    const uint decay_q24,
    const uint iter_seed
){
    const uint gid = get_global_id(0);
    if (gid >= n_words) return;

    const uint win = occ_grid_in[gid];
    const uint wst = occ_grid_st[gid];

    // Fast paths
    if (wst == win)
    {
        // All nibbles unchanged => decay all (same as earlier kernel), then write both.
        uint out = wst;

        #pragma unroll
        for (uint s = 0; s < 8; ++s)
        {
            const uint shift = s * 4u;
            const uint mask  = 0xFu << shift;

            const uint old4 = (wst >> shift) & 0xFu;
            const uint q    = old4 + 1u; // 1..16

            uint q_after = q;

            if (q > 1u)
            {
                const uint qd24   = q * decay_q24;
                const uint qfloor = (qd24 >> 24);
                const uint frac24 = (qd24 & 0x00FFFFFFu);

                uint cand   = qfloor + 1u;
                uint thresh = 0x01000000u - frac24;

                const uint r   = hash_u32(iter_seed ^ (gid * 9781u) ^ (s * 6271u) ^ wst);
                const uint r24 = r & 0x00FFFFFFu;

                if (cand > 1u && r24 < thresh) cand -= 1u;
                if (cand < 1u) cand = 1u;
                if (cand > 16u) cand = 16u;

                q_after = cand;
            }

            const uint new4 = q_after - 1u;
            out = (out & ~mask) | ((new4 & 0xFu) << shift);
        }

        occ_grid_st[gid] = out;
        occ_grid_in[gid] = out;
        return;
    }

    const uint x = wst ^ win;
    if (x == 0xFFFFFFFFu) { /* no special meaning; just here to show no other fastpath */ }

    // Mixed case: decay only nibbles where (st nibble == in nibble)
    uint out = wst;

    #pragma unroll
    for (uint s = 0; s < 8; ++s)
    {
        const uint shift = s * 4u;

        // If xor nibble != 0, then values differ => leave alone
        if (((x >> shift) & 0xFu) != 0u) continue;

        const uint mask = 0xFu << shift;

        const uint old4 = (wst >> shift) & 0xFu;
        const uint q    = old4 + 1u; // 1..16

        if (q <= 1u) continue; // already minimum

        const uint qd24   = q * decay_q24;
        const uint qfloor = (qd24 >> 24);
        const uint frac24 = (qd24 & 0x00FFFFFFu);

        uint cand   = qfloor + 1u;
        uint thresh = 0x01000000u - frac24;

        const uint r   = hash_u32(iter_seed ^ (gid * 9781u) ^ (s * 6271u) ^ wst ^ (win * 1597u));
        const uint r24 = r & 0x00FFFFFFu;

        if (cand > 1u && r24 < thresh) cand -= 1u;

        if (cand < 1u) cand = 1u;
        if (cand > 16u) cand = 16u;

        const uint new4 = cand - 1u;
        out = (out & ~mask) | ((new4 & 0xFu) << shift);
    }

    occ_grid_st[gid] = out;
    occ_grid_in[gid] = out;
}
