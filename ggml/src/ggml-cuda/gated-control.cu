#include "gated-control.cuh"

// a low-rank gate on two vectors, for one token:
//   h   = down^T [rms_norm(cur); rms_norm(msg)]           down [2n, rank]
//   out = cur - tanh(forget^T h)*cur + tanh(gain^T h)*msg  forget, gain [rank, n]
// two kernels: the down projection needs all of both norms, the gated mix needs all of h

static constexpr int GATED_CONTROL_THREADS = 256;

template <int nthreads>
static __device__ float block_reduce_sum(float v, float * red) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    v = warp_reduce_sum(v);
    if (lane == 0) {
        red[warp] = v;
    }
    __syncthreads();
    v = lane < nthreads/WARP_SIZE ? red[lane] : 0.0f;
    v = warp_reduce_sum(v);
    __syncthreads();
    return v;
}

// one block per output r of the down projection; every block normalizes both inputs itself
static __global__ void gated_control_down_f32(
        const float * __restrict__ cur, const float * __restrict__ msg, const float eps,
        const char * __restrict__ down, const int64_t down_nb1,
        float * __restrict__ h, const int n) {
    __shared__ float red[32];
    const int r = blockIdx.x;

    float sc = 0.0f, sm = 0.0f;
    for (int i = threadIdx.x; i < n; i += GATED_CONTROL_THREADS) {
        sc += cur[i]*cur[i];
        sm += msg[i]*msg[i];
    }
    sc = block_reduce_sum<GATED_CONTROL_THREADS>(sc, red);
    sm = block_reduce_sum<GATED_CONTROL_THREADS>(sm, red);
    const float rc = rsqrtf(sc/n + eps);
    const float rm = rsqrtf(sm/n + eps);

    const float * w = (const float *) (down + r*down_nb1);
    float dot = 0.0f;
    for (int i = threadIdx.x; i < n; i += GATED_CONTROL_THREADS) {
        dot += w[i]*cur[i]*rc + w[n + i]*msg[i]*rm;
    }
    dot = block_reduce_sum<GATED_CONTROL_THREADS>(dot, red);
    if (threadIdx.x == 0) {
        h[r] = dot;
    }
}

// one warp per channel e: the forget and gain rows of e are contiguous over the rank, read coalesced by the lanes
static __global__ void gated_control_mix_f32(
        const float * __restrict__ cur, const float * __restrict__ msg, const float * __restrict__ h,
        const char * __restrict__ forget, const int64_t forget_nb1,
        const char * __restrict__ gain,   const int64_t gain_nb1,
        float * __restrict__ out, const int n, const int rank) {
    const int e    = blockIdx.x*(blockDim.x/WARP_SIZE) + threadIdx.x/WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    if (e >= n) {
        return;
    }
    const float * wf = (const float *) (forget + e*forget_nb1);
    const float * wg = (const float *) (gain   + e*gain_nb1);
    float f = 0.0f, g = 0.0f;
    for (int r = lane; r < rank; r += WARP_SIZE) {
        const float hr = h[r];
        f += wf[r]*hr;
        g += wg[r]*hr;
    }
    f = warp_reduce_sum(f);
    g = warp_reduce_sum(g);
    if (lane == 0) {
        out[e] = cur[e] - tanhf(f)*cur[e] + tanhf(g)*msg[e];
    }
}

void ggml_cuda_op_gated_control(ggml_backend_cuda_context & ctx,
        const ggml_tensor * cur, const ggml_tensor * msg, float eps,
        const ggml_tensor * down, ggml_tensor * h,
        const ggml_tensor * forget, const ggml_tensor * gain, ggml_tensor * out) {
    const int n    = (int) cur->ne[0];
    const int rank = (int) down->ne[1];

    GGML_ASSERT(ggml_is_contiguous(cur) && ggml_is_contiguous(msg) && ggml_is_contiguous(h) && ggml_is_contiguous(out));
    GGML_ASSERT(down->ne[0] == 2*n && forget->ne[0] == rank && gain->ne[0] == rank && forget->ne[1] == n && gain->ne[1] == n);

    cudaStream_t stream = ctx.stream();

    gated_control_down_f32<<<rank, GATED_CONTROL_THREADS, 0, stream>>>(
        (const float *) cur->data, (const float *) msg->data, eps,
        (const char *) down->data, down->nb[1],
        (float *) h->data, n);

    const int warps = 8;
    gated_control_mix_f32<<<(n + warps - 1)/warps, warps*WARP_SIZE, 0, stream>>>(
        (const float *) cur->data, (const float *) msg->data, (const float *) h->data,
        (const char *) forget->data, forget->nb[1],
        (const char *) gain->data,   gain->nb[1],
        (float *) out->data, n, rank);
}
