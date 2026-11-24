#include "rel-bias-mask.cuh"

// one token, the bias computed where it is read instead of through a table:
//   sink[h]   = sum_i ws[i, h]*x[i]                       ws BF16 [n_embd, n_head]
//   out[kv,h] = mask[kv] + sum_k rb[k, idx[kv]]*r[k, h] - sink[h]
// one block per (head, chunk of keys): the block computes the sink of its head, then its keys
static constexpr int REL_DIRECT_THREADS = 256;

static __global__ void rel_bias_mask_direct_f16(
        const char * __restrict__ rb, const int64_t rb_nb1, const int n_rel,
        const char * __restrict__ r,  const int64_t r_nb1,
        const char * __restrict__ ws, const int64_t ws_nb1, const int n_embd,
        const float * __restrict__ x,
        const int32_t * __restrict__ idx,
        const float * __restrict__ mask,
        char * __restrict__ out, const int64_t o_nb2,
        const int n_kv) {
    __shared__ float red[REL_DIRECT_THREADS/WARP_SIZE];
    __shared__ float rh[64];
    const int h = blockIdx.y;

    const nv_bfloat16 * w = (const nv_bfloat16 * ) (ws + h*ws_nb1);
    float dot = 0.0f;
    for (int i = threadIdx.x; i < n_embd; i += REL_DIRECT_THREADS) {
        dot += __bfloat162float(w[i])*x[i];
    }
    dot = warp_reduce_sum(dot);
    if (threadIdx.x % WARP_SIZE == 0) {
        red[threadIdx.x / WARP_SIZE] = dot;
    }
    if (threadIdx.x < n_rel) {
        rh[threadIdx.x] = ((const float *) (r + h*r_nb1))[threadIdx.x];
    }
    __syncthreads();
    float sink = 0.0f;
#pragma unroll
    for (int k = 0; k < REL_DIRECT_THREADS/WARP_SIZE; ++k) {
        sink += red[k];
    }

    const int kv = blockIdx.x*REL_DIRECT_THREADS + threadIdx.x;
    if (kv >= n_kv) {
        return;
    }
    const float * col = (const float *) (rb + idx[kv]*rb_nb1);
    float b = 0.0f;
    for (int k = 0; k < n_rel; ++k) {
        b += col[k]*rh[k];
    }
    ((half *) (out + h*o_nb2))[kv] = __float2half(mask[kv] + (b - sink));
}

void ggml_cuda_op_rel_bias_mask_direct(ggml_backend_cuda_context & ctx,
        const ggml_tensor * rb, const ggml_tensor * r, const ggml_tensor * ws, const ggml_tensor * x,
        const ggml_tensor * idx, const ggml_tensor * mask, ggml_tensor * out) {
    GGML_ASSERT(rb->type == GGML_TYPE_F32 && r->type == GGML_TYPE_F32 && ws->type == GGML_TYPE_BF16 && x->type == GGML_TYPE_F32);
    GGML_ASSERT(idx->type == GGML_TYPE_I32 && mask->type == GGML_TYPE_F32 && out->type == GGML_TYPE_F16);
    GGML_ASSERT(rb->ne[0] <= 64 && r->ne[0] == rb->ne[0] && ws->ne[0] == x->ne[0] && ggml_is_contiguous(x) && ggml_is_contiguous(idx));
    GGML_ASSERT(out->ne[1] == 1 && out->ne[3] == 1 && mask->nb[0] == sizeof(float) && out->nb[0] == sizeof(half));

    const int n_kv   = (int) out->ne[0];
    const int n_head = (int) out->ne[2];
    const dim3 blocks((n_kv + REL_DIRECT_THREADS - 1)/REL_DIRECT_THREADS, n_head, 1);
    rel_bias_mask_direct_f16<<<blocks, REL_DIRECT_THREADS, 0, ctx.stream()>>>(
        (const char *) rb->data, rb->nb[1], (int) rb->ne[0],
        (const char *) r->data, r->nb[1],
        (const char *) ws->data, ws->nb[1], (int) ws->ne[0],
        (const float *) x->data,
        (const int32_t *) idx->data,
        (const float *) mask->data,
        (char *) out->data, out->nb[2],
        n_kv);
}
