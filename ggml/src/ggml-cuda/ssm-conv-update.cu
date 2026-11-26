#include "ssm-conv-update.cuh"

// causal conv with its state update and a scaled residual, for few tokens per sequence:
//   x     = concat(past, cur^T)              past [d_conv - 1, nr, n_s], cur^T [n_t, nr, n_s]
//   state = x[n_t:]                          the last d_conv - 1 columns, channel-major per sequence
//   out   = scale*(res + ssm_conv(x, w))     res [nr, n_t*n_s]
// optionally plus a second residual after the scale (no norm then), or normalized per head of SSM_CONV_UPDATE_THREADS channels (l2_norm or rms_norm of the head)
// one thread per channel: the concat, the state copy, the conv, the add, the scale and the norm are one kernel
#define SSM_CONV_UPDATE_THREADS 128

template <int d_conv, int norm>
static __global__ void ssm_conv_update_res_f32(
        const char * __restrict__ past, const int64_t past_nb1, const int64_t past_nb2,
        const char * __restrict__ cur,  const int64_t cur_nb0,  const int64_t cur_nb1, const int64_t cur_nb2,
        const char * __restrict__ w,    const int64_t w_nb1,
        const char * __restrict__ res,  const int64_t res_nb1,
        const char * __restrict__ res2, const int64_t res2_nb1,
        const char * __restrict__ dyn,  const int64_t dyn_nb1, const int head_size,
        char       * __restrict__ state, const int kw, const int64_t slot_nb, const int64_t chan_nb, const int64_t seq_nb,
        char       * __restrict__ out,   const int64_t out_nb1,
        const int nr, const int n_t, const float scale, const float eps) {
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    const int s = blockIdx.y;
    __shared__ float red[SSM_CONV_UPDATE_THREADS/WARP_SIZE];

    const bool active = c < nr;

    float win[d_conv];
    float wc[d_conv];
#pragma unroll
    for (int j = 0; j < d_conv; ++j) {
        wc[j] = active ? ((const float *) (w + c*w_nb1))[j] : 0.0f;
    }
#pragma unroll
    for (int j = 0; j < d_conv - 1; ++j) {
        win[j] = active ? ((const float *) (past + c*past_nb1 + s*past_nb2))[j] : 0.0f;
    }

    for (int t = 0; t < n_t; ++t) {
        win[d_conv - 1] = active ? *(const float *) (cur + t*cur_nb0 + c*cur_nb1 + s*cur_nb2) : 0.0f;

        const int64_t tok = (int64_t) s*n_t + t;

        float sum = 0.0f;
        if (dyn) {
            // the input-dependent taps of the head: lag l reads the window d_conv - 1 - l back
            const float * tap = (const float *) (dyn + tok*dyn_nb1) + (c/head_size)*d_conv;
#pragma unroll
            for (int j = 0; j < d_conv; ++j) {
                sum += win[j]*(wc[j] + (active ? tap[d_conv - 1 - j] : 0.0f));
            }
        } else {
#pragma unroll
            for (int j = 0; j < d_conv; ++j) {
                sum += win[j]*wc[j];
            }
        }

        float y = active ? scale*(((const float *) (res + tok*res_nb1))[c] + sum) : 0.0f;

        if constexpr (norm != 0) {
            // the block is one head: reduce the squares over it
            float sq = warp_reduce_sum(y*y);
            if (threadIdx.x % WARP_SIZE == 0) {
                red[threadIdx.x / WARP_SIZE] = sq;
            }
            __syncthreads();
            sq = 0.0f;
#pragma unroll
            for (int k = 0; k < SSM_CONV_UPDATE_THREADS/WARP_SIZE; ++k) {
                sq += red[k];
            }
            __syncthreads();
            // l2_norm: x/max(|x|, eps), rms_norm: x/sqrt(mean(x^2) + eps)
            y *= norm == 1 ? rsqrtf(fmaxf(sq, eps*eps)) : rsqrtf(sq/SSM_CONV_UPDATE_THREADS + eps);
        }

        if (active) {
            if (res2) {
                y += ((const float *) (res2 + tok*res2_nb1))[c];
            }
            ((float *) (out + tok*out_nb1))[c] = y;
        }

#pragma unroll
        for (int j = 0; j < d_conv - 1; ++j) {
            win[j] = win[j + 1];
        }

        // the window after token t is rollback slot n_t - 1 - t, for the last kw tokens
        if (active && t >= n_t - kw) {
            float * st = (float *) (state + (n_t - 1 - t)*slot_nb + c*chan_nb + s*seq_nb);
#pragma unroll
            for (int j = 0; j < d_conv - 1; ++j) {
                st[j] = win[j];
            }
        }
    }
}

void ggml_cuda_op_ssm_conv_update_res(ggml_backend_cuda_context & ctx,
        const ggml_tensor * concat, const ggml_tensor * state_dst, const ggml_tensor * conv,
        const ggml_tensor * res, ggml_tensor * out, float scale, int norm, float eps, const ggml_tensor * res2,
        const ggml_tensor * dyn, int head_size, int kw, int64_t slot_nb, int64_t chan_nb, int64_t seq_nb) {
    const ggml_tensor * past = concat->src[0];
    const ggml_tensor * xt   = concat->src[1];
    const ggml_tensor * w    = conv->src[1];

    const int d_conv = (int) w->ne[0];
    const int nr     = (int) xt->ne[1];
    const int n_t    = (int) xt->ne[0];
    const int n_s    = (int) xt->ne[2];

    GGML_ASSERT(past->ne[0] == d_conv - 1 && past->ne[1] == nr && past->ne[2] == n_s);
    GGML_ASSERT(past->nb[0] == sizeof(float) && w->nb[0] == sizeof(float));
    GGML_ASSERT(res->nb[0] == sizeof(float) && out->nb[0] == sizeof(float));
    GGML_ASSERT(state_dst->type == GGML_TYPE_F32);
    GGML_ASSERT(norm == 0 || nr % SSM_CONV_UPDATE_THREADS == 0);

    // the token stride of out: [nr, n_tokens], or [head size, n_heads, n_tokens] after a head norm
    GGML_ASSERT(out->ne[0] == nr || out->ne[0]*out->ne[1] == nr);
    const int64_t out_nb_tok = out->ne[0] == nr ? out->nb[1] : out->nb[2];
    GGML_ASSERT(!res2 || res2->ne[0] == nr);

    const int threads = SSM_CONV_UPDATE_THREADS;
    const dim3 blocks((nr + threads - 1)/threads, n_s, 1);
    cudaStream_t stream = ctx.stream();

#define SSM_CONV_UPDATE_RES(DC, NORM) \
    ssm_conv_update_res_f32<DC, NORM><<<blocks, threads, 0, stream>>>( \
        (const char *) past->data, past->nb[1], past->nb[2], \
        (const char *) xt->data, xt->nb[0], xt->nb[1], xt->nb[2], \
        (const char *) w->data, w->nb[1], \
        (const char *) res->data, res->nb[1], \
        res2 ? (const char *) res2->data : nullptr, res2 ? res2->nb[1] : 0, \
        dyn ? (const char *) dyn->data : nullptr, dyn ? dyn->nb[1] : 0, head_size, \
        (char *) state_dst->data, kw, slot_nb, chan_nb, seq_nb, \
        (char *) out->data, out_nb_tok, \
        nr, n_t, scale, eps)
#define SSM_CONV_UPDATE_RES_NORM(DC) \
    switch (norm) { \
        case 0: SSM_CONV_UPDATE_RES(DC, 0); break; \
        case 1: SSM_CONV_UPDATE_RES(DC, 1); break; \
        case 2: SSM_CONV_UPDATE_RES(DC, 2); break; \
        default: GGML_ABORT("ssm_conv_update_res: unsupported norm %d", norm); \
    }

    switch (d_conv) {
        case 2: SSM_CONV_UPDATE_RES_NORM(2); break;
        case 3: SSM_CONV_UPDATE_RES_NORM(3); break;
        case 4: SSM_CONV_UPDATE_RES_NORM(4); break;
        case 5: SSM_CONV_UPDATE_RES_NORM(5); break;
        default: GGML_ABORT("ssm_conv_update_res: unsupported kernel size %d", d_conv);
    }
#undef SSM_CONV_UPDATE_RES_NORM
#undef SSM_CONV_UPDATE_RES
}
