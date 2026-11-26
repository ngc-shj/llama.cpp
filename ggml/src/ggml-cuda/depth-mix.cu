#include "depth-mix.cuh"

// softmax attention of a few queries over a short stack of states, per token (one block each):
//   xn     = rms_norm(x)                                   x [n]
//   states = [stack_0 .. stack_{nb-1}, xn]                 stack columns [n] with stride stack_nb1
//   p      = softmax_b(scale*states^T q)                   q [n, nq]
//   out    = states p                                      out [n, nq]
// one block does the norm, the scores, the softmax and the weighted sum
static constexpr int DEPTH_MIX_THREADS = 512;

static __global__ void depth_mix_f32(
        const float * __restrict__ x, const float eps, float * __restrict__ xn_out, const float * __restrict__ xn_in,
        const int64_t x_nbt, const int64_t xn_out_nbt, const int64_t xn_in_nbt,
        const char * __restrict__ stack, const int64_t stack_nb1, const int64_t stack_nb2, const int nb,
        const char * __restrict__ q, const int64_t q_nb1, const int64_t q_nb2, const int nq,
        const float scale,
        char * __restrict__ out, const int64_t out_nb1, const int64_t out_nb2,
        const int n,
        const float * __restrict__ nw, const float neps, float * __restrict__ nout) {
    extern __shared__ float smem[];
    {
        const int64_t t = blockIdx.x;
        x      = x      ? (const float *) ((const char *) x      + t*x_nbt)      : nullptr;
        xn_out = xn_out ? (float *)       ((char *)       xn_out + t*xn_out_nbt) : nullptr;
        xn_in  = xn_in  ? (const float *) ((const char *) xn_in  + t*xn_in_nbt)  : nullptr;
        stack += t*stack_nb2;
        q     += t*q_nb2;
        out   += t*out_nb2;
    }
    float * xn  = smem;          // [n]
    float * red = xn + n;        // [32]
    float * p   = red + 32;      // [(nb + 1)*nq]
    float * o   = p + (nb + 1)*nq; // [n] with nout: the result before its norm

    const int tid    = threadIdx.x;
    const int lane   = tid % WARP_SIZE;
    const int warp   = tid / WARP_SIZE;
    const int nwarps = DEPTH_MIX_THREADS / WARP_SIZE;

    if (xn_in) {
        // cur is normalized already
        for (int i = tid; i < n; i += DEPTH_MIX_THREADS) {
            xn[i] = xn_in[i];
        }
    } else {
    float ss = 0.0f;
    for (int i = tid; i < n; i += DEPTH_MIX_THREADS) {
        ss += x[i]*x[i];
    }
    ss = warp_reduce_sum(ss);
    if (lane == 0) {
        red[warp] = ss;
    }
    __syncthreads();
    if (warp == 0) {
        ss = lane < nwarps ? red[lane] : 0.0f;
        ss = warp_reduce_sum(ss);
        if (lane == 0) {
            red[0] = ss;
        }
    }
    __syncthreads();
    const float rs = rsqrtf(red[0]/n + eps);
    for (int i = tid; i < n; i += DEPTH_MIX_THREADS) {
        xn[i]     = x[i]*rs;
        xn_out[i] = xn[i];
    }
    }
    __syncthreads();

    const int nd = (nb + 1)*nq;
    for (int d = warp; d < nd; d += nwarps) {
        const int b  = d / nq;
        const int iq = d % nq;
        const float * col = b < nb ? (const float *) (stack + b*stack_nb1) : xn;
        const float * qq  = (const float *) (q + iq*q_nb1);
        float dot = 0.0f;
        for (int i = lane; i < n; i += WARP_SIZE) {
            dot += col[i]*qq[i];
        }
        dot = warp_reduce_sum(dot);
        if (lane == 0) {
            p[b*nq + iq] = scale*dot;
        }
    }
    __syncthreads();

    if (tid < nq) {
        float mx = -INFINITY;
        for (int b = 0; b <= nb; ++b) {
            mx = fmaxf(mx, p[b*nq + tid]);
        }
        float sum = 0.0f;
        for (int b = 0; b <= nb; ++b) {
            const float e = expf(p[b*nq + tid] - mx);
            p[b*nq + tid] = e;
            sum += e;
        }
        const float inv = 1.0f/sum;
        for (int b = 0; b <= nb; ++b) {
            p[b*nq + tid] *= inv;
        }
    }
    __syncthreads();

    for (int i = tid; i < n; i += DEPTH_MIX_THREADS) {
        float acc[GGML_CUDA_DEPTH_MIX_MAX_Q] = { 0.0f };
        for (int b = 0; b <= nb; ++b) {
            const float v = b < nb ? ((const float *) (stack + b*stack_nb1))[i] : xn[i];
#pragma unroll
            for (int iq = 0; iq < GGML_CUDA_DEPTH_MIX_MAX_Q; ++iq) {
                if (iq < nq) {
                    acc[iq] += v*p[b*nq + iq];
                }
            }
        }
        if (nout) {
            o[i] = acc[0];
            continue;
        }
#pragma unroll
        for (int iq = 0; iq < GGML_CUDA_DEPTH_MIX_MAX_Q; ++iq) {
            if (iq < nq) {
                ((float *) (out + iq*out_nb1))[i] = acc[iq];
            }
        }
    }
    if (nout) {
        // rms_norm(result)*nw of the single query: the block holds the whole result
        __syncthreads();
        float ss = 0.0f;
        for (int i = tid; i < n; i += DEPTH_MIX_THREADS) {
            ss += o[i]*o[i];
        }
        ss = warp_reduce_sum(ss);
        if (lane == 0) {
            red[warp] = ss;
        }
        __syncthreads();
        if (warp == 0) {
            ss = lane < nwarps ? red[lane] : 0.0f;
            ss = warp_reduce_sum(ss);
            if (lane == 0) {
                red[0] = ss;
            }
        }
        __syncthreads();
        const float rs = rsqrtf(red[0]/n + neps);
        for (int i = tid; i < n; i += DEPTH_MIX_THREADS) {
            nout[i] = o[i]*rs*nw[i];
        }
    }
}

void ggml_cuda_op_depth_mix(ggml_backend_cuda_context & ctx,
        const ggml_tensor * norm, const ggml_tensor * stack, const ggml_tensor * q,
        const ggml_tensor * softmax, ggml_tensor * out, const ggml_tensor * nw, float neps, ggml_tensor * nout) {
    const ggml_tensor * x = norm->src[0];

    const int n  = (int) x->ne[0];
    const int nb = (int) stack->ne[1];
    const int nq = (int) q->ne[1];

    GGML_ASSERT(nq <= GGML_CUDA_DEPTH_MIX_MAX_Q && ggml_is_contiguous_rows(x) && ggml_is_contiguous(norm));
    GGML_ASSERT(stack->nb[0] == sizeof(float) && q->nb[0] == sizeof(float) && out->nb[0] == sizeof(float));

    const float eps   = ggml_get_op_params_f32(norm, 0);
    const float scale = ggml_get_op_params_f32(softmax, 0);

    const int n_t = (int) out->ne[2];
    GGML_ASSERT(!nout || n_t == 1);

    const size_t smem = (2*n + 32 + (nb + 1)*nq)*sizeof(float);
    depth_mix_f32<<<n_t, DEPTH_MIX_THREADS, smem, ctx.stream()>>>(
        (const float *) x->data, eps, (float *) norm->data, nullptr,
        x->nb[2], norm->nb[2], 0,
        (const char *) stack->data, stack->nb[1], stack->nb[2], nb,
        (const char *) q->data, q->nb[1], 0, nq, // q is shared by the tokens
        scale,
        (char *) out->data, out->nb[1], out->nb[2],
        n, nw ? (const float *) nw->data : nullptr, neps, nout ? (float *) nout->data : nullptr);
}

void ggml_cuda_op_depth_mix_normed(ggml_backend_cuda_context & ctx,
        const ggml_tensor * xn, const ggml_tensor * stack, const ggml_tensor * q,
        const ggml_tensor * softmax, ggml_tensor * out, const ggml_tensor * nw, float neps, ggml_tensor * nout) {
    const int n  = (int) xn->ne[0];
    const int nb = (int) stack->ne[1];
    const int nq = (int) q->ne[1];

    GGML_ASSERT(nq <= GGML_CUDA_DEPTH_MIX_MAX_Q && ggml_is_contiguous(xn));
    GGML_ASSERT(stack->nb[0] == sizeof(float) && q->nb[0] == sizeof(float) && out->nb[0] == sizeof(float));

    const float scale = ggml_get_op_params_f32(softmax, 0);

    const int n_t = (int) out->ne[2];
    GGML_ASSERT(!nout || n_t == 1);

    const size_t smem = (2*n + 32 + (nb + 1)*nq)*sizeof(float);
    depth_mix_f32<<<n_t, DEPTH_MIX_THREADS, smem, ctx.stream()>>>(
        nullptr, 0.0f, nullptr, (const float *) xn->data,
        0, 0, xn->nb[2],
        (const char *) stack->data, stack->nb[1], stack->nb[2], nb,
        (const char *) q->data, q->nb[1], 0, nq, // q is shared by the tokens
        scale,
        (char *) out->data, out->nb[1], out->nb[2],
        n, nw ? (const float *) nw->data : nullptr, neps, nout ? (float *) nout->data : nullptr);
}
