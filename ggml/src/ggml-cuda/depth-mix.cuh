#include "common.cuh"

#define GGML_CUDA_DEPTH_MIX_MAX_Q     8
#define GGML_CUDA_DEPTH_MIX_MAX_STACK 32

// fused rms_norm(x) -> concat(stack, xn) -> mul_mat(states, q) -> soft_max -> out_prod(states, p^T), one token,
// matched in ggml_cuda_try_fuse
void ggml_cuda_op_depth_mix(ggml_backend_cuda_context & ctx,
        const ggml_tensor * norm, const ggml_tensor * stack, const ggml_tensor * q,
        const ggml_tensor * softmax, ggml_tensor * out,
        const ggml_tensor * nw = nullptr, float neps = 0.0f, ggml_tensor * nout = nullptr);

// the same with cur normalized already: concat(stack, xn) -> mul_mat -> soft_max -> out_prod
// [-> rms_norm -> mul(nw) of a single query into nout]
void ggml_cuda_op_depth_mix_normed(ggml_backend_cuda_context & ctx,
        const ggml_tensor * xn, const ggml_tensor * stack, const ggml_tensor * q,
        const ggml_tensor * softmax, ggml_tensor * out,
        const ggml_tensor * nw = nullptr, float neps = 0.0f, ggml_tensor * nout = nullptr);
