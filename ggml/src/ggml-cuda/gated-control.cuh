#include "common.cuh"

// fused concat(cur, msg) -> rms_norm -> mul_mat(down) -> mul_mat(forget), tanh, mul, sub -> mul_mat(gain), tanh, mul -> add,
// per token, matched in ggml_cuda_try_fuse
void ggml_cuda_op_gated_control(ggml_backend_cuda_context & ctx,
        const ggml_tensor * cur, const ggml_tensor * msg, float eps,
        const ggml_tensor * down, ggml_tensor * h,
        const ggml_tensor * forget, const ggml_tensor * gain, ggml_tensor * out);
