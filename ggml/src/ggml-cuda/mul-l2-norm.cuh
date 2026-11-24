#include "common.cuh"

// fused mul(x, w) -> l2_norm, w broadcast over rows, matched in ggml_cuda_try_fuse
void ggml_cuda_op_mul_l2_norm(ggml_backend_cuda_context & ctx,
        const ggml_tensor * x, const ggml_tensor * w, float eps, ggml_tensor * out);
