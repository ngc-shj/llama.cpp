#include "common.cuh"

// fused concat(past, cur^T) -> state copy -> ssm_conv -> add(res) -> scale [-> l2_norm (norm 1) or rms_norm (norm 2) per
// head of 128 channels], matched in ggml_cuda_try_fuse
void ggml_cuda_op_ssm_conv_update_res(ggml_backend_cuda_context & ctx,
        const ggml_tensor * concat, const ggml_tensor * state_dst, const ggml_tensor * conv,
        const ggml_tensor * res, ggml_tensor * out, float scale, int norm = 0, float eps = 0.0f,
        const ggml_tensor * res2 = nullptr, const ggml_tensor * dyn = nullptr, int head_size = 0,
        int kw = 1, int64_t slot_nb = 0, int64_t chan_nb = 0, int64_t seq_nb = 0);
