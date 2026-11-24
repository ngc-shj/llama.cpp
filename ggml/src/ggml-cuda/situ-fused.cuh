#include "common.cuh"

// fused mul(gate_up, expert scale) -> situ(gate, up) of a gated MoE, matched in ggml_cuda_try_fuse
void ggml_cuda_op_situ_fused(ggml_backend_cuda_context & ctx,
        const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * s,
        float inv_b, float b, float inv_lb, float lb, ggml_tensor * out,
        const ggml_tensor * s_tab = nullptr, const ggml_tensor * ids = nullptr);
