#include "common.cuh"

// mul(qx, qw) -> l2_norm, set_rows(k), set_rows(v), mul(rx, rw) -> l2_norm: independent, one launch
void ggml_cuda_op_qkvr_tail(ggml_backend_cuda_context & ctx,
        const ggml_tensor * qx, const ggml_tensor * qw, float q_eps, ggml_tensor * q_out,
        ggml_tensor * k_set, ggml_tensor * v_set,
        const ggml_tensor * rx, const ggml_tensor * rw, float r_eps, ggml_tensor * r_out);
