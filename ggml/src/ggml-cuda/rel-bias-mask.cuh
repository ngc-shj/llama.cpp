#include "common.cuh"

// fused sub(table, sinks) -> cont(permute) -> get_rows(idx) -> cont(permute) -> add(mask) -> cpy(F16),
// the per-head attention mask with a relative-position bias, matched in ggml_cuda_try_fuse
void ggml_cuda_op_rel_bias_mask(ggml_backend_cuda_context & ctx,
        const ggml_tensor * table, const ggml_tensor * sinks, const ggml_tensor * idx,
        const ggml_tensor * mask, ggml_tensor * out);

// one token: mul_mat(rb, r) -> mul_mat(ws, x) -> the rel-bias mask above, the bias computed per key without the table
void ggml_cuda_op_rel_bias_mask_direct(ggml_backend_cuda_context & ctx,
        const ggml_tensor * rb, const ggml_tensor * r, const ggml_tensor * ws, const ggml_tensor * x,
        const ggml_tensor * idx, const ggml_tensor * mask, ggml_tensor * out);
