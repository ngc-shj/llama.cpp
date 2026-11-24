#include "rel-bias-mask.cuh"

// the attention mask with a relative-position bias gathered from a per-query table, in F16:
//   out[kv, t, h, s] = mask[kv, t, s] + table[idx[kv, q], h, q] - sinks[h, q]      q = s*n_t + t
static __global__ void rel_bias_mask_f16(
        const char * __restrict__ table, const int64_t tb_nb1, const int64_t tb_nb2,
        const char * __restrict__ sinks, const int64_t sk_nb0, const int64_t sk_nb1,
        const char * __restrict__ idx,   const int64_t ix_nb1,
        const char * __restrict__ mask,  const int64_t mk_nb1, const int64_t mk_nb2, const int64_t mk_nb3,
        char       * __restrict__ out,   const int64_t o_nb1,  const int64_t o_nb2,  const int64_t o_nb3,
        const int64_t n_kv, const int64_t n_t, const int64_t n_head, const int64_t n_s) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_kv*n_t*n_head*n_s) {
        return;
    }
    const int64_t kv = i % n_kv;
    const int64_t t  = (i / n_kv) % n_t;
    const int64_t h  = (i / (n_kv*n_t)) % n_head;
    const int64_t s  = i / (n_kv*n_t*n_head);
    const int64_t q  = s*n_t + t;

    const int32_t bin = ((const int32_t *) (idx + q*ix_nb1))[kv];
    const float   b   = ((const float *) (table + h*tb_nb1 + q*tb_nb2))[bin] - *(const float *) (sinks + h*sk_nb0 + q*sk_nb1);
    const float   m   = ((const float *) (mask + t*mk_nb1 + s*mk_nb3))[kv];

    ((half *) (out + t*o_nb1 + h*o_nb2 + s*o_nb3))[kv] = __float2half(m + b);
}

void ggml_cuda_op_rel_bias_mask(ggml_backend_cuda_context & ctx,
        const ggml_tensor * table, const ggml_tensor * sinks, const ggml_tensor * idx,
        const ggml_tensor * mask, ggml_tensor * out) {
    GGML_ASSERT(table->type == GGML_TYPE_F32 && sinks->type == GGML_TYPE_F32 && idx->type == GGML_TYPE_I32);
    GGML_ASSERT(mask->type == GGML_TYPE_F32 && out->type == GGML_TYPE_F16);
    GGML_ASSERT(table->nb[0] == sizeof(float) && idx->nb[0] == sizeof(int32_t) && mask->nb[0] == sizeof(float) && out->nb[0] == sizeof(half));

    const int64_t n_kv = out->ne[0], n_t = out->ne[1], n_head = out->ne[2], n_s = out->ne[3];
    const int64_t n    = n_kv*n_t*n_head*n_s;
    const int threads  = 256;

    rel_bias_mask_f16<<<(n + threads - 1)/threads, threads, 0, ctx.stream()>>>(
        (const char *) table->data, table->nb[1], table->nb[2],
        (const char *) sinks->data, sinks->nb[0], sinks->nb[1],
        (const char *) idx->data,   idx->nb[1],
        (const char *) mask->data,  mask->nb[1], mask->nb[2], mask->nb[3],
        (char *) out->data, out->nb[1], out->nb[2], out->nb[3],
        n_kv, n_t, n_head, n_s);
}
