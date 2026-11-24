#include "situ-fused.cuh"

// the expert scale and the situ activation of a gated MoE, elementwise:
//   g, u = s*gate, s*up                                   s broadcast over the rows
//   out  = b*tanh(g/b) * sigmoid(g) * lb*tanh(u/lb)
static __global__ void situ_fused_f32(
        const char * __restrict__ gate, const int64_t g_nb0, const int64_t g_nb1, const int64_t g_nb2,
        const char * __restrict__ up,   const int64_t u_nb0, const int64_t u_nb1, const int64_t u_nb2,
        const char * __restrict__ s,    const int64_t s_nb1, const int64_t s_nb2,
        const float * __restrict__ s_tab, const int32_t * __restrict__ ids, const int64_t ids_nb1,
        float * __restrict__ out, const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const float inv_b, const float b, const float inv_lb, const float lb) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= ne0*ne1*ne2) {
        return;
    }
    const int64_t i0 = i % ne0;
    const int64_t i1 = (i / ne0) % ne1;
    const int64_t i2 = i / (ne0*ne1);

    const float sc = s_tab ? s_tab[ids[i1 + i2*ids_nb1]] : s ? *(const float *) (s + i1*s_nb1 + i2*s_nb2) : 1.0f;
    const float g  = *(const float *) (gate + i0*g_nb0 + i1*g_nb1 + i2*g_nb2) * sc;
    const float u  = *(const float *) (up   + i0*u_nb0 + i1*u_nb1 + i2*u_nb2) * sc;

    const float act = (tanhf(g*inv_b)*b) * (1.0f/(1.0f + expf(-g)));
    out[i] = act * (tanhf(u*inv_lb)*lb);
}

void ggml_cuda_op_situ_fused(ggml_backend_cuda_context & ctx,
        const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * s,
        float inv_b, float b, float inv_lb, float lb, ggml_tensor * out, const ggml_tensor * s_tab, const ggml_tensor * ids) {
    GGML_ASSERT(ggml_is_contiguous(out) && ggml_are_same_shape(gate, out) && ggml_are_same_shape(up, out));
    GGML_ASSERT(gate->nb[0] == sizeof(float) && up->nb[0] == sizeof(float));

    const int64_t ne0 = out->ne[0], ne1 = out->ne[1], ne2 = out->ne[2];
    const int64_t n   = ne0*ne1*ne2;
    const int threads = 256;

    situ_fused_f32<<<(n + threads - 1)/threads, threads, 0, ctx.stream()>>>(
        (const char *) gate->data, gate->nb[0], gate->nb[1], gate->nb[2],
        (const char *) up->data,   up->nb[0],   up->nb[1],   up->nb[2],
        s ? (const char *) s->data : nullptr, s ? s->nb[1] : 0, s ? s->nb[2] : 0,
        s_tab ? (const float *) s_tab->data : nullptr, ids ? (const int32_t *) ids->data : nullptr, ids ? ids->nb[1]/sizeof(int32_t) : 0,
        (float *) out->data, ne0, ne1, ne2,
        inv_b, b, inv_lb, lb);
}
