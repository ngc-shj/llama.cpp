#include "mul-l2-norm.cuh"

// out = l2_norm(x*w, eps) per row of ne0, w broadcast over the rows it does not cover: one warp per row
static __global__ void mul_l2_norm_f32(
        const char * __restrict__ x, const int64_t x_nb1, const int64_t x_nb2, const int64_t x_nb3,
        const char * __restrict__ w, const int64_t w_nb1, const int64_t w_nb2, const int64_t w_nb3,
        const int64_t w_ne1, const int64_t w_ne2, const int64_t w_ne3,
        char * __restrict__ out, const int64_t o_nb1, const int64_t o_nb2, const int64_t o_nb3,
        const int ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, const float eps) {
    const int64_t row = (int64_t) blockIdx.x*(blockDim.x/WARP_SIZE) + threadIdx.x/WARP_SIZE;
    if (row >= ne1*ne2*ne3) {
        return;
    }
    const int lane = threadIdx.x % WARP_SIZE;
    const int64_t i1 = row % ne1, i2 = (row/ne1) % ne2, i3 = row/(ne1*ne2);

    const float * xr = (const float *) (x + i1*x_nb1 + i2*x_nb2 + i3*x_nb3);
    const float * wr = (const float *) (w + (i1 % w_ne1)*w_nb1 + (i2 % w_ne2)*w_nb2 + (i3 % w_ne3)*w_nb3);
    float       * o  = (float *) (out + i1*o_nb1 + i2*o_nb2 + i3*o_nb3);

    float sq = 0.0f;
    for (int i = lane; i < ne0; i += WARP_SIZE) {
        const float y = xr[i]*wr[i];
        sq += y*y;
    }
    sq = warp_reduce_sum(sq);
    const float s = rsqrtf(fmaxf(sq, eps*eps));
    for (int i = lane; i < ne0; i += WARP_SIZE) {
        o[i] = xr[i]*wr[i]*s;
    }
}

void ggml_cuda_op_mul_l2_norm(ggml_backend_cuda_context & ctx,
        const ggml_tensor * x, const ggml_tensor * w, float eps, ggml_tensor * out) {
    GGML_ASSERT(x->type == GGML_TYPE_F32 && w->type == GGML_TYPE_F32 && out->type == GGML_TYPE_F32);
    GGML_ASSERT(x->nb[0] == sizeof(float) && w->nb[0] == sizeof(float) && out->nb[0] == sizeof(float));
    GGML_ASSERT(w->ne[0] == x->ne[0] && ggml_are_same_shape(x, out));

    const int64_t nrows = ggml_nrows(out);
    const int warps = 8;
    mul_l2_norm_f32<<<(nrows + warps - 1)/warps, warps*WARP_SIZE, 0, ctx.stream()>>>(
        (const char *) x->data, x->nb[1], x->nb[2], x->nb[3],
        (const char *) w->data, w->nb[1], w->nb[2], w->nb[3], w->ne[1], w->ne[2], w->ne[3],
        (char *) out->data, out->nb[1], out->nb[2], out->nb[3],
        (int) out->ne[0], out->ne[1], out->ne[2], out->ne[3], eps);
}
