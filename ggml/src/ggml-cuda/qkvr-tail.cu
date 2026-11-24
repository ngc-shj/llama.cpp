#include "qkvr-tail.cuh"

// four independent tasks in one launch, each over its own range of blocks:
//   q = l2_norm(qx*qw), r = l2_norm(rx*rw)      one warp per row
//   k cache row idx[t] = F16(k[t]), v likewise   one thread per element
static constexpr int QKVR_THREADS = 256;
static constexpr int QKVR_WARPS   = QKVR_THREADS/WARP_SIZE;

struct qkvr_l2_task {
    const char * x; int64_t x_nb1, x_nb2;
    const char * w; int64_t w_nb1, w_ne1;
    char * out;     int64_t o_nb1, o_nb2;
    int ne0; int64_t ne1, ne2; float eps;
    int blocks;
};

struct qkvr_rows_task {
    const char * src; int64_t s_nb1;     // [n, n_rows] F32
    const void * idx;                    // [n_rows], I32 or I64
    bool idx64;
    char * dst; int64_t d_nb1;           // F16 rows
    int64_t n, n_rows;
    int blocks;
};

static __device__ void qkvr_l2(const qkvr_l2_task & t, int b) {
    const int64_t row = (int64_t) b*QKVR_WARPS + threadIdx.x/WARP_SIZE;
    if (row >= t.ne1*t.ne2) {
        return;
    }
    const int lane = threadIdx.x % WARP_SIZE;
    const int64_t i1 = row % t.ne1, i2 = row / t.ne1;
    const float * xr = (const float *) (t.x + i1*t.x_nb1 + i2*t.x_nb2);
    const float * wr = (const float *) (t.w + (i1 % t.w_ne1)*t.w_nb1);
    float       * o  = (float *) (t.out + i1*t.o_nb1 + i2*t.o_nb2);
    float sq = 0.0f;
    for (int i = lane; i < t.ne0; i += WARP_SIZE) {
        const float y = xr[i]*wr[i];
        sq += y*y;
    }
    sq = warp_reduce_sum(sq);
    const float s = rsqrtf(fmaxf(sq, t.eps*t.eps));
    for (int i = lane; i < t.ne0; i += WARP_SIZE) {
        o[i] = xr[i]*wr[i]*s;
    }
}

static __device__ void qkvr_rows(const qkvr_rows_task & t, int b) {
    const int64_t i = (int64_t) b*QKVR_THREADS + threadIdx.x;
    if (i >= t.n*t.n_rows) {
        return;
    }
    const int64_t r = i / t.n, c = i % t.n;
    const float v = ((const float *) (t.src + r*t.s_nb1))[c];
    const int64_t row = t.idx64 ? ((const int64_t *) t.idx)[r] : (int64_t) ((const int32_t *) t.idx)[r];
    ((half *) (t.dst + row*t.d_nb1))[c] = __float2half(v);
}

static __global__ void qkvr_tail_kernel(const qkvr_l2_task q, const qkvr_rows_task k, const qkvr_rows_task v, const qkvr_l2_task r) {
    int b = blockIdx.x;
    if (b < q.blocks) { qkvr_l2(q, b); return; }   b -= q.blocks;
    if (b < k.blocks) { qkvr_rows(k, b); return; } b -= k.blocks;
    if (b < v.blocks) { qkvr_rows(v, b); return; } b -= v.blocks;
    qkvr_l2(r, b);
}

static qkvr_l2_task make_l2(const ggml_tensor * x, const ggml_tensor * w, float eps, ggml_tensor * out) {
    GGML_ASSERT(x->type == GGML_TYPE_F32 && w->type == GGML_TYPE_F32 && out->type == GGML_TYPE_F32);
    GGML_ASSERT(x->nb[0] == sizeof(float) && w->nb[0] == sizeof(float) && out->nb[0] == sizeof(float) && out->ne[3] == 1);
    qkvr_l2_task t = { (const char *) x->data, x->nb[1], x->nb[2], (const char *) w->data, w->nb[1], w->ne[1],
                       (char *) out->data, out->nb[1], out->nb[2], (int) out->ne[0], out->ne[1], out->ne[2], eps, 0 };
    t.blocks = (int) ((out->ne[1]*out->ne[2] + QKVR_WARPS - 1)/QKVR_WARPS);
    return t;
}

static qkvr_rows_task make_rows(const ggml_tensor * set_rows) {
    const ggml_tensor * src = set_rows->src[0];
    const ggml_tensor * idx = set_rows->src[1];
    GGML_ASSERT(src->type == GGML_TYPE_F32 && (idx->type == GGML_TYPE_I32 || idx->type == GGML_TYPE_I64) && set_rows->type == GGML_TYPE_F16);
    GGML_ASSERT(src->nb[0] == sizeof(float) && ggml_is_contiguous(idx) && src->ne[2] == 1 && src->ne[3] == 1);
    qkvr_rows_task t = { (const char *) src->data, src->nb[1], idx->data, idx->type == GGML_TYPE_I64, (char *) set_rows->data,
                         set_rows->nb[1], src->ne[0], src->ne[1], 0 };
    t.blocks = (int) ((t.n*t.n_rows + QKVR_THREADS - 1)/QKVR_THREADS);
    return t;
}

void ggml_cuda_op_qkvr_tail(ggml_backend_cuda_context & ctx,
        const ggml_tensor * qx, const ggml_tensor * qw, float q_eps, ggml_tensor * q_out,
        ggml_tensor * k_set, ggml_tensor * v_set,
        const ggml_tensor * rx, const ggml_tensor * rw, float r_eps, ggml_tensor * r_out) {
    const qkvr_l2_task   q = make_l2(qx, qw, q_eps, q_out);
    const qkvr_rows_task k = make_rows(k_set);
    const qkvr_rows_task v = make_rows(v_set);
    const qkvr_l2_task   r = make_l2(rx, rw, r_eps, r_out);
    const int blocks = q.blocks + k.blocks + v.blocks + r.blocks;
    qkvr_tail_kernel<<<blocks, QKVR_THREADS, 0, ctx.stream()>>>(q, k, v, r);
}
