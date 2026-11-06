/*
 * Inferencia da DNN treinada (3000-512-512-256-128-11) a partir de model.bin.
 * Cada camada: y[o] = escala[o] * sum_i W[o][i]*x[i] + vies[o]; ReLU quando indicado.
 * escala[o] = s_w * gamma/sqrt(var+eps) (BatchNorm ja fundido) e vies[o] ja inclui o BN.
 * Pesos: int8 por tensor (zero_point 0) lidos do modelo_int8.pt. Sem PyTorch/TFLM.
 */
#include "model_inference.h"
#include <math.h>
#include <string.h>

typedef struct { uint32_t in, out, relu; const int8_t *W; const float *scale, *bias; } layer_t;
static layer_t s_layers[MODEL_MAX_LAYERS];
static int s_nlayers, s_ncls;
static float s_buf[2][512];

static size_t up16(size_t x) { return (x + 15u) & ~(size_t)15u; }
static uint32_t rd32(const uint8_t *p) { return p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t)p[3] << 24); }

int model_load(const uint8_t *b, size_t size)
{
    if (size < 32 || rd32(b) != 0x4D4C4D54u || rd32(b + 4) != 1u) return -1;
    s_nlayers = (int)rd32(b + 8);
    const uint32_t n_in = rd32(b + 12);
    s_ncls = (int)rd32(b + 16);
    if (s_nlayers < 1 || s_nlayers > MODEL_MAX_LAYERS || n_in != N_FEATURES) return -2;
    size_t off = up16(32 + 16u * s_nlayers);
    for (int l = 0; l < s_nlayers; l++) {
        const uint8_t *d = b + 32 + 16 * l;
        layer_t *L = &s_layers[l];
        L->in = rd32(d); L->out = rd32(d + 4); L->relu = rd32(d + 8);
        if (L->out > 512 || L->in > N_FEATURES) return -3;
        size_t w = up16((size_t)L->in * L->out), f = up16(4u * L->out);
        if (off + w + 2 * f > size) return -4;
        L->W = (const int8_t *)(b + off);   off += w;
        L->scale = (const float *)(b + off); off += f;
        L->bias = (const float *)(b + off);  off += f;
        if ((((uintptr_t)L->scale) & 3u) || (((uintptr_t)L->bias) & 3u)) return -5;
    }
    if ((int)s_layers[s_nlayers - 1].out != s_ncls) return -6;
    return 0;
}
int model_num_classes(void) { return s_ncls; }

#if !MODEL_INT8_ACTIVATIONS
static void layer_run(const layer_t *L, const float *x, float *y)
{
    for (uint32_t o = 0; o < L->out; o++) {
        const int8_t *w = L->W + (size_t)o * L->in;
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        uint32_t i = 0;
        for (; i + 4 <= L->in; i += 4) {
            a0 += (float)w[i] * x[i];     a1 += (float)w[i + 1] * x[i + 1];
            a2 += (float)w[i + 2] * x[i + 2]; a3 += (float)w[i + 3] * x[i + 3];
        }
        for (; i < L->in; i++) a0 += (float)w[i] * x[i];
        float v = ((a0 + a1) + (a2 + a3)) * L->scale[o] + L->bias[o];
        y[o] = (L->relu && v < 0.f) ? 0.f : v;
    }
}
#else
/* Ativacoes quantizadas dinamicamente (int8 simetrico, escala = max|x|/127). Mais rapido,
 * porem menos preciso que o modo float (ver README). Nao e identico a quantizacao dinamica do PyTorch. */
static void layer_run(const layer_t *L, const float *x, float *y)
{
    static int8_t qx[N_FEATURES];
    float mx = 0.f;
    for (uint32_t i = 0; i < L->in; i++) { float a = fabsf(x[i]); if (a > mx) mx = a; }
    const float sx = mx > 0.f ? mx / 127.f : 1.f, inv = 1.f / sx;
    for (uint32_t i = 0; i < L->in; i++) {
        float q = roundf(x[i] * inv); if (q > 127.f) q = 127.f; if (q < -127.f) q = -127.f;
        qx[i] = (int8_t)q;
    }
    for (uint32_t o = 0; o < L->out; o++) {
        const int8_t *w = L->W + (size_t)o * L->in;
        int32_t acc = 0;
        for (uint32_t i = 0; i < L->in; i++) acc += (int32_t)w[i] * qx[i];
        float v = (float)acc * (L->scale[o] * sx) + L->bias[o];
        y[o] = (L->relu && v < 0.f) ? 0.f : v;
    }
}
#endif

void model_infer(const float *feat, float *logits)
{
    const float *in = feat;
    for (int l = 0; l < s_nlayers; l++) {
        float *out = (l == s_nlayers - 1) ? logits : s_buf[l & 1];
        layer_run(&s_layers[l], in, out);
        in = out;
    }
}

int model_top1(const float *z, int n, float *prob)
{
    int best = 0; float mx = z[0];
    for (int i = 1; i < n; i++) if (z[i] > mx) { mx = z[i]; best = i; }
    float s = 0.f;
    for (int i = 0; i < n; i++) s += expf(z[i] - mx);
    *prob = 1.0f / s;       /* exp(0)/sum */
    return best;
}
