/*
 * Extracao de MFCC equivalente a torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=20,
 * melkwargs={n_fft:400, hop_length:150, n_mels:64}), com os padroes do torchaudio:
 *   - janela Hann periodica de 400, center=True com pad 'reflect' (200 de cada lado)
 *   - espectro de POTENCIA (|X|^2), 201 bins
 *   - banco Mel HTK, f_min=0, f_max=8000, norm=None (triangulos sem normalizacao de area)
 *   - AmplitudeToDB: 10*log10(max(x,1e-10)), top_db=80 relativo ao MAXIMO DO ENUNCIADO INTEIRO
 *   - DCT-II ortonormal (norm='ortho'), 20 coeficientes
 *   - pad_or_truncate para 150 quadros: preenche com ZEROS (no dominio MFCC) ou corta o excedente
 * O codigo e portavel (sem dependencias do ESP-IDF) para ser testado no PC.
 */
#include "mfcc.h"
#include <math.h>
#include <string.h>

static float s_win[N_FFT];
static float s_fb[N_MELS][N_FREQS];
static int   s_fb_lo[N_MELS], s_fb_hi[N_MELS];   /* faixa de bins nao nulos [lo,hi) */
static float s_dct[N_MFCC][N_MELS];
static float s_tw_re[N_FFT], s_tw_im[N_FFT];     /* exp(-2*pi*i*k/N_FFT) */
static float s_db[MAX_FRAMES][N_MELS];           /* log-mel (dB) por quadro */
static float s_re_in[N_FFT], s_im_in[N_FFT], s_re[N_FFT], s_im[N_FFT];
static const int s_factors[] = { 5, 5, 2, 2, 2, 2 };  /* 5*5*2*2*2*2 = 400 */
#define NFACT ((int)(sizeof(s_factors)/sizeof(s_factors[0])))

static double hz2mel(double f){ return 2595.0 * log10(1.0 + f / 700.0); }
static double mel2hz(double m){ return 700.0 * (pow(10.0, m / 2595.0) - 1.0); }

int mfcc_init(void)
{
    const double PI = 3.14159265358979323846;
    for (int i = 0; i < N_FFT; i++) {
        s_win[i] = (float)(0.5 - 0.5 * cos(2.0 * PI * i / N_FFT));
        s_tw_re[i] = (float)cos(2.0 * PI * i / N_FFT);
        s_tw_im[i] = (float)(-sin(2.0 * PI * i / N_FFT));
    }
    /* banco Mel (identico a torchaudio.functional.melscale_fbanks, mel_scale='htk', norm=None) */
    double f_pts[N_MELS + 2];
    double m_min = hz2mel(0.0), m_max = hz2mel(SAMPLE_RATE / 2.0);
    for (int i = 0; i < N_MELS + 2; i++) f_pts[i] = mel2hz(m_min + (m_max - m_min) * i / (N_MELS + 1));
    for (int m = 0; m < N_MELS; m++) {
        double fd_down = f_pts[m + 1] - f_pts[m], fd_up = f_pts[m + 2] - f_pts[m + 1];
        int lo = N_FREQS, hi = 0;
        for (int k = 0; k < N_FREQS; k++) {
            double fk = (double)(SAMPLE_RATE / 2) * k / (N_FREQS - 1);
            double down = -(f_pts[m] - fk) / fd_down;       /* (fk - f_m)/fd */
            double up   = (f_pts[m + 2] - fk) / fd_up;
            double v = down < up ? down : up; if (v < 0) v = 0;
            s_fb[m][k] = (float)v;
            if (v > 0) { if (k < lo) lo = k; if (k + 1 > hi) hi = k + 1; }
        }
        s_fb_lo[m] = lo; s_fb_hi[m] = hi;
    }
    /* DCT-II ortonormal: dct[k][n] = sqrt(2/N) * cos(pi/N*(n+0.5)*k), linha 0 * 1/sqrt(2) */
    for (int k = 0; k < N_MFCC; k++)
        for (int n = 0; n < N_MELS; n++) {
            double v = cos(PI / N_MELS * (n + 0.5) * k) * sqrt(2.0 / N_MELS);
            if (k == 0) v *= 1.0 / sqrt(2.0);
            s_dct[k][n] = (float)v;
        }
    return 0;
}

/* FFT complexa de raio misto (DIT recursiva), n = produto dos fatores restantes. */
static void fft_rec(float *ore, float *oim, const float *ire, const float *iim,
                    int n, int stride, int fi)
{
    if (n == 1) { ore[0] = ire[0]; oim[0] = iim[0]; return; }
    const int p = s_factors[fi], m = n / p;
    for (int q = 0; q < p; q++)
        fft_rec(ore + q * m, oim + q * m, ire + q * stride, iim + q * stride, m, stride * p, fi + 1);
    const int tstep = N_FFT / n;
    float tr[5], ti[5];
    for (int k = 0; k < m; k++) {
        for (int r = 0; r < p; r++) {
            const int idx = k + r * m;
            float ar = 0.f, ai = 0.f;
            for (int q = 0; q < p; q++) {
                const int j = ((q * idx) % n) * tstep;          /* W_n^(q*idx) */
                const float xr = ore[q * m + k], xi = oim[q * m + k];
                ar += xr * s_tw_re[j] - xi * s_tw_im[j];
                ai += xr * s_tw_im[j] + xi * s_tw_re[j];
            }
            tr[r] = ar; ti[r] = ai;
        }
        for (int r = 0; r < p; r++) { ore[k + r * m] = tr[r]; oim[k + r * m] = ti[r]; }
    }
}

int mfcc_compute(const int16_t *pcm, int n, float scale, float *feat)
{
    if (n <= N_FFT || n > WINDOW_SAMPLES) return -1;
    const int T = 1 + n / HOP_LENGTH;           /* quadros com center=True */
    float db_max = -1e30f;
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < N_FFT; i++) {
            int idx = t * HOP_LENGTH + i - N_FFT / 2;
            if (idx < 0) idx = -idx;                          /* reflect (sem repetir a borda) */
            if (idx >= n) idx = 2 * (n - 1) - idx;
            s_re_in[i] = (float)pcm[idx] * scale * s_win[i];
            s_im_in[i] = 0.f;
        }
        fft_rec(s_re, s_im, s_re_in, s_im_in, N_FFT, 1, 0);
        float pw[N_FREQS];
        for (int k = 0; k < N_FREQS; k++) pw[k] = s_re[k] * s_re[k] + s_im[k] * s_im[k];
        for (int m = 0; m < N_MELS; m++) {
            float acc = 0.f;
            for (int k = s_fb_lo[m]; k < s_fb_hi[m]; k++) acc += pw[k] * s_fb[m][k];
            if (acc < 1e-10f) acc = 1e-10f;
            const float d = 10.0f * log10f(acc);
            s_db[t][m] = d;
            if (d > db_max) db_max = d;
        }
    }
    const float floor_db = db_max - 80.0f;      /* top_db = 80 */
    memset(feat, 0, sizeof(float) * N_FEATURES);   /* zeros = padding do treinamento */
    const int Tuse = T < MAX_TIME ? T : MAX_TIME;
    for (int t = 0; t < Tuse; t++) {
        float row[N_MELS];
        for (int m = 0; m < N_MELS; m++) row[m] = s_db[t][m] < floor_db ? floor_db : s_db[t][m];
        for (int k = 0; k < N_MFCC; k++) {
            float acc = 0.f;
            for (int m = 0; m < N_MELS; m++) acc += s_dct[k][m] * row[m];
            feat[k * MAX_TIME + t] = acc;
        }
    }
    return 0;
}
