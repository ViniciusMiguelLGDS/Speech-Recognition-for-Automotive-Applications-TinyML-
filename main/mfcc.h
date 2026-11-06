#pragma once
#include <stdint.h>
#include "config.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Cria janela Hann, banco Mel, matriz DCT e twiddles. Chamar uma vez. Retorna 0 em sucesso. */
int mfcc_init(void);
/* Reproduz torchaudio.transforms.MFCC + pad_or_truncate.
 * pcm: n amostras int16 (n >= N_FFT/2+1). scale: fator p/ levar a [-1,1] (ex.: AUDIO_GAIN/32768).
 * feat: N_FEATURES floats, ordem [coef][quadro] (mesma do CSV de treino). */
/* Retorna 0 em sucesso, -1 se n exceder WINDOW_SAMPLES (buffers estaticos) ou for muito curto. */
int mfcc_compute(const int16_t *pcm, int n, float scale, float *feat);
#ifdef __cplusplus
}
#endif
