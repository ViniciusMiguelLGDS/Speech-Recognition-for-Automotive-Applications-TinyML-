#pragma once
#include <stdint.h>
#include "esp_err.h"
#include "config.h"
esp_err_t audio_init(void);               /* configura I2S e inicia a tarefa de captura */
/* Bloqueia ate chegarem STRIDE_SAMPLES novas amostras. Retorna 0 em sucesso, -1 em timeout. */
int audio_wait_stride(uint32_t timeout_ms);
/* Copia as ultimas WINDOW_SAMPLES amostras (mais antiga primeiro). Devolve (por t_newest_us)
 * o instante (esp_timer) em que o bloco com a amostra mais recente foi lido do I2S. */
void audio_snapshot(int16_t *dst, int64_t *t_newest_us);
uint32_t audio_overruns(void);            /* quantas vezes a captura perdeu a espera do consumidor */
int audio_filled(void);                   /* 1 se ja ha WINDOW_SAMPLES amostras acumuladas */
