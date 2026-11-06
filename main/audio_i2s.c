/*
 * Captura INMP441 com ESP32-S3 (ESP-IDF 5.5.2).
 *
 * - I2S Philips, 16 kHz, palavras de 32 bits.
 * - Recepcao MONO, somente slot esquerdo (L/R do INMP441 em GND).
 *   O BCLK continua 64 x fs = 1,024 MHz (2 slots de 32 bits por quadro WS);
 *   o driver apenas descarta o slot direito.
 * - O INMP441 entrega 24 bits alinhados a esquerda na palavra de 32 bits;
 *   r >> 16 fornece os 16 bits mais significativos (PCM16).
 *
 * Diagnostico de saltos (AUDIO_DIAG_GLITCH):
 *   detecta amostras consecutivas cuja diferenca (em 24 bits) e maior que
 *   GLITCH_JUMP, o que nao ocorre com audio real em nivel normal, e imprime
 *   as palavras brutas vizinhas em hexadecimal e o XOR entre elas.
 */

#include "audio_i2s.h"

#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <limits.h>

#include "driver/i2s_std.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

#define BLOCK_FRAMES 256
#define DIAG_BLOCKS  64

/*
 * Diagnostico opcional (manter 0 no uso normal).
 * 1 = liga o pull-up interno do pino SD.
 *   - raw passa a 0xFFFFFFFF constante -> SD NAO esta sendo acionado
 *     (microfone sem clock valido / sem alimentacao / SD sem contato).
 *   - raw continua 0x00000000          -> algo forca SD em nivel baixo.
 *   - raw varia                         -> microfone funcionando.
 * Com o microfone ativo, a saida do INMP441 domina o pull-up (~45 kOhm).
 */
#ifndef AUDIO_DIAG_SD_PULLUP
#define AUDIO_DIAG_SD_PULLUP 0
#endif

/* Pull-down em SD, conforme datasheet do INMP441 (100 kOhm SD->GND).
 * Usa o pull-down interno do ESP32-S3 (~45 kOhm). */
#ifndef AUDIO_SD_PULLDOWN
#define AUDIO_SD_PULLDOWN 1
#endif

/* Diagnostico de saltos entre amostras consecutivas. */
#ifndef AUDIO_DIAG_GLITCH
#define AUDIO_DIAG_GLITCH 1
#endif
#define GLITCH_JUMP      (1 << 22)   /* 1/4 da faixa de 24 bits */
#define GLITCH_LOG_MAX   3           /* no maximo 3 saltos impressos por ~1 s */

static const char *TAG = "audio";

static i2s_chan_handle_t s_rx;

static int16_t s_ring[WINDOW_SAMPLES];

static uint64_t s_total;
static int64_t  s_t_newest_us;

static SemaphoreHandle_t s_lock;
static SemaphoreHandle_t s_stride_sem;

static uint32_t s_overruns;
static uint64_t s_next_notify;

static void audio_task(void *arg)
{
    (void)arg;

    /* Mono: uma palavra de 32 bits por quadro (somente slot esquerdo). */
    static int32_t raw[BLOCK_FRAMES];
    static int16_t blk[BLOCK_FRAMES];

    float dc_x = 0.0f;
    float dc_y = 0.0f;
    const float R = 0.995f;

    int32_t raw_min = INT32_MAX;
    int32_t raw_max = INT32_MIN;
    int16_t pcm_min = INT16_MAX;
    int16_t pcm_max = INT16_MIN;

    uint32_t clip_count  = 0;
    uint32_t diag_blocks = 0;
    int printed = 0;

    /* Estado do detector de saltos (persiste entre blocos). */
    bool     have_prev     = false;
    int32_t  prev_raw      = 0;
    uint32_t glitch_count  = 0;
    uint32_t glitch_logged = 0;

    while (1) {
        size_t got = 0;

        esp_err_t e = i2s_channel_read(s_rx, raw, sizeof(raw), &got, 1000);

        if (e != ESP_OK || got == 0) {
            ESP_LOGW(TAG, "i2s_channel_read: %s", esp_err_to_name(e));
            continue;
        }

        /* Quadros mono recebidos: uma palavra de 32 bits cada. */
        const int frames = got / sizeof(int32_t);
        if (frames <= 0) {
            continue;
        }

        /* Primeiros 5 blocos: palavras brutas (amostras consecutivas do slot esquerdo). */
        if (printed < 5) {
            ESP_LOGI(TAG,
                     "I2S got=%u | raw[0]=0x%08lx raw[1]=0x%08lx raw[2]=0x%08lx raw[3]=0x%08lx",
                     (unsigned)got,
                     (unsigned long)raw[0], (unsigned long)raw[1],
                     (unsigned long)raw[2], (unsigned long)raw[3]);
            printed++;
        }

        for (int i = 0; i < frames; i++) {
            const int32_t r = raw[i];

            if (r < raw_min) raw_min = r;
            if (r > raw_max) raw_max = r;

#if AUDIO_DIAG_GLITCH
            /* Compara as amostras de 24 bits consecutivas. */
            if (have_prev) {
                const int32_t d = (r >> 8) - (prev_raw >> 8);
                if (d > GLITCH_JUMP || d < -GLITCH_JUMP) {
                    glitch_count++;
                    if (glitch_logged < GLITCH_LOG_MAX) {
                        const uint32_t nx = (i + 1 < frames) ? (uint32_t)raw[i + 1] : 0u;
                        ESP_LOGW(TAG,
                                 "salto: ant=0x%08lx atual=0x%08lx prox=0x%08lx | xor(ant,atual)=0x%08lx",
                                 (unsigned long)(uint32_t)prev_raw,
                                 (unsigned long)(uint32_t)r,
                                 (unsigned long)nx,
                                 (unsigned long)((uint32_t)prev_raw ^ (uint32_t)r));
                        glitch_logged++;
                    }
                }
            }
            prev_raw  = r;
            have_prev = true;
#endif

            /* Palavra I2S de 32 bits -> PCM16 (16 MSB). */
            float x = (float)(r >> 16);

#if AUDIO_DC_BLOCK
            const float y = x - dc_x + R * dc_y;
            dc_x = x;
            dc_y = y;
            x = y;
#endif

            x *= AUDIO_GAIN;

            if (x > 32767.0f) {
                x = 32767.0f;
                clip_count++;
            } else if (x < -32768.0f) {
                x = -32768.0f;
                clip_count++;
            }

            blk[i] = (int16_t)x;

            if (blk[i] < pcm_min) pcm_min = blk[i];
            if (blk[i] > pcm_max) pcm_max = blk[i];
        }

        /* Estatisticas a cada DIAG_BLOCKS blocos (~1 s). */
        diag_blocks++;
        if (diag_blocks >= DIAG_BLOCKS) {
            ESP_LOGI(TAG,
                     "I2S raw: min=%ld max=%ld | PCM16: min=%d max=%d | saturacoes=%lu | "
                     "saltos=%lu | slot=%d pu=%d pd=%d",
                     (long)raw_min, (long)raw_max,
                     (int)pcm_min, (int)pcm_max,
                     (unsigned long)clip_count,
                     (unsigned long)glitch_count,
                     (int)I2S_STD_SLOT_LEFT, AUDIO_DIAG_SD_PULLUP, AUDIO_SD_PULLDOWN);

            raw_min = INT32_MAX;
            raw_max = INT32_MIN;
            pcm_min = INT16_MAX;
            pcm_max = INT16_MIN;
            clip_count    = 0;
            glitch_count  = 0;
            glitch_logged = 0;
            diag_blocks   = 0;
        }

        const int64_t now = esp_timer_get_time();

        /* Buffer circular mono. */
        xSemaphoreTake(s_lock, portMAX_DELAY);
        for (int i = 0; i < frames; i++) {
            s_ring[(s_total + (uint64_t)i) % WINDOW_SAMPLES] = blk[i];
        }
        s_total += (uint64_t)frames;
        s_t_newest_us = now;
        xSemaphoreGive(s_lock);

        /* Notifica a aplicacao a cada STRIDE_SAMPLES. */
        if (s_total >= s_next_notify) {
            s_next_notify += STRIDE_SAMPLES;
            if (xSemaphoreGive(s_stride_sem) != pdTRUE) {
                s_overruns++;
            }
        }
    }
}

esp_err_t audio_init(void)
{
    s_lock       = xSemaphoreCreateMutex();
    s_stride_sem = xSemaphoreCreateBinary();

    if (s_lock == NULL || s_stride_sem == NULL) {
        return ESP_ERR_NO_MEM;
    }

    i2s_chan_config_t cc = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    cc.dma_desc_num  = 6;
    cc.dma_frame_num = BLOCK_FRAMES;

    esp_err_t e = i2s_new_channel(&cc, NULL, &s_rx);
    if (e != ESP_OK) {
        return e;
    }

    /* I2S Philips: 16 kHz, 32 bits, mono (slot esquerdo). */
    i2s_std_config_t sc = {
        .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
                        I2S_DATA_BIT_WIDTH_32BIT,
                        I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = PIN_I2S_SCK,
            .ws   = PIN_I2S_WS,
            .dout = I2S_GPIO_UNUSED,
            .din  = PIN_I2S_SD,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv   = false,
            },
        },
    };

    /* No ESP32-S3 (IDF 5.5.2) a macro deixa slot_mask = BOTH mesmo em mono;
     * selecionamos o slot esquerdo explicitamente (L/R do INMP441 em GND). */
    sc.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;

    e = i2s_channel_init_std_mode(s_rx, &sc);
    if (e != ESP_OK) {
        return e;
    }

#if AUDIO_DIAG_SD_PULLUP
    gpio_pulldown_dis(PIN_I2S_SD);
    gpio_pullup_en(PIN_I2S_SD);
#elif AUDIO_SD_PULLDOWN
    gpio_pullup_dis(PIN_I2S_SD);
    gpio_pulldown_en(PIN_I2S_SD);
#endif

    e = i2s_channel_enable(s_rx);
    if (e != ESP_OK) {
        return e;
    }

    s_next_notify = STRIDE_SAMPLES;

    /* Log de configuracao ANTES de criar a tarefa, em nivel W para ficar visivel. */
    ESP_LOGW(TAG, "Captura I2S iniciada em mono (slot esquerdo).");
    ESP_LOGW(TAG,
             "I2S cfg: fs=%d Hz | 32 bits Philips | slot_mask=%d (1=esq) | "
             "BCLK=GPIO%d WS=GPIO%d SD=GPIO%d | pullup_SD=%d pulldown_SD=%d | diag_saltos=%d",
             SAMPLE_RATE, (int)sc.slot_cfg.slot_mask,
             PIN_I2S_SCK, PIN_I2S_WS, PIN_I2S_SD,
             AUDIO_DIAG_SD_PULLUP, AUDIO_SD_PULLDOWN, AUDIO_DIAG_GLITCH);

    BaseType_t task_result = xTaskCreatePinnedToCore(
        audio_task, "audio", 4096, NULL, 6, NULL, 0);

    if (task_result != pdPASS) {
        return ESP_ERR_NO_MEM;
    }

    return ESP_OK;
}

int audio_wait_stride(uint32_t timeout_ms)
{
    return xSemaphoreTake(s_stride_sem, pdMS_TO_TICKS(timeout_ms)) == pdTRUE ? 0 : -1;
}

int audio_filled(void)
{
    return s_total >= WINDOW_SAMPLES;
}

uint32_t audio_overruns(void)
{
    return s_overruns;
}

void audio_snapshot(int16_t *dst, int64_t *t_newest_us)
{
    xSemaphoreTake(s_lock, portMAX_DELAY);

    const uint64_t start = s_total >= WINDOW_SAMPLES ? s_total - WINDOW_SAMPLES : 0;
    const int first      = (int)(start % WINDOW_SAMPLES);
    const int tail_count = WINDOW_SAMPLES - first;

    memcpy(dst, s_ring + first, (size_t)tail_count * sizeof(int16_t));
    memcpy(dst + tail_count, s_ring, (size_t)first * sizeof(int16_t));

    if (t_newest_us != NULL) {
        *t_newest_us = s_t_newest_us;
    }

    xSemaphoreGive(s_lock);
}