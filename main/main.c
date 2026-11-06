/*
 * Reconhecimento de comandos de voz (DNN PyTorch INT8) no ESP32-S3, ESP-IDF nativo, offline.
 * Fluxo: I2S(INMP441) -> anel -> MFCC (torchaudio-equivalente) -> DNN -> maquina de estados -> LEDs/LCD.
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_system.h"
#include "esp_idf_version.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "config.h"
#include "audio_i2s.h"
#include "mfcc.h"
#include "model_inference.h"
#include "command_state.h"
#include "model_labels.h"
#include "led_control.h"
#include "lcd_i2c.h"

static const char *TAG = "app";
extern const uint8_t model_bin_start[] asm("_binary_model_bin_start");
extern const uint8_t model_bin_end[]   asm("_binary_model_bin_end");

static int16_t s_win[WINDOW_SAMPLES];
static float   s_feat[N_FEATURES];
static cmd_state_t s_state;
static bool s_lcd_ok;

static void lcd2(const char *l0, const char *l1)
{ if (s_lcd_ok) { lcd_print_line(0, l0); lcd_print_line(1, l1); } }

static float window_rms(const int16_t *x, int n)
{
    double acc = 0; for (int i = 0; i < n; i++) { double v = x[i] / 32768.0; acc += v * v; }
    return (float)sqrt(acc / n);
}

static void log_memory(void)
{
    ESP_LOGI(TAG, "heap interno livre=%u (min=%u) | PSRAM livre=%u | pilha livre da tarefa=%u B",
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
             (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL),
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
             (unsigned)uxTaskGetStackHighWaterMark(NULL) * 4u);
}

#if APP_DIAG_MODE == 1
static void diag_level_meter(void)
{
    while (1) {
        if (audio_wait_stride(2000)) continue;
        int64_t t; audio_snapshot(s_win, &t);
        int peak = 0; for (int i = WINDOW_SAMPLES - STRIDE_SAMPLES; i < WINDOW_SAMPLES; i++) { int a = abs(s_win[i]); if (a > peak) peak = a; }
        printf("RMS(0.5s)=%.4f (%.1f dBFS) pico=%d/32768 (%.1f dBFS)\n",
               window_rms(s_win + WINDOW_SAMPLES - RMS_GATE_SAMPLES, RMS_GATE_SAMPLES),
               20 * log10f(window_rms(s_win + WINDOW_SAMPLES - RMS_GATE_SAMPLES, RMS_GATE_SAMPLES) + 1e-9f),
               peak, 20 * log10f(peak / 32768.0f + 1e-9f));
    }
}
#elif APP_DIAG_MODE == 2
/* Linhas: "P,<n>,<int16 ...>" / "F,<3000 floats>" / "L,<logits>" -> tools/comparar_dispositivo.py */
static void diag_capture(void)
{
    float logits[MODEL_NUM_CLASSES];
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(5000));
        ESP_LOGW(TAG, "DIAG: fale/reproduza o comando AGORA (janela de %d ms)", WINDOW_SAMPLES * 1000 / SAMPLE_RATE);
        vTaskDelay(pdMS_TO_TICKS(WINDOW_SAMPLES * 1000 / SAMPLE_RATE + 200));
        int64_t t; audio_snapshot(s_win, &t);
        mfcc_compute(s_win, WINDOW_SAMPLES, 1.0f / 32768.0f, s_feat);   /* ganho/DC ja aplicados na captura */
        model_infer(s_feat, logits);
        printf("P,%d", WINDOW_SAMPLES); for (int i = 0; i < WINDOW_SAMPLES; i++) printf(",%d", s_win[i]); printf("\n");
        printf("F"); for (int i = 0; i < N_FEATURES; i++) printf(",%.5f", s_feat[i]); printf("\n");
        printf("L"); for (int i = 0; i < MODEL_NUM_CLASSES; i++) printf(",%.5f", logits[i]); printf("\n");
    }
}
#endif

void app_main(void)
{
    /* 1) LEDs apagados antes de tudo */
    ESP_ERROR_CHECK(led_init());
    ESP_LOGI(TAG, "ESP-IDF %s | LEDs inicializados (todos apagados)", esp_get_idf_version());

    /* 2) LCD */
    s_lcd_ok = (lcd_init() == ESP_OK);
    if (!s_lcd_ok) ESP_LOGE(TAG, "LCD indisponivel; seguindo so com LEDs e serial");
    lcd2("Iniciando...", "");

    /* 3) Modelo: aponta para a flash ou copia para a PSRAM */
    const uint8_t *blob = model_bin_start;
    size_t blob_sz = (size_t)(model_bin_end - model_bin_start);
#if MODEL_COPY_TO_PSRAM
    uint8_t *ps = heap_caps_aligned_alloc(16, blob_sz, MALLOC_CAP_SPIRAM);
    if (ps) { memcpy(ps, blob, blob_sz); blob = ps; ESP_LOGI(TAG, "pesos copiados da flash para a PSRAM (%u B)", (unsigned)blob_sz); }
    else ESP_LOGW(TAG, "sem PSRAM: lendo pesos direto da flash (mais lento)");
#endif
    int r = model_load(blob, blob_sz);
    if (r || model_num_classes() != MODEL_NUM_CLASSES) {
        ESP_LOGE(TAG, "model_load falhou (%d, classes=%d)", r, model_num_classes());
        lcd2("ERRO modelo", ""); while (1) vTaskDelay(portMAX_DELAY);
    }
    if (mfcc_init()) { ESP_LOGE(TAG, "mfcc_init falhou"); while (1) vTaskDelay(portMAX_DELAY); }

    /* 4) Audio */
    if (audio_init() != ESP_OK) { ESP_LOGE(TAG, "audio_init falhou"); lcd2("ERRO microfone", ""); while (1) vTaskDelay(portMAX_DELAY); }

    cmd_state_init(&s_state, MODEL_CLASSES, MODEL_NUM_CLASSES, CONF_THRESHOLD, CONSEC_REQUIRED, REFRACTORY_MS);
    lcd2("Aguardando", "Hey Carro");
    ESP_LOGI(TAG, "pronto: aguardando \"Hey Carro\" | janela=%d amostras, passo=%d, conf>=%.2f, consec=%d",
             WINDOW_SAMPLES, STRIDE_SAMPLES, CONF_THRESHOLD, CONSEC_REQUIRED);
    log_memory();

#if APP_DIAG_MODE == 1
    diag_level_meter();
#elif APP_DIAG_MODE == 2
    diag_capture();
#endif

    float logits[MODEL_NUM_CLASSES];
    int64_t first_agree_newest_us = 0;      /* instante da amostra mais nova da 1a janela concordante */
    int prev_cand = -1; int wnd = 0; int64_t last_mem_us = 0;

    while (1) {
        if (audio_wait_stride(3000)) { ESP_LOGW(TAG, "sem dados do microfone"); continue; }
        if (!audio_filled()) continue;
        vTaskDelay(1);   /* deixa a IDLE rodar */

        /* ===== INICIO da medicao computacional (t0): comeca o processamento da janela ===== */
        const int64_t t0 = esp_timer_get_time();
        int64_t t_newest;
        audio_snapshot(s_win, &t_newest);
        const float rms = window_rms(s_win + WINDOW_SAMPLES - RMS_GATE_SAMPLES, RMS_GATE_SAMPLES);
        mfcc_compute(s_win, WINDOW_SAMPLES, 1.0f / 32768.0f, s_feat);     /* ganho ja aplicado na captura */
        const int64_t t1 = esp_timer_get_time();                           /* fim do MFCC */
        model_infer(s_feat, logits);
        const int64_t t2 = esp_timer_get_time();                           /* fim da inferencia */
        float conf; const int top = model_top1(logits, MODEL_NUM_CLASSES, &conf);
        cmd_event_t ev = cmd_state_step(&s_state, top, conf, rms >= RMS_GATE, t0 / 1000);
        if (ev.type != EV_NONE) led_set_mask(ev.led_mask);
        const int64_t t3 = esp_timer_get_time();                           /* fim: LED atualizado */
        /* ===== FIM da medicao computacional ===== */

        /* ponta-a-ponta (aproximado): da amostra mais nova da 1a janela concordante ate o LED */
        if (top != prev_cand) { prev_cand = top; first_agree_newest_us = t_newest; }

#if LOG_EVERY_WINDOW
        ESP_LOGI(TAG, "#%d top=%-12s p=%.2f rms=%.4f | mfcc=%.1f ms inf=%.1f ms total=%.1f ms | estado=%s",
                 wnd, MODEL_CLASSES[top].name, conf, rms, (t1 - t0) / 1000.f, (t2 - t1) / 1000.f,
                 (t3 - t0) / 1000.f, s_state.active ? "ATIVO" : "AGUARDANDO");
#endif
        wnd++;
        if (ev.type != EV_NONE) {
            const float comp_ms = (t3 - t0) / 1000.0f;
            const float e2e_ms = (t3 - first_agree_newest_us) / 1000.0f;
            ESP_LOGW(TAG, ">>> %s | conf=%.3f | MFCC=%.1f ms | inferencia=%.1f ms | LATENCIA COMPUTACIONAL=%.1f ms | "
                     "PONTA-A-PONTA(desde a 1a janela concordante)=%.0f ms | LEDs=0x%02X",
                     MODEL_CLASSES[ev.class_idx].name, ev.conf, (t1 - t0) / 1000.f, (t2 - t1) / 1000.f,
                     comp_ms, e2e_ms, ev.led_mask);
            if (ev.type == EV_WAKE) {
                lcd2("Sistema ATIVADO", "Diga o comando");
            } else {
                char l1[24]; snprintf(l1, sizeof l1, "Reacao:%5.1f ms", comp_ms);
                lcd2(MODEL_CLASSES[ev.class_idx].name, l1);              /* LCD atualizado APOS t3 */
            }
            prev_cand = -1;
        }
        const int64_t now = esp_timer_get_time();
        if (now - last_mem_us > 10000000) { last_mem_us = now; log_memory(); ESP_LOGI(TAG, "atrasos de captura (overruns)=%u", (unsigned)audio_overruns()); }
    }
}
