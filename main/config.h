/* Parametros centrais do projeto. Tudo que precisa ser calibrado esta aqui. */
#pragma once

/* ---------- Pinos ---------- */

/* Microfone INMP441 */
#define PIN_I2S_SD    2
#define PIN_I2S_WS    39
#define PIN_I2S_SCK   41

/* Visor I2C */
#define PIN_LCD_SCL   4
#define PIN_LCD_SDA   15

/* LEDs */
#define PIN_LED1 13
#define PIN_LED2 47
#define PIN_LED3 9
#define PIN_LED4 11
#define PIN_LED5 17
#define PIN_LED6 8

#define LED_ACTIVE_LEVEL 1

/* ---------- Audio / MFCC ---------- */
#define SAMPLE_RATE   16000
#define N_FFT         400
#define HOP_LENGTH    150
#define N_MELS        64
#define N_MFCC        20
#define MAX_TIME      150
#define N_FREQS       (N_FFT / 2 + 1)
#define N_FEATURES    (N_MFCC * MAX_TIME)

/* Janela analisada a cada inferencia */
#ifndef WINDOW_SAMPLES
#define WINDOW_SAMPLES   24000
#endif

#define STRIDE_SAMPLES 4000
#define MAX_FRAMES (1 + WINDOW_SAMPLES / HOP_LENGTH)

/* Entrada I2S */
#define AUDIO_GAIN        1.0f
#define AUDIO_DC_BLOCK    1

/* ---------- Decisao / estabilidade ---------- */
#define CONF_THRESHOLD     0.85f
#define CONSEC_REQUIRED    2
#define REFRACTORY_MS      1000
#define RMS_GATE           0.010f
#define RMS_GATE_SAMPLES   8000

/* ---------- Modelo ---------- */
#define MODEL_COPY_TO_PSRAM 1

#ifndef MODEL_INT8_ACTIVATIONS
#define MODEL_INT8_ACTIVATIONS 0
#endif

/* ---------- Diagnostico ---------- */
#define APP_DIAG_MODE 1
#define LOG_EVERY_WINDOW 1