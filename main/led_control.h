#pragma once
#include <stdint.h>
#include "esp_err.h"
/* Configura os 6 GPIOs como saida e os coloca em nivel de APAGADO ANTES de qualquer outra coisa. */
esp_err_t led_init(void);
/* bit0..bit5 = LED1..LED6; bit=1 acende. */
void led_set_mask(uint8_t mask);
uint8_t led_get_mask(void);
