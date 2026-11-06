#include "led_control.h"
#include "config.h"
#include "driver/gpio.h"

static const gpio_num_t LED_PINS[6] = { PIN_LED1, PIN_LED2, PIN_LED3, PIN_LED4, PIN_LED5, PIN_LED6 };
static uint8_t s_mask;

esp_err_t led_init(void)
{
    for (int i = 0; i < 6; i++) {
        gpio_reset_pin(LED_PINS[i]);
        /* nivel apagado ANTES de virar saida, evitando pulso na inicializacao */
        gpio_set_level(LED_PINS[i], !LED_ACTIVE_LEVEL);
        gpio_config_t c = { .pin_bit_mask = 1ULL << LED_PINS[i], .mode = GPIO_MODE_OUTPUT,
                            .pull_up_en = GPIO_PULLUP_DISABLE, .pull_down_en = GPIO_PULLDOWN_DISABLE,
                            .intr_type = GPIO_INTR_DISABLE };
        esp_err_t e = gpio_config(&c);
        if (e != ESP_OK) return e;
        gpio_set_level(LED_PINS[i], !LED_ACTIVE_LEVEL);
    }
    s_mask = 0;
    return ESP_OK;
}

void led_set_mask(uint8_t mask)
{
    for (int i = 0; i < 6; i++)
        gpio_set_level(LED_PINS[i], ((mask >> i) & 1) ? LED_ACTIVE_LEVEL : !LED_ACTIVE_LEVEL);
    s_mask = mask;
}
uint8_t led_get_mask(void) { return s_mask; }
