#pragma once
#include <stdint.h>
#include "esp_err.h"

/* OLED I2C 128x64 SSD1306; API mantida para compatibilidade com main.c. */
esp_err_t lcd_init(void);
uint8_t lcd_get_address(void);
void lcd_clear(void);
void lcd_print_line(int row, const char *text);