#include "lcd_i2c.h"
#include "config.h"
#include <ctype.h>
#include <string.h>
#include "driver/i2c_master.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define W 128
#define H 64
#define BUF_SIZE (W * H / 8)
#define I2C_TIMEOUT 100
#define I2C_FREQ 400000

static const char *TAG = "oled";
static i2c_master_bus_handle_t bus;
static i2c_master_dev_handle_t dev;
static uint8_t address;
static uint8_t fb[BUF_SIZE];

/* Fonte 5x7: letras, numeros e sinais usados nas mensagens do main.c. */
static void glyph(char ch, uint8_t g[5])
{
    static const uint8_t blank[5] = {0};
    const uint8_t *p = blank;
#define G(ch_, a,b,c,d,e) case ch_: { static const uint8_t v[5]={a,b,c,d,e}; p=v; break; }
    switch (toupper((unsigned char)ch)) {
    G('A',0x7E,0x11,0x11,0x11,0x7E) G('B',0x7F,0x49,0x49,0x49,0x36)
    G('C',0x3E,0x41,0x41,0x41,0x22) G('D',0x7F,0x41,0x41,0x22,0x1C)
    G('E',0x7F,0x49,0x49,0x49,0x41) G('F',0x7F,0x09,0x09,0x09,0x01)
    G('G',0x3E,0x41,0x49,0x49,0x7A) G('H',0x7F,0x08,0x08,0x08,0x7F)
    G('I',0x00,0x41,0x7F,0x41,0x00) G('J',0x20,0x40,0x41,0x3F,0x01)
    G('K',0x7F,0x08,0x14,0x22,0x41) G('L',0x7F,0x40,0x40,0x40,0x40)
    G('M',0x7F,0x02,0x0C,0x02,0x7F) G('N',0x7F,0x04,0x08,0x10,0x7F)
    G('O',0x3E,0x41,0x41,0x41,0x3E) G('P',0x7F,0x09,0x09,0x09,0x06)
    G('Q',0x3E,0x41,0x51,0x21,0x5E) G('R',0x7F,0x09,0x19,0x29,0x46)
    G('S',0x46,0x49,0x49,0x49,0x31) G('T',0x01,0x01,0x7F,0x01,0x01)
    G('U',0x3F,0x40,0x40,0x40,0x3F) G('V',0x1F,0x20,0x40,0x20,0x1F)
    G('W',0x3F,0x40,0x38,0x40,0x3F) G('X',0x63,0x14,0x08,0x14,0x63)
    G('Y',0x07,0x08,0x70,0x08,0x07) G('Z',0x61,0x51,0x49,0x45,0x43)
    G('0',0x3E,0x51,0x49,0x45,0x3E) G('1',0x00,0x42,0x7F,0x40,0x00)
    G('2',0x42,0x61,0x51,0x49,0x46) G('3',0x21,0x41,0x45,0x4B,0x31)
    G('4',0x18,0x14,0x12,0x7F,0x10) G('5',0x27,0x45,0x45,0x45,0x39)
    G('6',0x3C,0x4A,0x49,0x49,0x30) G('7',0x01,0x71,0x09,0x05,0x03)
    G('8',0x36,0x49,0x49,0x49,0x36) G('9',0x06,0x49,0x49,0x29,0x1E)
    G(':',0x00,0x36,0x36,0x00,0x00) G('.',0x00,0x60,0x60,0x00,0x00)
    G(',',0x00,0x40,0x20,0x00,0x00) G('-',0x08,0x08,0x08,0x08,0x08)
    G('_',0x40,0x40,0x40,0x40,0x40) G('/',0x20,0x10,0x08,0x04,0x02)
    G('!',0x00,0x00,0x5F,0x00,0x00) G('?',0x02,0x01,0x51,0x09,0x06)
    G('>',0x00,0x41,0x22,0x14,0x08) G('<',0x08,0x14,0x22,0x41,0x00)
    default: break;
    }
#undef G
    memcpy(g, p, 5);
}

static esp_err_t send_cmds(const uint8_t *data, size_t n)
{
    uint8_t packet[32];
    if (!dev || n > sizeof(packet)-1) return ESP_ERR_INVALID_STATE;
    packet[0] = 0x00;
    memcpy(packet+1, data, n);
    return i2c_master_transmit(dev, packet, n+1, I2C_TIMEOUT);
}

static esp_err_t refresh(void)
{
    const uint8_t window[] = {0x21,0,127,0x22,0,7};
    esp_err_t e = send_cmds(window, sizeof(window));
    if (e != ESP_OK) return e;
    uint8_t packet[129];
    packet[0] = 0x40;
    for (int i=0; i<BUF_SIZE; i+=128) {
        memcpy(packet+1, fb+i, 128);
        e = i2c_master_transmit(dev, packet, sizeof(packet), I2C_TIMEOUT);
        if (e != ESP_OK) return e;
    }
    return ESP_OK;
}

static void pixel_char(int x, int y, char ch)
{
    uint8_t g[5]; glyph(ch, g);
    for (int col=0; col<6; ++col) {
        uint8_t bits = col<5 ? g[col] : 0;
        for (int row=0; row<8; ++row) {
            int xx=x+col, yy=y+row;
            if (xx<0 || xx>=W || yy<0 || yy>=H) continue;
            size_t k=(size_t)(yy/8)*W+xx;
            uint8_t m=(uint8_t)(1u<<(yy%8));
            if (bits & (1u<<row)) fb[k] |= m;
            else fb[k] &= (uint8_t)~m;
        }
    }
}

esp_err_t lcd_init(void)
{
    if (dev) return ESP_OK;
    i2c_master_bus_config_t bc = {
        .i2c_port=I2C_NUM_0, .sda_io_num=PIN_LCD_SDA, .scl_io_num=PIN_LCD_SCL,
        .clk_source=I2C_CLK_SRC_DEFAULT, .glitch_ignore_cnt=7,
        .flags.enable_internal_pullup=true
    };
    esp_err_t e=i2c_new_master_bus(&bc,&bus);
    if (e!=ESP_OK) { ESP_LOGE(TAG,"i2c_new_master_bus: %s",esp_err_to_name(e)); return e; }

    ESP_LOGI(TAG, "Iniciando varredura I2C...");

    for (uint8_t addr = 0x08; addr < 0x78; addr++) {
        esp_err_t probe = i2c_master_probe(bus, addr, 50);

        if (probe == ESP_OK) {
            ESP_LOGW(TAG, "Dispositivo I2C encontrado: 0x%02X", addr);

            if (addr == 0x3C || addr == 0x3D) {
                address = addr;
            }
        }
    }
    if (!address) {
        ESP_LOGE(TAG,"OLED nao respondeu em 0x3C/0x3D; verifique alimentacao e SDA/SCL");
        return ESP_ERR_NOT_FOUND;
    }

    i2c_device_config_t dc={
        .dev_addr_length=I2C_ADDR_BIT_LEN_7, .device_address=address,
        .scl_speed_hz=I2C_FREQ
    };
    e=i2c_master_bus_add_device(bus,&dc,&dev);
    if (e!=ESP_OK) { dev=NULL; return e; }
    vTaskDelay(pdMS_TO_TICKS(100));

    const uint8_t init[]={0xAE,0xD5,0x80,0xA8,0x3F,0xD3,0x00,0x40,
        0x8D,0x14,0x20,0x00,0xA1,0xC8,0xDA,0x12,0x81,0x7F,
        0xD9,0xF1,0xDB,0x40,0xA4,0xA6,0x2E,0xAF};
    e=send_cmds(init,sizeof(init));
    if (e!=ESP_OK) { ESP_LOGE(TAG,"inicializacao OLED: %s",esp_err_to_name(e)); return e; }
    memset(fb,0,sizeof(fb));
    e=refresh();
    if (e==ESP_OK) ESP_LOGI(TAG,"OLED 128x64 inicializado em 0x%02X (SSD1306)",address);
    return e;
}

uint8_t lcd_get_address(void) { return address; }

void lcd_clear(void)
{
    if (!dev) return;
    memset(fb,0,sizeof(fb));
    esp_err_t e=refresh();
    if (e!=ESP_OK) ESP_LOGW(TAG,"limpeza OLED: %s",esp_err_to_name(e));
}

void lcd_print_line(int row, const char *text)
{
    if (!dev) return;
    if (row<0 || row>=8) { ESP_LOGW(TAG,"linha invalida: %d",row); return; }
    int y=row*8;
    for (int yy=y; yy<y+8; ++yy)
        for (int x=0; x<W; ++x)
            fb[(size_t)(yy/8)*W+x] &= (uint8_t)~(1u<<(yy%8));

    if (text) {
        for (int i=0; i<21 && text[i]; ++i)
            pixel_char(i*6,y,text[i]);
    }
    esp_err_t e=refresh();
    if (e!=ESP_OK) ESP_LOGW(TAG,"atualizacao OLED: %s",esp_err_to_name(e));
}