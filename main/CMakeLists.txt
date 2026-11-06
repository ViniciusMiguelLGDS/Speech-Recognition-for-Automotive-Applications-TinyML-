idf_component_register(
    SRCS "main.c" "audio_i2s.c" "mfcc.c" "model_inference.c" "led_control.c" "lcd_i2c.c" "command_state.c"
    INCLUDE_DIRS "."
    REQUIRES esp_driver_i2s esp_driver_i2c esp_driver_gpio esp_timer esp_psram heap log
    EMBED_FILES "../model/model.bin"   # gera _binary_model_bin_start / _end (fica na flash)
)
# Otimizacao para os lacos de MFCC/DNN
target_compile_options(${COMPONENT_LIB} PRIVATE -O2 -Wall -Wextra -Wno-unused-parameter)
