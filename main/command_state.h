#pragma once
#include <stdint.h>
#include <stdbool.h>
#include "config.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef enum { CMD_KIND_OP = 0, CMD_KIND_WAKE = 1 } cmd_kind_t;
typedef struct { const char *name; cmd_kind_t kind; int led; /* 0..4 para LED1..LED5, -1 se nao tem */ } class_info_t;

typedef enum { EV_NONE = 0, EV_WAKE, EV_COMMAND } cmd_event_type_t;
typedef struct {
    cmd_event_type_t type;
    int class_idx;
    float conf;
    uint8_t led_mask;     /* bit0..bit5 = LED1..LED6 (estado desejado apos o evento) */
} cmd_event_t;

typedef struct {
    const class_info_t *cls; int ncls;
    float conf_thr; int consec_req; int refractory_ms;
    bool active;              /* sistema ativado por "Hey Carro" */
    int last_op_led;          /* LED operacional do ultimo comando (-1 nenhum) */
    int cand, cand_count;     /* candidato atual e quantas janelas consecutivas */
    bool armed;               /* pode aceitar (evita repetir a mesma frase) */
    int last_accept_class; int64_t last_accept_ms;
} cmd_state_t;

void cmd_state_init(cmd_state_t *s, const class_info_t *cls, int ncls,
                    float conf_thr, int consec_req, int refractory_ms);
/* Chamar uma vez por janela analisada. energy_ok=false => janela silenciosa. */
cmd_event_t cmd_state_step(cmd_state_t *s, int top_class, float conf, bool energy_ok, int64_t now_ms);
uint8_t cmd_state_led_mask(const cmd_state_t *s);
#ifdef __cplusplus
}
#endif
