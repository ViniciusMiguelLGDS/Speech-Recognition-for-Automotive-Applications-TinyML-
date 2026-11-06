/*
 * Maquina de estados (ver README):
 *   AGUARDANDO --("Hey Carro" confiavel)--> ATIVO (LED6 aceso, permanece ate reiniciar)
 *   ATIVO: cada comando operacional aceito acende SO o LED do ultimo comando (+LED6).
 * Anti falso-positivo: energia minima, confianca minima, N janelas consecutivas com a mesma
 * classe, periodo refratario e "re-armamento" (a mesma frase mantida na janela nao dispara 2x).
 */
#include "command_state.h"

void cmd_state_init(cmd_state_t *s, const class_info_t *cls, int ncls,
                    float conf_thr, int consec_req, int refractory_ms)
{
    s->cls = cls; s->ncls = ncls; s->conf_thr = conf_thr; s->consec_req = consec_req;
    s->refractory_ms = refractory_ms; s->active = false; s->last_op_led = -1;
    s->cand = -1; s->cand_count = 0; s->armed = true; s->last_accept_class = -1;
    s->last_accept_ms = -1000000;
}

uint8_t cmd_state_led_mask(const cmd_state_t *s)
{
    uint8_t m = 0;
    if (s->active) m |= (1u << 5);
    if (s->active && s->last_op_led >= 0) m |= (1u << s->last_op_led);
    return m;
}

cmd_event_t cmd_state_step(cmd_state_t *s, int top, float conf, bool energy_ok, int64_t now_ms)
{
    cmd_event_t ev = { EV_NONE, -1, 0.f, cmd_state_led_mask(s) };
    const bool valid_class = top >= 0 && top < s->ncls;
    /* classe relevante no estado atual? (aguardando: so Hey Carro; ativo: so comandos operacionais) */
    bool relevant = false;
    if (valid_class) relevant = s->active ? (s->cls[top].kind == CMD_KIND_OP)
                                          : (s->cls[top].kind == CMD_KIND_WAKE);
    const bool good = energy_ok && conf >= s->conf_thr && relevant;

    if (!good) {                      /* janela rejeitada: solta o candidato e re-arma */
        s->cand = -1; s->cand_count = 0; s->armed = true;
        return ev;
    }
    if (top != s->last_accept_class) s->armed = true;   /* palavra diferente => pode aceitar */

    if (top == s->cand) s->cand_count++; else { s->cand = top; s->cand_count = 1; }

    if (s->cand_count >= s->consec_req && s->armed &&
        (now_ms - s->last_accept_ms) >= s->refractory_ms) {
        s->armed = false; s->last_accept_class = top; s->last_accept_ms = now_ms;
        s->cand = -1; s->cand_count = 0;
        ev.class_idx = top; ev.conf = conf;
        if (s->cls[top].kind == CMD_KIND_WAKE) { s->active = true; ev.type = EV_WAKE; }
        else { s->last_op_led = s->cls[top].led; ev.type = EV_COMMAND; }
        ev.led_mask = cmd_state_led_mask(s);
    }
    return ev;
}
