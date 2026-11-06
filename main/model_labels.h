/* GERADO por tools/convert_model.py a partir de encoder.joblib - NAO EDITE. */
#pragma once
#include "command_state.h"

#define MODEL_NUM_CLASSES 11

static const class_info_t MODEL_CLASSES[MODEL_NUM_CLASSES] = {
    /*  0 */ { "Abrir Dois", CMD_KIND_OP, 1 },
    /*  1 */ { "Abrir Quatro", CMD_KIND_OP, 3 },
    /*  2 */ { "Abrir Tres", CMD_KIND_OP, 2 },
    /*  3 */ { "Abrir Um", CMD_KIND_OP, 0 },
    /*  4 */ { "Desligar Ar", CMD_KIND_OP, 4 },
    /*  5 */ { "Fechar Dois", CMD_KIND_OP, 1 },
    /*  6 */ { "Fechar Quatro", CMD_KIND_OP, 3 },
    /*  7 */ { "Fechar Tres", CMD_KIND_OP, 2 },
    /*  8 */ { "Fechar Um", CMD_KIND_OP, 0 },
    /*  9 */ { "Hey Carro", CMD_KIND_WAKE, -1 },
    /* 10 */ { "Ligar Ar", CMD_KIND_OP, 4 },
};
