#pragma once
#include <stdint.h>
#include <stddef.h>
#include "config.h"
#ifdef __cplusplus
extern "C" {
#endif
#define MODEL_MAX_LAYERS 8
/* Valida e mapeia o blob gerado por tools/convert_model.py (formato 'TMLM' v1).
 * O blob precisa permanecer valido e alinhado em 4 bytes. Retorna 0 em sucesso. */
int model_load(const uint8_t *blob, size_t size);
int model_num_classes(void);
/* Executa a rede: feat[N_FEATURES] -> logits[num_classes]. */
void model_infer(const float *feat, float *logits);
/* softmax; devolve indice da classe vencedora e sua probabilidade. */
int model_top1(const float *logits, int n, float *prob);
#ifdef __cplusplus
}
#endif
