# -*- coding: utf-8 -*-
"""
Verifica, na pasta do dataset, quais arquivos de áudio têm um nome que
NÃO corresponde a nenhum dos 11 comandos esperados, e imprime o nome
desses arquivos na tela.

Um arquivo é considerado "válido" quando, removendo a extensão e o
sufixo numérico final (ex.: '_01'), o nome bate com um dos comandos
esperados (ignorando acentos, maiúsculas/minúsculas e espaços/hífens).
Ex.: 'hey_carro_01.mp3' -> ok (bate com "Hey Carro")
     'ligar_ar01.wav'   -> pode não bater, dependendo do padrão usado
     'abrircarro_1.mp3' -> não bate com nenhum comando -> é listado

Como usar:
  1. Ajuste DATASET_PATH abaixo para a pasta com os áudios.
  2. Rode: python verificar_nomes_dataset.py
"""

import os
import re
import unicodedata

# ==========================================================
# Configuração
# ==========================================================

# Pasta local com os áudios a verificar
DATASET_PATH = "./dataset"

AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".ogg", ".m4a")

LABELS_ESPERADOS = [
    "Hey Carro",
    "Abrir Um",
    "Abrir Dois",
    "Abrir Três",
    "Abrir Quatro",
    "Fechar Um",
    "Fechar Dois",
    "Fechar Três",
    "Fechar Quatro",
    "Ligar Ar",
    "Desligar Ar",
]


def normaliza_label(texto):
    """Remove acentos/maiúsculas/espaços, para comparar nomes de
    arquivos com os labels esperados."""
    texto = unicodedata.normalize("NFKD", texto).encode("ASCII", "ignore").decode("ASCII")
    return texto.strip().lower().replace(" ", "_").replace("-", "_")


LABEL_LOOKUP = {normaliza_label(l): l for l in LABELS_ESPERADOS}


def label_from_filename(filename):
    """
    Extrai o slug do rótulo a partir do nome do arquivo, removendo a
    extensão e o sufixo numérico final (ex.: '_01', '_12').
    """
    base = os.path.splitext(filename)[0]
    base = re.sub(r"_\d+$", "", base)
    return normaliza_label(base)


def find_audio_files(base_path, audio_extensions=AUDIO_EXTENSIONS):
    """Percorre recursivamente as subpastas de base_path e retorna os
    caminhos dos áudios encontrados."""
    arquivos = []
    for root, _, files in os.walk(base_path):
        for f in files:
            if f.lower().endswith(audio_extensions):
                arquivos.append(os.path.join(root, f))
    return arquivos


def main():
    if not os.path.isdir(DATASET_PATH):
        raise FileNotFoundError(
            f"A pasta '{DATASET_PATH}' não existe. Ajuste DATASET_PATH no início do script."
        )

    arquivos = find_audio_files(DATASET_PATH)
    print(f"Total de arquivos de áudio encontrados em '{DATASET_PATH}': {len(arquivos)}\n")

    invalidos = []

    for caminho in arquivos:
        nome_arquivo = os.path.basename(caminho)
        slug = label_from_filename(nome_arquivo)

        if slug not in LABEL_LOOKUP:
            invalidos.append(caminho)

    if invalidos:
        print(f"{len(invalidos)} arquivo(s) com nome fora do padrão esperado:\n")
        for caminho in invalidos:
            print(f"  - {os.path.basename(caminho)}   (caminho: {caminho})")
    else:
        print("Todos os arquivos correspondem a algum dos 11 comandos esperados.")


if __name__ == "__main__":
    main()
