# -*- coding: utf-8 -*-
"""
Extração de MFCC (Mel-Frequency Cepstral Coefficients) a partir dos
áudios em augmented_dataset, transformando cada áudio em um vetor de
características e registrando tudo em um arquivo CSV.

Baseado nas configurações e funções de MFCC usadas em
'codigo_tcc_corrigido_final.py' (SAMPLE_RATE, N_MFCC, MAX_TIME,
mfcc_transform, extract_features e pad_or_truncate são os mesmos).

O que este script faz:
  1. Percorre AUGMENTED_DATASET_PATH, onde cada subpasta é o nome de
     um comando (label) e contém os áudios originais + aumentados
     daquele comando (gerados pelo script de data augmentation).
  2. Para cada áudio, calcula a matriz de MFCC (N_MFCC x MAX_TIME),
     e a "achata" (flatten) em um vetor de N_MFCC * MAX_TIME números.
  3. Salva uma linha por áudio no CSV de saída, com o nome do
     arquivo, o rótulo (label) e todos os valores do vetor de MFCC.
  4. O trabalho é paralelizado entre processos (como no script de
     data augmentation), para aproveitar os múltiplos núcleos do
     processador (ex.: um i5).

Requisitos (instale antes de rodar):
    pip install torch torchaudio numpy progressbar2

Como usar:
  1. Ajuste AUGMENTED_DATASET_PATH abaixo, se necessário.
  2. Ajuste CSV_OUTPUT_PATH para onde quer salvar o CSV.
  3. Rode: python extrair_mfcc_para_csv.py
"""

import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import torchaudio

try:
    import progressbar
    HAS_PROGRESSBAR = True
except ImportError:
    HAS_PROGRESSBAR = False


# ==========================================================
# 1. CONFIGURAÇÕES — ajuste aqui para o seu computador
# ==========================================================

# Pasta com os áudios originais + aumentados (gerada pelo script de
# data augmentation), organizada com uma subpasta por comando.
AUGMENTED_DATASET_PATH = "./augmented_dataset"

# Arquivo CSV de saída com os vetores de MFCC
CSV_OUTPUT_PATH = "./mfcc_features.csv"

AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".ogg", ".m4a")

# Mesmas configurações de MFCC usadas no código original
SAMPLE_RATE = 16000
#N_MFCC = 60
#MAX_TIME = 150  # quantidade fixa de frames
#N_MFCC = 20
#MAX_TIME = 100
N_MFCC = 20
MAX_TIME = 150

# Número de processos paralelos. Um i5 típico tem 4 núcleos físicos
# (às vezes com hyper-threading, 8 threads). Deixamos 1 núcleo livre
# para o sistema não travar durante o processamento; ajuste se quiser.
#NUM_WORKERS = max(1, (os.cpu_count() or 4) - 1)
NUM_WORKERS = 1

# Cada worker deve usar só 1 thread interna do PyTorch, senão os
# processos competem entre si pelos mesmos núcleos.
torch.set_num_threads(1)


# ==========================================================
# 2. Transformação MFCC (idêntica ao código original)
# ==========================================================

mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=N_MFCC,
    melkwargs={
        "n_fft": 400,
        "hop_length": 150,
        "n_mels": 64
    }
)


def extract_features(path):
    """Carrega o áudio, converte para mono, reamostra para SAMPLE_RATE
    e calcula a matriz de MFCC (N_MFCC x T)."""
    waveform, sr = torchaudio.load(path)

    # converter para mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # reamostrar
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr,
            new_freq=SAMPLE_RATE
        )
        waveform = resampler(waveform)

    mfcc = mfcc_transform(waveform)

    # remove dimensão do canal
    mfcc = mfcc.squeeze(0)

    return mfcc


def pad_or_truncate(mfcc):
    """Padroniza o número de frames (colunas) do MFCC para MAX_TIME,
    preenchendo com zeros ou cortando o excedente."""
    T = mfcc.shape[1]

    if T < MAX_TIME:
        padding = MAX_TIME - T
        mfcc = torch.nn.functional.pad(mfcc, (0, padding))
    else:
        mfcc = mfcc[:, :MAX_TIME]

    return mfcc


# ==========================================================
# 3. Localizar os áudios (uma subpasta = um label)
# ==========================================================

def find_audio_files_por_label(base_path, audio_extensions=AUDIO_EXTENSIONS):
    """
    Percorre AUGMENTED_DATASET_PATH assumindo que cada subpasta de
    primeiro nível é o nome do comando (label), como é gerado pelo
    script de data augmentation. Retorna uma lista de tuplas
    (caminho_completo, label, nome_do_arquivo).
    """
    arquivos = []
    for label in sorted(os.listdir(base_path)):
        pasta_label = os.path.join(base_path, label)
        if not os.path.isdir(pasta_label):
            continue

        for file in sorted(os.listdir(pasta_label)):
            if file.lower().endswith(audio_extensions):
                caminho = os.path.join(pasta_label, file)
                arquivos.append((caminho, label, file))

    return arquivos


# ==========================================================
# 4. Worker — processa UM áudio (roda em um processo separado)
# ==========================================================

def process_audio(item):
    """
    Extrai o MFCC de um áudio, padroniza o tamanho e o transforma em
    um vetor (flatten). Retorna (status, label, nome_arquivo, vetor).
    """
    torch.set_num_threads(1)

    caminho, label, nome_arquivo = item

    try:
        mfcc = extract_features(caminho)
        mfcc = pad_or_truncate(mfcc)

        # Transforma a matriz (N_MFCC x MAX_TIME) em um vetor 1D
        vetor = mfcc.flatten().numpy().astype(np.float32)

        return ("ok", label, nome_arquivo, vetor)

    except Exception as e:
        print(f"Erro ao processar '{caminho}': {e}")
        return ("erro", label, nome_arquivo, None)


# ==========================================================
# 5. Rotina principal
# ==========================================================

def main():
    if not os.path.isdir(AUGMENTED_DATASET_PATH):
        raise FileNotFoundError(
            f"A pasta '{AUGMENTED_DATASET_PATH}' não existe. Rode primeiro o "
            f"script de data augmentation, ou ajuste AUGMENTED_DATASET_PATH."
        )

    arquivos = find_audio_files_por_label(AUGMENTED_DATASET_PATH)
    print(f"Total de arquivos de áudio encontrados em '{AUGMENTED_DATASET_PATH}': {len(arquivos)}")

    if not arquivos:
        print("Nenhum áudio encontrado. Rode o script de data augmentation primeiro.")
        return

    n_colunas_mfcc = N_MFCC * MAX_TIME
    header = ["arquivo", "label"] + [f"mfcc_{i}" for i in range(n_colunas_mfcc)]

    print(f"Extraindo MFCC ({N_MFCC} coeficientes x {MAX_TIME} frames = {n_colunas_mfcc} colunas por vetor)")
    print(f"Processando em paralelo usando {NUM_WORKERS} processo(s)...")

    bar = progressbar.ProgressBar(max_value=len(arquivos)) if HAS_PROGRESSBAR else None
    processados = 0
    erros = 0
    concluidos = 0

    with open(CSV_OUTPUT_PATH, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

        # Cada áudio é uma tarefa independente, distribuída entre os
        # processos. A escrita no CSV continua sequencial (só o processo
        # principal escreve no arquivo), mas o cálculo pesado do MFCC
        # roda em paralelo nos vários núcleos.
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {executor.submit(process_audio, item): item for item in arquivos}

            for future in as_completed(futures):
                status, label, nome_arquivo, vetor = future.result()

                if status == "ok":
                    writer.writerow([nome_arquivo, label] + vetor.tolist())
                    processados += 1
                else:
                    erros += 1

                concluidos += 1
                if bar is not None:
                    bar.update(concluidos)
                elif concluidos % 10 == 0:
                    print(f"  {concluidos}/{len(arquivos)} concluídos...")

    print(
        f"\nExtração concluída: {processados} vetor(es) de MFCC salvos em '{CSV_OUTPUT_PATH}' "
        f"({n_colunas_mfcc} colunas de MFCC + 'arquivo' + 'label')."
    )
    if erros:
        print(f"{erros} arquivo(s) falharam durante o processamento (veja as mensagens de erro acima).")


if __name__ == "__main__":
    main()
