# -*- coding: utf-8 -*-
"""
Extração de MFCC a partir dos áudios em augmented_dataset, gerando
10 arquivos CSV DIFERENTES, cada um com uma combinação diferente de
N_MFCC (nº de coeficientes) e MAX_TIME (nº de frames).

É a mesma lógica de extrair_mfcc_para_csv.py, só que repetida para
várias configurações de MFCC — útil para comparar depois qual
combinação dá o melhor resultado no treinamento da DNN.

O que este script faz:
  1. Localiza uma única vez todos os áudios em AUGMENTED_DATASET_PATH
     (cada subpasta = um comando/label).
  2. Para cada uma das 10 combinações (N_MFCC, MAX_TIME) definidas em
     CONFIGURACOES, recalcula o MFCC de TODOS os áudios com aquela
     configuração e salva um CSV próprio em OUTPUT_DIR, nomeado com
     os valores usados (ex.: mfcc_features_nmfcc60_maxtime150.csv).
  3. Cada configuração é processada em paralelo entre vários
     processos, para aproveitar os múltiplos núcleos do processador.

Aviso: como são 10 configurações, o tempo total de execução é
aproximadamente 10x o de rodar extrair_mfcc_para_csv.py uma única vez.

Requisitos (instale antes de rodar):
    pip install torch torchaudio numpy progressbar2

Como usar:
  1. Ajuste AUGMENTED_DATASET_PATH e OUTPUT_DIR abaixo.
  2. Ajuste a lista CONFIGURACOES, se quiser outras combinações.
  3. Rode: python extrair_mfcc_multiplas_configuracoes.py
"""

import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache

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

# Pasta onde os 10 CSVs de saída serão salvos
OUTPUT_DIR = "./mfcc_csvs"

AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".ogg", ".m4a")

SAMPLE_RATE = 16000

# As 10 combinações (N_MFCC, MAX_TIME) que serão testadas.
# Ajuste esta lista livremente — pode ter mais ou menos de 10 itens.
CONFIGURACOES = [
    (20, 100),
    (20, 125),
    (20, 150),

    (30, 100),
    (30, 125),
    (30, 150),

    (40, 100),
    (40, 125),
    (40, 150),

    (60, 100),
    (60, 125),
    (60, 150),
]

# Número de processos paralelos. Deixamos 1 núcleo livre para o
# sistema não travar durante o processamento.
# NUM_WORKERS = max(1, (os.cpu_count() or 4) - 1)
NUM_WORKERS = 1

torch.set_num_threads(1)


# ==========================================================
# 2. Transformação MFCC (parametrizada por N_MFCC)
# ==========================================================

@lru_cache(maxsize=None)
def get_mfcc_transform(n_mfcc):
    """
    Cria (e reaproveita, graças ao lru_cache) o transform de MFCC para
    um dado número de coeficientes. n_mels precisa ser sempre maior
    que n_mfcc, então ele é calculado dinamicamente.
    """
    n_mels = max(64, n_mfcc + 20)
    return torchaudio.transforms.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": 400,
            "hop_length": 150,
            "n_mels": n_mels
        }
    )


def extract_features(path, n_mfcc):
    """Carrega o áudio, converte para mono, reamostra para SAMPLE_RATE
    e calcula a matriz de MFCC (n_mfcc x T)."""
    waveform, sr = torchaudio.load(path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)

    mfcc_transform = get_mfcc_transform(n_mfcc)
    mfcc = mfcc_transform(waveform)
    mfcc = mfcc.squeeze(0)

    return mfcc


def pad_or_truncate(mfcc, max_time):
    """Padroniza o número de frames (colunas) do MFCC para max_time,
    preenchendo com zeros ou cortando o excedente."""
    T = mfcc.shape[1]

    if T < max_time:
        padding = max_time - T
        mfcc = torch.nn.functional.pad(mfcc, (0, padding))
    else:
        mfcc = mfcc[:, :max_time]

    return mfcc


# ==========================================================
# 3. Localizar os áudios (uma subpasta = um label)
# ==========================================================

def find_audio_files_por_label(base_path, audio_extensions=AUDIO_EXTENSIONS):
    """
    Percorre AUGMENTED_DATASET_PATH assumindo que cada subpasta de
    primeiro nível é o nome do comando (label). Retorna uma lista de
    tuplas (caminho_completo, label, nome_do_arquivo).
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
# 4. Worker — processa UM áudio com UMA configuração específica
# ==========================================================

def process_audio(item):
    """
    item = (caminho, label, nome_arquivo, n_mfcc, max_time)
    Extrai o MFCC de um áudio com a configuração dada, padroniza o
    tamanho e o transforma em vetor. Retorna (status, label,
    nome_arquivo, vetor).
    """
    torch.set_num_threads(1)

    caminho, label, nome_arquivo, n_mfcc, max_time = item

    try:
        mfcc = extract_features(caminho, n_mfcc)
        mfcc = pad_or_truncate(mfcc, max_time)
        vetor = mfcc.flatten().numpy().astype(np.float32)
        return ("ok", label, nome_arquivo, vetor)

    except Exception as e:
        print(f"Erro ao processar '{caminho}' (n_mfcc={n_mfcc}, max_time={max_time}): {e}")
        return ("erro", label, nome_arquivo, None)


# ==========================================================
# 5. Processa UMA configuração inteira e gera o CSV correspondente
# ==========================================================

def processar_configuracao(arquivos, n_mfcc, max_time, indice, total_configs):
    n_colunas_mfcc = n_mfcc * max_time
    csv_name = f"mfcc_features_nmfcc{n_mfcc}_maxtime{max_time}.csv"
    csv_path = os.path.join(OUTPUT_DIR, csv_name)
    header = ["arquivo", "label"] + [f"mfcc_{i}" for i in range(n_colunas_mfcc)]

    print(
        f"\n[{indice}/{total_configs}] N_MFCC={n_mfcc}, MAX_TIME={max_time} "
        f"({n_colunas_mfcc} colunas de MFCC) -> {csv_path}"
    )

    tarefas = [
        (caminho, label, nome_arquivo, n_mfcc, max_time)
        for caminho, label, nome_arquivo in arquivos
    ]

    bar = progressbar.ProgressBar(max_value=len(tarefas)) if HAS_PROGRESSBAR else None
    processados = 0
    erros = 0
    concluidos = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {executor.submit(process_audio, t): t for t in tarefas}

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
                elif concluidos % 20 == 0:
                    print(f"  {concluidos}/{len(tarefas)} concluídos...")

    print(f"  -> {processados} vetor(es) salvos em '{csv_path}'.")
    if erros:
        print(f"{erros} arquivo(s) falharam nesta configuração.")

    return csv_path


# ==========================================================
# 6. Rotina principal
# ==========================================================

def main():
    if not os.path.isdir(AUGMENTED_DATASET_PATH):
        raise FileNotFoundError(
            f"A pasta '{AUGMENTED_DATASET_PATH}' não existe. Rode primeiro o "
            f"script de data augmentation, ou ajuste AUGMENTED_DATASET_PATH."
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    arquivos = find_audio_files_por_label(AUGMENTED_DATASET_PATH)
    print(f"Total de arquivos de áudio encontrados em '{AUGMENTED_DATASET_PATH}': {len(arquivos)}")

    if not arquivos:
        print("Nenhum áudio encontrado. Rode o script de data augmentation primeiro.")
        return

    print(f"Serão geradas {len(CONFIGURACOES)} versões do CSV, cada uma com uma combinação de N_MFCC/MAX_TIME.")
    print(f"Cada configuração usa {NUM_WORKERS} processo(s) em paralelo.")

    csvs_gerados = []
    for indice, (n_mfcc, max_time) in enumerate(CONFIGURACOES, start=1):
        csv_path = processar_configuracao(
            arquivos, n_mfcc, max_time, indice, len(CONFIGURACOES)
        )
        csvs_gerados.append(csv_path)

    print(f"\nConcluído! {len(csvs_gerados)} arquivo(s) CSV gerados em '{OUTPUT_DIR}':")
    for caminho in csvs_gerados:
        print(f"  - {caminho}")


if __name__ == "__main__":
    main()
