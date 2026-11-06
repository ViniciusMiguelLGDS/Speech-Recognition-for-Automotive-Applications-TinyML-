# -*- coding: utf-8 -*-
"""
Extração de MFCC (N_MFCC = 20, MAX_TIME = 150) a partir do
augmented_dataset, já dividindo em TREINO e TESTE POR PESSOA:
  - 8 pessoas -> treino
  - 2 pessoas -> teste

Por que dividir por pessoa (e não aleatoriamente por áudio)?
Os áudios aumentados são derivados dos originais. Se a divisão fosse
aleatória, uma versão aumentada de um áudio poderia ir para o treino
e o original (ou outra versão dele) para o teste, o que causa
vazamento de dados (data leakage) e infla a acurácia. Dividindo por
pessoa, o teste mede de fato a generalização para vozes nunca vistas.

Como a pessoa é identificada:
  Cada comando tem 100 áudios originais (10 por pessoa), numerados de
  01 a 100 no nome do arquivo (ex.: 'hey_carro_37.mp3'). Assume-se que
  os áudios de cada pessoa são consecutivos:
      01-10 -> pessoa 1, 11-20 -> pessoa 2, ..., 91-100 -> pessoa 10
  Se a numeração do seu dataset for diferente (ex.: intercalada),
  troque MODO_IDENTIFICACAO_PESSOA ou edite a função identifica_pessoa().

Saídas:
  mfcc_treino.csv  e  mfcc_teste.csv
  Colunas: arquivo, label, pessoa, mfcc_0 ... mfcc_2999
  (a coluna 'pessoa' serve só para conferência; remova-a antes de
  treinar o modelo).

Requisitos:
    pip install torch torchaudio numpy progressbar2
"""

import csv
import os
import re
from collections import Counter
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
# 1. CONFIGURAÇÕES
# ==========================================================

AUGMENTED_DATASET_PATH = "./augmented_dataset"

CSV_TREINO_PATH = "./mfcc_treino.csv"
CSV_TESTE_PATH = "./mfcc_teste.csv"

AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".ogg", ".m4a")

SAMPLE_RATE = 16000
N_MFCC = 20
MAX_TIME = 150

NUM_WORKERS = 1  # processador antigo: processamento sequencial

torch.set_num_threads(1)

# ---------- Divisão por pessoa ----------
N_PESSOAS = 10
AUDIOS_POR_PESSOA_POR_LABEL = 10  # 100 áudios / 10 pessoas

# Pessoas (1 a 10) que irão para o TESTE. As outras 8 vão para o treino.
PESSOAS_TESTE = [9, 10]

# "bloco"      -> 01-10 = pessoa 1, 11-20 = pessoa 2, ...
# "intercalado"-> pessoa = ((numero - 1) % 10) + 1  (01, 11, 21... = pessoa 1)
MODO_IDENTIFICACAO_PESSOA = "bloco"

# True  -> o teste usa apenas os áudios ORIGINAIS das 2 pessoas
#          (recomendado: o teste reflete áudios reais, sem augmentation).
# False -> o teste também inclui as versões aumentadas.
TESTE_APENAS_ORIGINAIS = True


# ==========================================================
# 2. Transformação MFCC
# ==========================================================

mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=N_MFCC,
    melkwargs={
        "n_fft": 400,
        "hop_length": 150,
        "n_mels": 64,
    },
)


def extract_features(path):
    waveform, sr = torchaudio.load(path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=SAMPLE_RATE
        )(waveform)

    mfcc = mfcc_transform(waveform)
    return mfcc.squeeze(0)


def pad_or_truncate(mfcc):
    T = mfcc.shape[1]
    if T < MAX_TIME:
        mfcc = torch.nn.functional.pad(mfcc, (0, MAX_TIME - T))
    else:
        mfcc = mfcc[:, :MAX_TIME]
    return mfcc


# ==========================================================
# 3. Identificação da pessoa a partir do nome do arquivo
# ==========================================================

# Ex.: 'hey_carro_37_original.wav' / 'hey_carro_37_aug05.wav'
RE_SAIDA = re.compile(r"^(?P<base>.+)_(?P<tipo>original|aug\d+)$")
RE_NUMERO = re.compile(r"^(?P<slug>.+)_(?P<num>\d+)$")


def parse_nome(nome_arquivo):
    """Retorna (numero_do_audio_original, eh_original) ou None se o
    nome não seguir o padrão esperado."""
    nome = os.path.splitext(nome_arquivo)[0]

    m = RE_SAIDA.match(nome)
    if not m:
        return None
    eh_original = m.group("tipo") == "original"

    m2 = RE_NUMERO.match(m.group("base"))
    if not m2:
        return None

    return int(m2.group("num")), eh_original


def identifica_pessoa(numero):
    """Converte o número do áudio original (1..100) em pessoa (1..10)."""
    if MODO_IDENTIFICACAO_PESSOA == "intercalado":
        pessoa = ((numero - 1) % N_PESSOAS) + 1
    else:  # "bloco"
        pessoa = ((numero - 1) // AUDIOS_POR_PESSOA_POR_LABEL) + 1

    return pessoa if 1 <= pessoa <= N_PESSOAS else None


# ==========================================================
# 4. Localizar os áudios e decidir treino/teste
# ==========================================================

def listar_tarefas(base_path):
    """
    Retorna (tarefas, ignorados), onde cada tarefa é uma tupla
    (caminho, label, nome_arquivo, pessoa, split).
    """
    tarefas = []
    ignorados = 0

    for label in sorted(os.listdir(base_path)):
        pasta_label = os.path.join(base_path, label)
        if not os.path.isdir(pasta_label):
            continue

        for file in sorted(os.listdir(pasta_label)):
            if not file.lower().endswith(AUDIO_EXTENSIONS):
                continue

            info = parse_nome(file)
            if info is None:
                ignorados += 1
                continue

            numero, eh_original = info
            pessoa = identifica_pessoa(numero)
            if pessoa is None:
                ignorados += 1
                continue

            if pessoa in PESSOAS_TESTE:
                if TESTE_APENAS_ORIGINAIS and not eh_original:
                    # versões aumentadas das pessoas de teste não são usadas
                    continue
                split = "teste"
            else:
                split = "treino"

            tarefas.append((os.path.join(pasta_label, file), label, file, pessoa, split))

    return tarefas, ignorados


# ==========================================================
# 5. Worker
# ==========================================================

def process_audio(item):
    torch.set_num_threads(1)
    caminho, label, nome_arquivo, pessoa, split = item

    try:
        mfcc = pad_or_truncate(extract_features(caminho))
        vetor = mfcc.flatten().numpy().astype(np.float32)
        return ("ok", label, nome_arquivo, pessoa, split, vetor)
    except Exception as e:
        print(f"Erro ao processar '{caminho}': {e}")
        return ("erro", label, nome_arquivo, pessoa, split, None)


# ==========================================================
# 6. Rotina principal
# ==========================================================

def main():
    if not os.path.isdir(AUGMENTED_DATASET_PATH):
        raise FileNotFoundError(
            f"A pasta '{AUGMENTED_DATASET_PATH}' não existe. Rode primeiro o "
            f"data augmentation, ou ajuste AUGMENTED_DATASET_PATH."
        )

    tarefas, ignorados = listar_tarefas(AUGMENTED_DATASET_PATH)
    if not tarefas:
        print("Nenhum áudio válido encontrado.")
        return

    pessoas_treino = [p for p in range(1, N_PESSOAS + 1) if p not in PESSOAS_TESTE]
    print(f"Pessoas no TREINO: {pessoas_treino}")
    print(f"Pessoas no TESTE : {sorted(PESSOAS_TESTE)}")
    print(f"Áudios a processar: {len(tarefas)} "
          f"(treino: {sum(t[4] == 'treino' for t in tarefas)}, "
          f"teste: {sum(t[4] == 'teste' for t in tarefas)})")

    n_colunas = N_MFCC * MAX_TIME
    header = ["arquivo", "label", "pessoa"] + [f"mfcc_{i}" for i in range(n_colunas)]

    bar = progressbar.ProgressBar(max_value=len(tarefas)) if HAS_PROGRESSBAR else None
    contagem = Counter()
    erros = 0
    concluidos = 0

    with open(CSV_TREINO_PATH, "w", newline="", encoding="utf-8") as f_treino, \
         open(CSV_TESTE_PATH, "w", newline="", encoding="utf-8") as f_teste:

        writers = {
            "treino": csv.writer(f_treino),
            "teste": csv.writer(f_teste),
        }
        for w in writers.values():
            w.writerow(header)

        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(process_audio, t) for t in tarefas]

            for future in as_completed(futures):
                status, label, nome, pessoa, split, vetor = future.result()

                if status == "ok":
                    writers[split].writerow([nome, label, pessoa] + vetor.tolist())
                    contagem[(split, label)] += 1
                else:
                    erros += 1

                concluidos += 1
                if bar is not None:
                    bar.update(concluidos)
                elif concluidos % 10 == 0:
                    print(f"  {concluidos}/{len(tarefas)} concluídos...")

    total_treino = sum(v for (s, _), v in contagem.items() if s == "treino")
    total_teste = sum(v for (s, _), v in contagem.items() if s == "teste")

    print(f"\nTreino: {total_treino} vetores -> '{CSV_TREINO_PATH}'")
    print(f"Teste : {total_teste} vetores -> '{CSV_TESTE_PATH}'")
    print("\nVetores por label (treino / teste):")
    labels = sorted({l for (_, l) in contagem})
    for l in labels:
        print(f"  {l:15s} {contagem[('treino', l)]:5d} / {contagem[('teste', l)]:5d}")

    if ignorados:
        print(f"{ignorados} arquivo(s) ignorados (nome fora do padrão esperado).")
    if erros:
        print(f"{erros} arquivo(s) falharam (veja as mensagens acima).")


if __name__ == "__main__":
    main()
