# -*- coding: utf-8 -*-
"""
Data Augmentation de áudio para comandos de voz — versão para rodar
localmente (sem Google Colab / sem Google Drive).

O que este script faz:
  1. Lê os áudios de uma pasta local (dataset_path).
  2. Descobre o rótulo (label) de cada áudio a partir do NOME do arquivo
     (ex.: "hey_carro_01.mp3" -> "Hey Carro").
  3. Para cada áudio original, salva:
       - o próprio áudio original (convertido para mono + SAMPLE_RATE)
       - N_AUMENTADOS versões aumentadas (ruído gaussiano, time shift e
         pitch shift combinados de forma aleatória)
     dentro de uma pasta por comando em augmented_dataset_path.
  4. Antes de processar, verifica quais áudios JÁ têm todos os arquivos
     de saída (original + aumentados) salvos de uma execução anterior
     e PULA esses áudios, processando só os que faltam. Isso permite
     rodar o script de novo (ex.: depois de adicionar novos áudios)
     sem refazer o trabalho já concluído.

Requisitos (instale antes de rodar):
    pip install torch torchaudio numpy progressbar2

Como usar:
  1. Ajuste DATASET_PATH abaixo para a pasta com os áudios originais
     (ex.: "C:/Users/voce/Desktop/dataset" ou "./dataset").
  2. Ajuste AUGMENTED_DATASET_PATH para onde quer salvar o resultado.
  3. Rode: python data_augmentation_local.py
"""

import os
import re
import unicodedata
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

# Pasta local com os áudios originais (não precisa estar organizada em
# subpastas por comando; os arquivos podem estar todos juntos).
DATASET_PATH = "./dataset"

# Pasta onde os áudios originais + aumentados serão salvos
# (uma subpasta será criada para cada um dos 11 comandos).
AUGMENTED_DATASET_PATH = "./augmented_dataset"

SAMPLE_RATE = 16000

# Número total de variações por áudio: 1 original + N_AUMENTADOS aumentadas
N_VARIACOES_TOTAL = 20
N_AUMENTADOS = N_VARIACOES_TOTAL - 1

AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".ogg", ".m4a")

# Número de processos paralelos. Um i5 típico tem 4 núcleos físicos
# (às vezes com hyper-threading, 8 threads). Deixamos 1 núcleo livre
# para o sistema não travar durante o processamento; ajuste se quiser.
#NUM_WORKERS = max(1, (os.cpu_count() or 4) - 1)
NUM_WORKERS = 1 #pelo processador ser antigo, será realizado de forma sequencial

# Cada worker deve usar só 1 thread interna do PyTorch, senão os
# processos competem entre si pelos mesmos núcleos e a paralelização
# não ganha nada (na prática pode até ficar mais lento).
torch.set_num_threads(1)


# ==========================================================
# 2. As 11 saídas esperadas (comandos)
# ==========================================================

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
    arquivos/pastas com os labels esperados."""
    texto = unicodedata.normalize("NFKD", texto).encode("ASCII", "ignore").decode("ASCII")
    return texto.strip().lower().replace(" ", "_").replace("-", "_")


LABEL_LOOKUP = {normaliza_label(l): l for l in LABELS_ESPERADOS}


def label_from_filename(filename):
    """
    Extrai o slug do rótulo a partir do nome do arquivo, removendo a
    extensão e o sufixo numérico final (ex.: '_01', '_12').
    Ex.: 'hey_carro_01.mp3'    -> 'hey_carro'
         'abrir_quatro_07.wav' -> 'abrir_quatro'
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


def ja_processado(base_name, label):
    """
    Verifica se um áudio já tem TODOS os arquivos de saída esperados
    (o original convertido + todas as N_AUMENTADOS versões aumentadas)
    já salvos em AUGMENTED_DATASET_PATH.

    Isso é o que permite rodar o script de novo (por exemplo, depois
    de adicionar áudios novos à pasta original) sem refazer o trabalho
    que já tinha sido feito antes. Se algum dos arquivos esperados
    estiver faltando (ex.: o processo foi interrompido no meio), o
    áudio é considerado incompleto e será reprocessado do zero.
    """
    pasta_label = os.path.join(AUGMENTED_DATASET_PATH, label)

    original_out_path = os.path.join(pasta_label, f"{base_name}_original.wav")
    if not os.path.isfile(original_out_path):
        return False

    for i in range(N_AUMENTADOS):
        aug_out_path = os.path.join(pasta_label, f"{base_name}_aug{i + 1:02d}.wav")
        if not os.path.isfile(aug_out_path):
            return False

    return True


# ==========================================================
# 3. Funções de Data Augmentation
# ==========================================================

def add_gaussian_noise(waveform, noise_level=0.01):
    """Adiciona ruído gaussiano ao waveform."""
    noise = torch.randn_like(waveform) * noise_level
    return waveform + noise


def time_shift(waveform, shift_limit=0.2):
    """Desloca o waveform no tempo por uma quantidade aleatória."""
    _, num_samples = waveform.shape
    shift_amount = int(np.random.uniform(-shift_limit, shift_limit) * num_samples)

    if shift_amount > 0:
        waveform = torch.cat((torch.zeros(1, shift_amount), waveform[:, :-shift_amount]), dim=1)
    elif shift_amount < 0:
        waveform = torch.cat((waveform[:, -shift_amount:], torch.zeros(1, -shift_amount)), dim=1)
    return waveform


def pitch_shift(waveform, sample_rate, n_steps=2):
    """Aplica um deslocamento de pitch aleatório."""
    waveform_cpu = waveform.cpu()
    pitch_shifted_waveform = torchaudio.transforms.PitchShift(
        sample_rate, n_steps=np.random.randint(-n_steps, n_steps + 1)
    )(waveform_cpu)
    return pitch_shifted_waveform.to(waveform.device)


def gerar_audio_aumentado(waveform, sample_rate):
    """
    Gera UMA versão aumentada do áudio, combinando aleatoriamente ruído
    gaussiano, deslocamento temporal e pitch shift, com intensidades
    também aleatórias. Isso garante variações diferentes entre si para
    cada áudio original.
    """
    aug_waveform = waveform.clone()

    tecnicas = []
    if np.random.rand() < 0.7:
        tecnicas.append("noise")
    if np.random.rand() < 0.7:
        tecnicas.append("shift")
    if np.random.rand() < 0.7:
        tecnicas.append("pitch")

    if not tecnicas:
        tecnicas.append(str(np.random.choice(["noise", "shift", "pitch"])))

    np.random.shuffle(tecnicas)

    for tecnica in tecnicas:
        if tecnica == "noise":
            nivel_ruido = np.random.uniform(0.005, 0.02)
            aug_waveform = add_gaussian_noise(aug_waveform, noise_level=nivel_ruido)
        elif tecnica == "shift":
            limite_shift = np.random.uniform(0.1, 0.3)
            aug_waveform = time_shift(aug_waveform, shift_limit=limite_shift)
        elif tecnica == "pitch":
            intensidade_pitch = np.random.randint(1, 4)
            aug_waveform = pitch_shift(aug_waveform, sample_rate, n_steps=intensidade_pitch)

    return aug_waveform


# ==========================================================
# 4. Worker — processa UM arquivo (roda em um processo separado)
# ==========================================================

def process_file(original_path):
    """
    Processa um único áudio: identifica o rótulo, converte para mono +
    SAMPLE_RATE, salva o original e gera as versões aumentadas.

    Roda dentro de um processo worker do ProcessPoolExecutor, então
    esta função (e tudo que ela chama) precisa ser independente de
    estado criado no processo principal após o fork/spawn.

    Retorna uma tupla (status, label_ou_none) onde status é
    "ok", "ignorado" ou "erro".
    """
    # Garante que cada worker também fique restrito a 1 thread do PyTorch
    torch.set_num_threads(1)

    file = os.path.basename(original_path)
    slug = label_from_filename(file)

    if slug not in LABEL_LOOKUP:
        return ("ignorado", None)

    label = LABEL_LOOKUP[slug]
    base_name = os.path.splitext(file)[0]

    try:
        waveform, sr = torchaudio.load(original_path)

        # Converte para mono (se necessário)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Reamostra para SAMPLE_RATE
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform)

        # 1) Salva o áudio ORIGINAL (já pré-processado: mono + SAMPLE_RATE)
        original_out_path = os.path.join(AUGMENTED_DATASET_PATH, label, f"{base_name}_original.wav")
        torchaudio.save(original_out_path, waveform, SAMPLE_RATE)

        # 2) Gera e salva as versões aumentadas
        for i in range(N_AUMENTADOS):
            aug_waveform = gerar_audio_aumentado(waveform, SAMPLE_RATE)
            aug_out_path = os.path.join(
                AUGMENTED_DATASET_PATH, label, f"{base_name}_aug{i + 1:02d}.wav"
            )
            torchaudio.save(aug_out_path, aug_waveform, SAMPLE_RATE)

        return ("ok", label)

    except Exception as e:
        print(f"Erro ao processar '{original_path}': {e}")
        return ("erro", label)


# ==========================================================
# 5. Rotina principal
# ==========================================================

def main():
    if not os.path.isdir(DATASET_PATH):
        raise FileNotFoundError(
            f"A pasta '{DATASET_PATH}' não existe. Ajuste DATASET_PATH no início do script."
        )

    # Cria uma subpasta para cada um dos 11 comandos esperados
    for label in LABELS_ESPERADOS:
        os.makedirs(os.path.join(AUGMENTED_DATASET_PATH, label), exist_ok=True)

    print(f"O dataset (original + aumentado) será salvo em: {AUGMENTED_DATASET_PATH}/")

    arquivos_dataset = find_audio_files(DATASET_PATH)
    print(f"Total de arquivos de áudio encontrados em '{DATASET_PATH}': {len(arquivos_dataset)}")

    if not arquivos_dataset:
        print("Nenhum áudio encontrado. Verifique DATASET_PATH e as extensões suportadas.")
        return

    # ------------------------------------------------------------
    # Antes de processar, verifica quais áudios JÁ têm o data
    # augmentation completo salvo em AUGMENTED_DATASET_PATH, para
    # não refazer o que já foi feito em uma execução anterior.
    # ------------------------------------------------------------
    tarefas = []
    ja_prontos = 0
    ignorados = 0

    for original_path in arquivos_dataset:
        file = os.path.basename(original_path)
        slug = label_from_filename(file)

        if slug not in LABEL_LOOKUP:
            ignorados += 1
            continue

        label = LABEL_LOOKUP[slug]
        base_name = os.path.splitext(file)[0]

        if ja_processado(base_name, label):
            ja_prontos += 1
            continue

        tarefas.append(original_path)

    print(
        f"{ja_prontos} áudio(s) já tinham o data augmentation completo e foram pulados. "
        f"{len(tarefas)} áudio(s) serão processados agora."
    )

    if not tarefas:
        print("Nada a processar — todos os áudios já foram aumentados anteriormente.")
        return

    print(f"Aplicando data augmentation em paralelo usando {NUM_WORKERS} processo(s)...")

    bar = progressbar.ProgressBar(max_value=len(tarefas)) if HAS_PROGRESSBAR else None
    processed = 0
    erros = 0
    concluidos = 0

    # Cada arquivo é uma tarefa independente, distribuída entre os processos.
    # Isso é o que aproveita os múltiplos núcleos do i5: enquanto um processo
    # está calculando o pitch shift de um áudio, outro já está salvando outro.
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_file, path): path for path in tarefas}

        for future in as_completed(futures):
            status, _label = future.result()

            if status == "ok":
                processed += 1
            elif status == "ignorado":
                ignorados += 1
            elif status == "erro":
                erros += 1

            concluidos += 1
            if bar is not None:
                bar.update(concluidos)
            elif concluidos % 10 == 0:
                print(f"  {concluidos}/{len(tarefas)} concluídos...")

    print(
        f"\nAugmentation concluído: {processed} áudios processados, gerando "
        f"{processed * N_VARIACOES_TOTAL} arquivos ({N_VARIACOES_TOTAL} variações por áudio: "
        f"1 original + {N_AUMENTADOS} aumentados) em '{AUGMENTED_DATASET_PATH}/'."
    )
    if ja_prontos:
        print(f"{ja_prontos} arquivo(s) já estavam prontos de uma execução anterior e foram pulados.")
    if ignorados:
        print(
            f"{ignorados} arquivo(s) foram ignorados por não corresponderem a "
            f"nenhum dos 11 comandos esperados."
        )
    if erros:
        print(f"{erros} arquivo(s) falharam durante o processamento (veja as mensagens de erro acima).")


if __name__ == "__main__":
    main()
