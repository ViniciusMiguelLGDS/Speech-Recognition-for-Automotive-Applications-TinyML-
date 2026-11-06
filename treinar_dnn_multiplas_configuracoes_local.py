# -*- coding: utf-8 -*-
"""
Treinamento da DNN — Comparando várias configurações de MFCC
(versão para rodar LOCALMENTE, sem Google Colab / sem Google Drive)

Lê o arquivo mfcc_csvs.zip (local, no seu computador) contendo os
CSVs gerados por extrair_mfcc_multiplas_configuracoes.py, cada um com
uma combinação diferente de N_MFCC/MAX_TIME. Para cada CSV encontrado:

  1. Reconstrói as matrizes de MFCC e separa treino/validação/teste.
  2. Treina a mesma DNN do codigo_tcc_corrigido_final.py.
  3. Salva localmente o melhor modelo DAQUELA configuração sempre que
     a validação melhora.
  4. Usa early stopping: o treino para sozinho quando a rede converge
     (validação para de melhorar por várias épocas seguidas).

No final, compara os resultados de todas as configurações (acurácia
no teste), salva uma tabela comparativa e gráficos em disco, e copia
o melhor modelo geral para um arquivo próprio.

Requisitos (instale antes de rodar):
    pip install torch torchaudio pandas numpy scikit-learn matplotlib seaborn tabulate joblib

Como usar:
  1. Ajuste MFCC_ZIP_PATH e OUTPUT_DIR abaixo.
  2. Rode: python treinar_dnn_multiplas_configuracoes_local.py
"""

import os
import re
import glob
import shutil
import zipfile

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # gera os gráficos em arquivo mesmo sem display gráfico
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from joblib import dump

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ==========================================================
# 1. CONFIGURAÇÕES — ajuste aqui para o seu computador
# ==========================================================

# Caminho local do .zip com os CSVs gerados por
# extrair_mfcc_multiplas_configuracoes.py
MFCC_ZIP_PATH = "./mfcc_csvs.zip"

# Pasta local onde o zip será extraído
EXTRACT_DIR = "./mfcc_csvs"

# Pasta local onde os modelos, o encoder, a tabela comparativa e os
# gráficos serão salvos
OUTPUT_DIR = "./modelos_mfcc"

# Hiperparâmetros de treino (usados em CADA configuração)
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS_MAX = 300      # teto de épocas; na prática o early stopping deve parar antes
PATIENCE = 15         # nº de épocas sem melhora na validação até considerar convergência
MIN_DELTA = 1e-4      # melhora mínima na loss de validação para contar como "melhorou"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================
# 2. Dataset e Arquitetura da DNN
# ==========================================================

class VoiceDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        x = x.unsqueeze(0)  # adiciona a dimensão de canal
        return x.float(), self.y[idx]


class DNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.network = nn.Sequential(

            # Camada 1
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Camada 2
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Camada 3
            nn.Linear(256, 128),
            nn.ReLU(),

            # Saída
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)


# ==========================================================
# 3. Extrair o zip e localizar todos os CSVs (uma configuração cada)
# ==========================================================

PADRAO_NOME = re.compile(r"nmfcc(\d+)_maxtime(\d+)")


def extrair_zip_e_localizar_csvs():
    if not os.path.isfile(MFCC_ZIP_PATH):
        raise FileNotFoundError(
            f"O arquivo '{MFCC_ZIP_PATH}' não foi encontrado. Ajuste MFCC_ZIP_PATH "
            f"no início do script."
        )

    # Limpa completamente a pasta de extração antes de descompactar.
    # Sem isso, CSVs de uma execução anterior (ex.: de um mfcc_csvs.zip
    # mais antigo, com menos configurações) continuam na pasta e se
    # misturam com os novos, fazendo a mesma configuração aparecer
    # duas vezes na lista abaixo — e ser treinada duas vezes.
    if os.path.isdir(EXTRACT_DIR):
        shutil.rmtree(EXTRACT_DIR)
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    with zipfile.ZipFile(MFCC_ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

    csv_paths = sorted(glob.glob(os.path.join(EXTRACT_DIR, "**", "*.csv"), recursive=True))

    if not csv_paths:
        raise FileNotFoundError(
            f"Nenhum arquivo .csv encontrado dentro de '{MFCC_ZIP_PATH}'. Confira se o "
            f"zip realmente contém os CSVs gerados por extrair_mfcc_multiplas_configuracoes.py."
        )

    # Segunda camada de proteção: mesmo já limpando EXTRACT_DIR acima,
    # deduplica por (n_mfcc, max_time) para o caso de o próprio .zip
    # conter mais de um arquivo para a mesma configuração (ex.: um
    # arquivo antigo e um novo com o mesmo nome, em subpastas diferentes
    # dentro do zip). Mantém apenas o primeiro encontrado e avisa.
    configuracoes_por_chave = {}
    for csv_path in csv_paths:
        nome = os.path.basename(csv_path)
        match = PADRAO_NOME.search(nome)

        if not match:
            print(f"Não consegui identificar N_MFCC/MAX_TIME em '{nome}'; pulando este arquivo.")
            continue

        n_mfcc = int(match.group(1))
        max_time = int(match.group(2))
        chave = (n_mfcc, max_time)

        if chave in configuracoes_por_chave:
            print(
                f"Configuração N_MFCC={n_mfcc}, MAX_TIME={max_time} encontrada mais de uma "
                f"vez no zip (ex.: '{configuracoes_por_chave[chave]['csv_path']}' e '{csv_path}'). "
                f"Usando apenas o primeiro arquivo e ignorando o(s) repetido(s)."
            )
            continue

        configuracoes_por_chave[chave] = {"csv_path": csv_path, "n_mfcc": n_mfcc, "max_time": max_time}

    configuracoes = sorted(configuracoes_por_chave.values(), key=lambda c: (c["n_mfcc"], c["max_time"]))
    return configuracoes


# ==========================================================
# 4. Treinar UMA configuração
# ==========================================================

def treinar_configuracao(encoder, csv_path, n_mfcc, max_time, indice, total):
    """
    Carrega um CSV de MFCC, treina a DNN com early stopping, salva o
    melhor modelo daquela configuração localmente e retorna um
    dicionário com as métricas finais.
    """
    print(f"\n{'=' * 70}")
    print(f"[{indice}/{total}] Configuração: N_MFCC={n_mfcc} | MAX_TIME={max_time}")
    print(f"{'=' * 70}")

    model_path = os.path.join(OUTPUT_DIR, f"melhor_modelo_nmfcc{n_mfcc}_maxtime{max_time}.pth")

    # ---- Carregar dados ----
    df = pd.read_csv(csv_path)
    mfcc_cols = [c for c in df.columns if c.startswith("mfcc_")]

    esperado = n_mfcc * max_time
    if len(mfcc_cols) != esperado:
        raise ValueError(
            f"'{csv_path}' tem {len(mfcc_cols)} colunas de MFCC, mas o nome do "
            f"arquivo indica N_MFCC*MAX_TIME = {esperado}."
        )

    X = df[mfcc_cols].to_numpy(dtype=np.float32).reshape(-1, n_mfcc, max_time)
    X = torch.from_numpy(X)
    y = torch.tensor(encoder.transform(df["label"]))

    print(f"Amostras: {len(X)} | Formato de cada matriz de MFCC: ({n_mfcc}, {max_time})")

    # ---- Split treino / validação / teste ----
    idx_train_full, idx_test = train_test_split(
        np.arange(len(X)), test_size=0.2, random_state=42, stratify=y
    )
    idx_train, idx_val = train_test_split(
        idx_train_full, test_size=0.2, random_state=42, stratify=y[idx_train_full]
    )

    train_loader = DataLoader(
        VoiceDataset(X[idx_train], y[idx_train]), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(VoiceDataset(X[idx_val], y[idx_val]), batch_size=BATCH_SIZE)
    test_loader = DataLoader(VoiceDataset(X[idx_test], y[idx_test]), batch_size=BATCH_SIZE)

    # ---- Modelo, critério, otimizador ----
    model = DNN(input_size=n_mfcc * max_time, num_classes=len(encoder.classes_)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ---- Treinamento com early stopping ----
    train_loss_history = []
    val_loss_history = []
    best_val_loss = float("inf")
    epochs_sem_melhora = 0

    for epoch in range(EPOCHS_MAX):
        model.train()
        total_train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        train_loss_history.append(total_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_val_loss += loss.item()
        val_loss_history.append(total_val_loss)

        melhorou = total_val_loss < (best_val_loss - MIN_DELTA)
        if melhorou:
            best_val_loss = total_val_loss
            epochs_sem_melhora = 0
            torch.save(model.state_dict(), model_path)
            marcador = "Novo melhor modelo salvo"
        else:
            epochs_sem_melhora += 1
            marcador = f"sem melhora ({epochs_sem_melhora}/{PATIENCE})"

        print(
            f"  Epoch {epoch + 1}/{EPOCHS_MAX} | Loss treino: {total_train_loss:.4f} | "
            f"Loss val: {total_val_loss:.4f} | {marcador}"
        )

        if epochs_sem_melhora >= PATIENCE:
            print(f"  Convergência detectada na época {epoch + 1}. Parando esta configuração.")
            break

    # ---- Avaliação final com o melhor modelo desta configuração ----
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    predictions, targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
            targets.extend(y_batch.numpy())

    test_acc = accuracy_score(targets, predictions)
    print(f"  Acurácia no teste (melhor modelo desta configuração): {test_acc:.4f}")

    return {
        "n_mfcc": n_mfcc,
        "max_time": max_time,
        "csv_path": csv_path,
        "model_path": model_path,
        "best_val_loss": best_val_loss,
        "test_acc": test_acc,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "predictions": predictions,
        "targets": targets,
    }


# ==========================================================
# 5. Comparação final entre configurações + gráficos
# ==========================================================

def salvar_comparacao(resultados):
    tabela = sorted(
        [
            [r["n_mfcc"], r["max_time"], r["n_mfcc"] * r["max_time"], r["test_acc"], r["best_val_loss"]]
            for r in resultados
        ],
        key=lambda linha: linha[3],
        reverse=True
    )

    headers = ["N_MFCC", "MAX_TIME", "Nº features", "Acuracia_teste", "Melhor_loss_val"]

    print("\n" + tabulate(
        [[a, b, c, f"{d:.4f}", f"{e:.4f}"] for a, b, c, d, e in tabela],
        headers=headers, tablefmt="github"
    ))

    tabela_csv_path = os.path.join(OUTPUT_DIR, "comparacao_configuracoes.csv")
    pd.DataFrame(tabela, columns=headers).to_csv(tabela_csv_path, index=False)
    print(f"\nTabela comparativa salva em: {tabela_csv_path}")

    # Gráfico de barras comparando a acurácia de cada configuração
    rotulos = [f"{r['n_mfcc']}x{r['max_time']}" for r in resultados]
    acuracias = [r["test_acc"] for r in resultados]
    ordem = np.argsort(acuracias)[::-1]
    rotulos = [rotulos[i] for i in ordem]
    acuracias = [acuracias[i] for i in ordem]

    plt.figure(figsize=(9, 5))
    plt.bar(rotulos, acuracias, color="#4C72B0")
    plt.ylim(0, 1)
    plt.xlabel("Configuração (N_MFCC x MAX_TIME)")
    plt.ylabel("Acurácia no teste")
    plt.title("Comparação entre configurações de MFCC")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    grafico_path = os.path.join(OUTPUT_DIR, "comparacao_configuracoes.png")
    plt.savefig(grafico_path, dpi=150)
    plt.close()
    print(f"Gráfico comparativo salvo em: {grafico_path}")


def salvar_diagnostico_melhor(melhor, encoder):
    # Matriz de confusão
    cm = confusion_matrix(melhor["targets"], melhor["predictions"])
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", linewidths=0.5, linecolor="lightgray",
        cbar_kws={"label": "Quantidade"},
        xticklabels=encoder.classes_, yticklabels=encoder.classes_
    )
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title(f"Matriz de Confusão — N_MFCC={melhor['n_mfcc']}, MAX_TIME={melhor['max_time']}")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "matriz_confusao_melhor.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # Relatório de classificação (também salvo em .txt)
    relatorio = classification_report(melhor["targets"], melhor["predictions"], target_names=encoder.classes_)
    print("\nRelatório de classificação (melhor configuração):\n")
    print(relatorio)
    relatorio_path = os.path.join(OUTPUT_DIR, "classification_report_melhor.txt")
    with open(relatorio_path, "w", encoding="utf-8") as f:
        f.write(relatorio)

    # Curva de treinamento
    plt.figure(figsize=(8, 6))
    plt.plot(melhor["train_loss_history"], label="Loss treino")
    plt.plot(melhor["val_loss_history"], label="Loss validação")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.title(f"Curva de Treinamento — N_MFCC={melhor['n_mfcc']}, MAX_TIME={melhor['max_time']}")
    plt.legend()
    plt.tight_layout()
    curva_path = os.path.join(OUTPUT_DIR, "curva_treinamento_melhor.png")
    plt.savefig(curva_path, dpi=150)
    plt.close()

    print(f"\nMatriz de confusão salva em: {cm_path}")
    print(f"Relatório de classificação salvo em: {relatorio_path}")
    print(f"Curva de treinamento salva em: {curva_path}")


# ==========================================================
# 6. Rotina principal
# ==========================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Dispositivo usado para o treino: {device}")

    configuracoes = extrair_zip_e_localizar_csvs()
    print(f"\n{len(configuracoes)} configuração(ões) encontrada(s):\n")
    for c in configuracoes:
        print(f"  - N_MFCC={c['n_mfcc']:>3} | MAX_TIME={c['max_time']:>4} | {os.path.basename(c['csv_path'])}")

    # Encoder compartilhado entre todas as configurações: como todas
    # vêm do mesmo dataset (só muda o jeito de calcular o MFCC), os
    # comandos são os mesmos em todos os CSVs. Por isso ele é
    # ajustado (fit) uma única vez, a partir do primeiro CSV.
    df_referencia = pd.read_csv(configuracoes[0]["csv_path"], usecols=["label"])
    encoder = LabelEncoder()
    encoder.fit(df_referencia["label"])

    encoder_path = os.path.join(OUTPUT_DIR, "encoder.joblib")
    dump(encoder, encoder_path)

    print(f"\nClasses aprendidas pelo encoder ({len(encoder.classes_)}):")
    for c in encoder.classes_:
        print(" -", c)
    print(f"Encoder salvo em: {encoder_path}")

    # Treina uma DNN para cada configuração encontrada
    resultados = []
    for indice, config in enumerate(configuracoes, start=1):
        resultado = treinar_configuracao(
            encoder, config["csv_path"], config["n_mfcc"], config["max_time"],
            indice, len(configuracoes)
        )
        resultados.append(resultado)

    print(f"\n\nTreinamento concluído para {len(resultados)} configuração(ões).")

    # Comparação final + gráficos
    salvar_comparacao(resultados)

    melhor = max(resultados, key=lambda r: r["test_acc"])
    print(
        f"\nMelhor configuração geral: N_MFCC={melhor['n_mfcc']}, MAX_TIME={melhor['max_time']} "
        f"-> acurácia no teste = {melhor['test_acc']:.4f}"
    )

    melhor_geral_path = os.path.join(OUTPUT_DIR, "melhor_modelo_geral.pth")
    shutil.copy(melhor["model_path"], melhor_geral_path)
    print(f"Cópia do melhor modelo salva em: {melhor_geral_path}")

    salvar_diagnostico_melhor(melhor, encoder)

    print(f"\nTudo pronto! Resultados completos disponíveis em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
