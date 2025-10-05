import argparse
import numpy as np
import pandas as pd
from statistics import mean, stdev
from sklearn.model_selection import StratifiedKFold, cross_val_score
from src.utils import load_data, save_artifact, save_results_to_csv, create_dirs, rank_features
from src.preprocess import get_features_target
from src.model import create_full_pipeline
from src.config import BEST_MODEL_PATH, RESULTS_CSV_PATH, RANDOM_STATE, N_SPLITS, ARTIFACTS_DIR, REPORTS_DIR

def train_and_cross_validate(data_path: str, k_folds: int, seed: int):
    """
    Treina e avalia o modelo utilizando Stratified K-Fold CV, salvando
    o modelo final e os resultados das métricas.
    """
    create_dirs([ARTIFACTS_DIR, REPORTS_DIR])

    print(f"Carregando dados de: {data_path}")
    df = load_data(data_path)
    if df is None:
        return

    X, y = get_features_target(df)
    rank_features(X.fillna(X.median()), y) # Ranqueamento preliminar
    
    pipeline = create_full_pipeline()
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    
    print(f"Iniciando a Validação Cruzada Estratificada ({k_folds} Folds)...")

    # --- 1. Calcular Acurácia por Fold ---
    # Usando cross_val_score para simplificar e garantir a correta aplicação do Pipeline
    accuracy_scores = cross_val_score(
        pipeline, X, y, cv=skf, scoring='accuracy', n_jobs=-1
    )
    
    # Simular o treinamento final para obter o modelo 'best.pkl'
    # Na prática, treinamos o pipeline com todos os dados para o artefato final
    pipeline.fit(X, y)
    best_pipeline = pipeline 

    # --- 2. Calcular ROC AUC e PR AUC ---
    roc_scores = cross_val_score(pipeline, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    pr_scores = cross_val_score(pipeline, X, y, cv=skf, scoring="average_precision", n_jobs=-1)

    # --- 3. Apresentar e Salvar Resultados ---
    
    mean_accuracy = np.mean(accuracy_scores)
    std_dev = np.std(accuracy_scores) # Usando numpy para consistência
    
    print("\n--- Resumo da Validação Cruzada ---")
    print(f"Lista de acurácias: {accuracy_scores.tolist()}")
    print(f"\nAcurácia Máxima: {max(accuracy_scores)*100:.2f} %")
    print(f"Acurácia Mínima: {min(accuracy_scores)*100:.2f} %")
    print(f"Acurácia Média Global: {mean_accuracy*100:.2f} %")
    print(f"Desvio Padrão: {std_dev:.4f}")

    print(f"\nROC AUC Média: {roc_scores.mean():.4f}")
    print(f"ROC AUC Std Dev: {roc_scores.std():.4f}")
    
    print(f"PR Média: {pr_scores.mean():.4f}")
    print(f"PR Std Dev: {pr_scores.std():.4f}")

    save_results_to_csv(accuracy_scores.tolist(), RESULTS_CSV_PATH)
    save_artifact(best_pipeline, BEST_MODEL_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Treina o Classificador de Potabilidade de Água usando Stratified K-Fold CV."
    )
    parser.add_argument(
        "--data",
        type=str,
        default='data/raw/water.csv', # Default para execução no root
        help="Caminho para o arquivo CSV de dados de entrada."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=N_SPLITS,
        help="Número de folds (K) para a Validação Cruzada Estratificada."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE,
        help="Seed aleatória para reprodutibilidade."
    )
    # O argumento '--out' é aceito, mas não tem efeito, pois os caminhos são fixos em config.py
    parser.add_argument(
        "--out",
        type=str,
        default='reports/',
        help="Diretório de saída (caminho fixo em config.py)."
    )
    args = parser.parse_args()

    train_and_cross_validate(args.data, args.k, args.seed)
