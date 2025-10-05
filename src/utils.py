import os
import pickle
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from src.config import ARTIFACTS_DIR, REPORTS_DIR, FEATURE_COLUMNS

def create_dirs(base_dir_list: list[str]):
    """Garante que os diretórios de saída existam."""
    for dir_path in base_dir_list:
        os.makedirs(dir_path, exist_ok=True)

def load_data(data_path: str) -> pd.DataFrame | None:
    """Carrega o dataset a partir do caminho especificado."""
    try:
        # A suposição é que o caminho 'data_path' leva ao 'water.csv'
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        print(f"Erro: Arquivo de dados não encontrado em {data_path}")
        return None

def save_artifact(artifact, path: str):
    """Salva um objeto Python (ex: modelo) em um arquivo pickle."""
    with open(path, 'wb') as file:
        pickle.dump(artifact, file)
    print(f"Artefato salvo em: {path}")

def load_artifact(path: str):
    """Carrega um objeto Python (ex: modelo) de um arquivo pickle."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artefato não encontrado em {path}")
    with open(path, 'rb') as file:
        return pickle.load(file)

def save_results_to_csv(results_list: list[float], path: str):
    """Salva uma lista de resultados (ex: acurácias do CV) em um arquivo CSV."""
    df_results = pd.DataFrame({'accuracy': results_list})
    df_results.to_csv(path, index=False)
    print(f"Resultados de acurácia salvos em: {path}")

def rank_features(X: pd.DataFrame, y: pd.Series):
    """
    Executa e imprime o ranqueamento das features usando SelectKBest (f_classif),
    conforme o script original.
    """
    print("\n--- Ranqueamento de Features (SelectKBest f_classif) ---")
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X, y)
    
    feature_scores = dict(zip(X.columns, selector.scores_))
    ranked_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    
    for feature, score in ranked_features:
        print(f"{feature}: {score:.2f}")
    print("-" * 50)
