# src/evaluate.py
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from yellowbrick.classifier import ConfusionMatrix
from src.utils import load_data, load_artifact, create_dirs
from src.preprocess import get_features_target
from src.config import BEST_MODEL_PATH, FIGURES_DIR, RAW_DATA_PATH

def evaluate_model(model_path: str, data_path: str, output_dir: str):
    """
    Carrega o modelo treinado e os dados, realiza a avaliação final
    e salva as figuras de performance.
    """
    create_dirs([output_dir])

    # 1. Carregar Modelo e Dados
    print(f"Carregando modelo de: {model_path}")
    pipeline = load_artifact(model_path)

    print(f"Carregando dados de: {data_path}")
    df = load_data(data_path)
    if df is None:
        return

    X, y = get_features_target(df)

    # 2. Previsões e Métricas
    y_pred = pipeline.predict(X)
    
    # Relatório de Classificação
    print("\n--- Relatório de Classificação Final ---")
    print(classification_report(y, y_pred))

    # 3. Gerar e Salvar Figuras (Matriz de Confusão)
    fig_path = os.path.join(output_dir, 'confusion_matrix.png')
    
    print(f"Gerando Matriz de Confusão e salvando em {fig_path}...")
    
    # O pipeline já tem o fit completo do train_cv.py, mas Yellowbrick precisa
    # que chamemos o fit/score novamente, ou passamos os dados X/y brutos.
    # O Yellowbrick executa o fit/transform internamente na Pipeline para a visualização.
    
    classes = ['not potable', 'potable'] # Conforme o rótulo no script original
    
    plt.figure(figsize=(8, 6))
    cm = ConfusionMatrix(
        pipeline,
        classes=classes
    )
    
    # Fit e Score para gerar a matriz
    cm.fit(X, y) # Treina o visualizador, aplicando o Pipeline internamente
    cm.score(X, y) # Avalia
    
    cm.ax.set_title("Matriz de Confusão do Modelo Final")
    cm.poof(outpath=fig_path, clear=True)
    plt.close()
    
    print("Avaliação concluída. Artefatos visuais salvos.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Avalia o modelo treinado e salva as figuras de performance."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=BEST_MODEL_PATH,
        help="Caminho para o artefato do modelo treinado (e.g., artifacts/best.pkl)."
    )
    parser.add_argument(
        "--data",
        type=str,
        default='data/raw/water.csv', # Default para execução no root
        help="Caminho para o arquivo CSV de dados usado para a avaliação."
    )
    parser.add_argument(
        "--out",
        type=str,
        default=FIGURES_DIR,
        help="Diretório de saída para salvar as figuras de performance."
    )
    args = parser.parse_args()
    
    evaluate_model(args.model, args.data, args.out)
